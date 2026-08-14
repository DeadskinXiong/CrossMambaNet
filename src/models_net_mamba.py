import torch
import torch.nn as nn
from timm.models.layers import DropPath
from models_mamba import Block, RMSNorm, rms_norm_fn, StrideEmbed
from timm.models.layers import trunc_normal_, lecun_normal_
import math
from functools import partial
from cross_mamba_1d import CrossMamba1D    #新加
# 关键：导入标准 Mamba 给 Decoder 用
from mamba_ssm.modules.mamba_simple import Mamba  # 添加这行

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.Conv2d, nn.Conv1d)):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class NetMamba(nn.Module):
    def __init__(
        self,
        byte_length=1600,
        stride_size=4,
        in_chans=1,
        embed_dim=192,
        depth=4,
        decoder_embed_dim=128,
        decoder_depth=2,
        num_classes=1000,
        norm_pix_loss=False,
        drop_rate=0.,
        drop_path_rate=0.1,
        bimamba_type="none",
        is_pretrain=False,
        device=None,
        dtype=None,
        use_cross_mamba=False,
        view_mode="dual",
        mask_mode="shared",
        **kwargs
    ):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.is_pretrain = is_pretrain
        self.stride_size = stride_size
        self.use_cross_mamba = use_cross_mamba  # 新加

        self.view_mode = view_mode
        self.mask_mode = mask_mode

        valid_view_modes = {"single", "same", "dual"}
        valid_mask_modes = {"shared", "independent"}

        if self.view_mode not in valid_view_modes:
            raise ValueError(
                f"Unsupported view_mode={self.view_mode}. "
                f"Expected one of {valid_view_modes}."
            )

        if self.mask_mode not in valid_mask_modes:
            raise ValueError(
                f"Unsupported mask_mode={self.mask_mode}. "
                f"Expected one of {valid_mask_modes}."
            )

        if self.use_cross_mamba and self.view_mode == "single":
            raise ValueError(
                "CrossMamba1D requires a second input. "
                "Use view_mode='same' or view_mode='dual'."
            )

        print(
            "[Ablation configuration] "
            f"use_cross_mamba={self.use_cross_mamba}, "
            f"view_mode={self.view_mode}, "
            f"mask_mode={self.mask_mode}"
        )

        # --------------------------------------------------------------------------
        # NetMamba encoder specifics
        self.patch_embed = StrideEmbed(byte_length, stride_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_cls_token = 1
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + num_cls_token, embed_dim))
        # Mamba blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList([
            create_block(
                embed_dim,
                ssm_cfg=None,
                norm_epsilon=1e-5,
                rms_norm=True,
                residual_in_fp32=True,
                fused_add_norm=True,
                layer_idx=i,
                if_bimamba=False,
                bimamba_type=bimamba_type,
                drop_path=inter_dpr[i],
                if_devide_out=True,
                init_layer_scale=None,
                use_cross_mamba=use_cross_mamba,  # 新加
            )  for i in range(depth)])
        self.norm_f = RMSNorm(embed_dim, eps=1e-5)
        # --------------------------------------------------------------------------

        if is_pretrain:
            # --------------------------------------------------------------------------
            # NetMamba decoder specifics
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + num_cls_token, decoder_embed_dim))
            decoder_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, decoder_depth)]  # stochastic depth decay rule
            decoder_inter_dpr = [0.0] + decoder_dpr
            self.decoder_blocks = nn.ModuleList([
                create_block(
                    decoder_embed_dim,
                    ssm_cfg=None,
                    norm_epsilon=1e-5,
                    rms_norm=True,
                    residual_in_fp32=True,
                    fused_add_norm=True,
                    layer_idx=i,
                    if_bimamba=False,
                    bimamba_type=bimamba_type,
                    drop_path=decoder_inter_dpr[i],
                    if_devide_out=True,
                    init_layer_scale=None,
                )
                for i in range(decoder_depth)])
            self.decoder_norm_f = RMSNorm(decoder_embed_dim, eps=1e-5)
            self.decoder_pred = nn.Linear(decoder_embed_dim, stride_size * in_chans, bias=True)  # decoder to stride
            # --------------------------------------------------------------------------
        else:
            # --------------------------------------------------------------------------
            # NetMamba classifier specifics
            self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
            # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.patch_embed.apply(segm_init_weights)
        if not self.is_pretrain:
            self.head.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        if self.is_pretrain:
            trunc_normal_(self.decoder_pos_embed, std=.02)
            trunc_normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(partial(_init_weights, n_layer=depth,))
        
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}
    
    def stride_patchify(self, imgs):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size**2 *1)
        """
        B, C, H, W = imgs.shape
        assert C == 1, "Input images should be grayscale"
        stride_size = self.stride_size
        x = imgs.reshape(B, H*W // stride_size, stride_size)
        return x

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [B N D], sequence
        """
        B, N, D = x.shape  # batch, length, dim
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1) # ids_restore[i] = i-th noise element's rank

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) # x_masked are acctually non-masked elements

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, if_mask=True, x2=None):
        """
        x: [B, 1, H, W] - 主视图
        x2: [B, 1, H, W] - 辅助视图（可选）
        """
        B, C, H, W = x.shape

        # Step 1: Patch Embedding
        x = self.patch_embed(x.reshape(B, C, -1))  # (B, 400, 256)

        if x2 is not None:
            B2, C2, H2, W2 = x2.shape
            x2 = self.patch_embed(x2.reshape(B2, C2, -1))  # (B, 400, 256)

        # Step 2: Add Position Embedding（都不含 cls_token）
        x = x + self.pos_embed[:, :-1, :]
        if x2 is not None:
            x2 = x2 + self.pos_embed[:, :-1, :]

        # Step 3: Masking（关键修改）
        if if_mask:
            B, N, D = x.shape

            if x2 is None:
                # 配置A：标准单视图随机Mask
                x, mask, ids_restore = self.random_masking(
                    x,
                    mask_ratio
                )

            elif self.mask_mode == "shared":
                # 配置B和D：两个视图使用同一组保留位置
                len_keep = int(N * (1 - mask_ratio))

                noise = torch.rand(B, N, device=x.device)

                ids_shuffle = torch.argsort(noise, dim=1)
                ids_restore = torch.argsort(ids_shuffle, dim=1)
                ids_keep = ids_shuffle[:, :len_keep]

                x = torch.gather(
                    x,
                    dim=1,
                    index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
                )

                x2 = torch.gather(
                    x2,
                    dim=1,
                    index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
                )

                mask = torch.ones(
                    [B, N],
                    device=x.device
                )
                mask[:, :len_keep] = 0
                mask = torch.gather(
                    mask,
                    dim=1,
                    index=ids_restore
                )

            elif self.mask_mode == "independent":
                # 配置C：两个视图分别独立采样Mask
                # Decoder仍然使用x1的mask和ids_restore
                x, mask, ids_restore = self.random_masking(
                    x,
                    mask_ratio
                )

                x2, _, _ = self.random_masking(
                    x2,
                    mask_ratio
                )

            else:
                raise RuntimeError(
                    f"Unexpected mask_mode: {self.mask_mode}"
                )

        else:
            mask = None
            ids_restore = None

        # Step 4: Add CLS Token（两者都加，保持同步）
        cls_token = self.cls_token + self.pos_embed[:, -1, :]
        cls_tokens = cls_token.expand(B, -1, -1)

        x = torch.cat([x, cls_tokens], dim=1)  # (B, 41, 256)

        if x2 is not None:
            x2 = torch.cat([x2, cls_tokens], dim=1)  # (B, 41, 256)

        x = self.pos_drop(x)

        # Step 5: Apply CrossMamba1D Blocks
        residual = None
        hidden_states = x

        for blk in self.blocks:
            # CrossMamba1D 接收 x1=hidden_states, x2=x2
            hidden_states, residual = blk(hidden_states, residual, x2=x2)

        # Final norm
        fused_add_norm_fn = rms_norm_fn
        x = fused_add_norm_fn(
            self.drop_path(hidden_states),
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True,
        )

        if if_mask:
            return x, mask, ids_restore
        else:
            return x

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        visible_tokens = x[:, :-1, :]
        x_ = torch.cat([visible_tokens, mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x_, x[:, -1:, :]], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Mamba blocks
        residual = None
        hidden_states = x
        for blk in self.decoder_blocks:
            hidden_states, residual = blk(hidden_states, residual)
        fused_add_norm_fn = rms_norm_fn
        x = fused_add_norm_fn(
            self.drop_path(hidden_states),
            self.decoder_norm_f.weight,
            self.decoder_norm_f.bias,
            eps=self.decoder_norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True,
        )

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, :-1, :]
        return x

    def forward_rec_loss(self, imgs, pred, mask):
        """
        imgs: [N, 1, H, W]
        pred: [N, L, p*p*1]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.stride_patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.9, x2=None):
        # imgs是主视图x1
        B, C, H, W = imgs.shape
        assert C == 1, "Input images should be grayscale"

        if self.view_mode == "single":
            # 配置A：完全不使用第二视图
            x2_used = None

        elif self.view_mode == "same":
            # 配置B：第二分支读取与主分支完全相同的张量
            x2_used = imgs

        elif self.view_mode == "dual":
            # 配置C和D：使用Dataset生成的独立增强视图
            if x2 is None:
                raise ValueError(
                    "view_mode='dual' requires x2, but x2 is None."
                )
            x2_used = x2

        else:
            raise RuntimeError(f"Unexpected view_mode: {self.view_mode}")

        if self.is_pretrain:
            latent, mask, ids_restore = self.forward_encoder(
                imgs,
                mask_ratio=mask_ratio,
                x2=x2_used
            )

            pred = self.forward_decoder(latent, ids_restore)

            # 始终只重建主视图imgs，即x1
            loss = self.forward_rec_loss(imgs, pred, mask)

            return loss, pred, mask

        x = self.forward_encoder(
            imgs,
            mask_ratio=mask_ratio,
            if_mask=False,
            x2=x2_used
        )

        return self.head(x[:, -1, :])
        

def net_mamba_pretrain(**kwargs):
    model = NetMamba(
        is_pretrain=True, stride_size=4, embed_dim=256, depth=4,
        decoder_embed_dim=128, decoder_depth=2, **kwargs)
    return model

def net_mamba_classifier(**kwargs):
    model = NetMamba(
        is_pretrain=False, stride_size=4, embed_dim=256, depth=4,
        **kwargs)
    return model

def net_mamba_bl400_pretrain(**kwargs):
    model = NetMamba(
        is_pretrain=True, stride_size=4, embed_dim=256, depth=4,
        decoder_embed_dim=128, decoder_depth=2, 
        byte_length=400, **kwargs)
    return model

def net_mamba_bl400_classifier(**kwargs):
    model = NetMamba(
        is_pretrain=False, stride_size=4, embed_dim=256, depth=4,
        byte_length=400, **kwargs)
    return model

def net_mamba_bl800_pretrain(**kwargs):
    model = NetMamba(
        is_pretrain=True, stride_size=4, embed_dim=256, depth=4,
        decoder_embed_dim=128, decoder_depth=2, 
        byte_length=800, **kwargs)
    return model

def net_mamba_bl800_classifier(**kwargs):
    model = NetMamba(
        is_pretrain=False, stride_size=4, embed_dim=256, depth=4,
        byte_length=800, **kwargs)
    return model

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    if_bimamba=False,
    bimamba_type="none",
    if_devide_out=False,
    init_layer_scale=None,
    use_cross_mamba=False,      # 新增 1
):
    # 1. 先统一把 factory_kwargs 准备好
    factory_kwargs = {"device": device, "dtype": dtype}

    if use_cross_mamba:
        mixer_cls = partial(CrossMamba1D, d_state=16, d_conv=4, expand=2)
    else:
        if if_bimamba:
            bimamba_type = "v1"
        if ssm_cfg is None:
            ssm_cfg = {}
        # factory_kwargs 已定义，直接复用
        mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba_type=bimamba_type,
                            if_devide_out=if_devide_out, init_layer_scale=init_layer_scale,
                            **ssm_cfg, **factory_kwargs)
    # 下面原封不动照抄原来的代码
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block