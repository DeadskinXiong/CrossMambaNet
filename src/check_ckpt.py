#!/usr/bin/env python3
import torch
import sys
sys.path.insert(0, 'src')          # 让 Python 找到 src 下的模块

from models_net_mamba import net_mamba_classifier

# ===== 1. 配置 =====
CKPT_PATH      = 'checkpoints/pre-train.pth'   # ← 改成你的文件
NUM_CLASSES    = 253                          # ← 改成你训练时的数量

# ===== 2. 实例化模型 =====
model = net_mamba_classifier(num_classes=NUM_CLASSES)
model_state = model.state_dict()

# ===== 3. 加载 checkpoint =====
ckpt = torch.load(CKPT_PATH, weights_only=False, map_location='cpu')
# 如果 checkpoint 是 dict['model'] 格式，先解开
if 'model' in ckpt:
    ckpt_state = ckpt['model']
elif 'state_dict' in ckpt:
    ckpt_state = ckpt['state_dict']
else:
    ckpt_state = ckpt

# ===== 4. 比对 =====
missing    = set(model_state.keys()) - set(ckpt_state.keys())
unexpected = set(ckpt_state.keys())  - set(model_state.keys())
common     = set(model_state.keys()) & set(ckpt_state.keys())

print('======== Missing (ckpt 缺少) ========')
for k in sorted(missing):
    print(k)

print('\n======== Unexpected (ckpt 多出) ========')
for k in sorted(unexpected):
    print(k)

print('\n======== Shape Mismatch ========')
for k in sorted(common):
    if model_state[k].shape != ckpt_state[k].shape:
        print(f'{k:50s}  model{model_state[k].shape} vs ckpt{ckpt_state[k].shape}')

print('\n======== Summary ========')
print(f'Total model keys : {len(model_state)}')
print(f'Total ckpt keys  : {len(ckpt_state)}')
print(f'Missing          : {len(missing)}')
print(f'Unexpected       : {len(unexpected)}')
print(f'Shape mismatch   : {sum(1 for k in common if model_state[k].shape != ckpt_state[k].shape)}')