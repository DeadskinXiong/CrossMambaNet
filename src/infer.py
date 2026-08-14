# infer.py
import torch, argparse, json, os
from torchvision import transforms
from PIL import Image
from torchvision.datasets import ImageFolder
import models_net_mamba
from util.misc import load_model
from torchvision.transforms import RandomErasing

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    RandomErasing(p=0.3, scale=(0.02, 0.05)),
    transforms.Normalize([0.5], [0.5]),
    transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x)),
])
'''
#原始transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='net_mamba_classifier')
    parser.add_argument('--resume', required=True, help='netmamba_flow_cls.pth')
    parser.add_argument('--img', required=True, help='unknown.png')
    parser.add_argument('--nb_classes', type=int, default=7)
    parser.add_argument('--byte_length', type=int, default=1600)
    # 关键：添加 use_cross_mamba 参数
    parser.add_argument('--use_cross_mamba', action='store_true',
                        help='Use CrossMamba1D instead of standard Mamba blocks')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 关键修改：传递 use_cross_mamba 参数
    model = models_net_mamba.__dict__[args.model](
        num_classes=args.nb_classes,
        byte_length=args.byte_length,
        use_cross_mamba=args.use_cross_mamba,  # 添加这行
    )

    # 加载权重
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    model.eval()

    # ---------- 动态读取类别映射 ----------
    train_dir = 'dataset/CrossPlatform-Android/dataset_sampled/train'
    assert os.path.isdir(train_dir), f'训练目录不存在：{train_dir}'
    tmp_dataset = ImageFolder(root=train_dir, transform=None)
    idx2name = {v: k for k, v in tmp_dataset.class_to_idx.items()}
    # --------------------------------------

    img = Image.open(args.img).convert('L')
    x = transform(img).unsqueeze(0).to(device)

    # 关键修改：CrossMamba1D 需要双视图输入，但推理时只有单张图
    # 方案：复制一份作为 x2，或者修改模型支持单视图推理
    # 这里采用复制方案，与 DualViewDataset 逻辑一致
    with torch.no_grad():
        if args.use_cross_mamba:
            # CrossMamba1D 需要 x2，复制 x 作为第二个视图
            logits = model(x, x2=x)
        else:
            logits = model(x)

        prob = torch.softmax(logits, dim=1)
        pred_idx = logits.argmax(1).item()

    print(json.dumps({
        'predicted_class_index': pred_idx,
        'predicted_class_name': idx2name[pred_idx],
        'probabilities': {idx2name[i]: float(p) for i, p in enumerate(prob.cpu().squeeze().tolist())}
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()