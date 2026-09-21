import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from utils import images_to_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate dataset-wide BDCT sensitivity map")
    parser.add_argument('--data_root', type=str, required=True, help='Path to training image directory (ImageFolder format)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader worker processes')
    parser.add_argument('--out_pth', type=str, default='sensitivity.pt', help='Output path for sensitivity tensor')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to perform BDCT and tensor reductions')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    print(f"Loading images from {args.data_root}...")
    dataset = datasets.ImageFolder(args.data_root, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    max_vals = None
    min_vals = None

    print(f"Computing empirical element-wise min and max across {len(dataset)} images on {device}...")
    with torch.no_grad():
        for images, _ in tqdm(loader):
            images = images.to(device, non_blocking=True)
            
            dct_blocks = images_to_batch(images)

            # Reduce along batch dimension
            batch_max = dct_blocks.amax(dim=0)
            batch_min = dct_blocks.amin(dim=0)

            if max_vals is None:
                max_vals = batch_max
                min_vals = batch_min
            else:
                max_vals = torch.maximum(max_vals, batch_max)
                min_vals = torch.minimum(min_vals, batch_min)

    sensitivity = (max_vals - min_vals).cpu()

    # safeguard against zero variance
    sensitivity = torch.clamp(sensitivity, min=1e-5)

    print(f"Calculated sensitivity tensor shape: {sensitivity.shape}")
    print(f"Sensitivity Stats -> Min: {sensitivity.min().item():.4f}, Max: {sensitivity.max().item():.4f}, Mean: {sensitivity.mean().item():.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_pth)), exist_ok=True)
    torch.save(sensitivity, args.out_pth+"sensitivity.pt")
    print(f"Saved sensitivity map to {args.out_pth}")


if __name__ == '__main__':
    main()