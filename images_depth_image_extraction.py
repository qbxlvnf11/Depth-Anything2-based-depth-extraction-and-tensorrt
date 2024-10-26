import argparse
import math
import time
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from PIL import Image

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--dep_folder_path', type=str, default=None)
    parser.add_argument('--rgb_dep_folder_path', type=str, default=None)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    
    args = parser.parse_args()
    
    margin_width = 50

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(args.encoder)).to(DEVICE).eval()
    # print(f'depth_anything: {depth_anything}')

    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    if os.path.isfile(args.image_path):
        if args.image_path.endswith('txt'):
            with open(args.image_path, 'r') as f:
                lines = f.read().splitlines()
        else:
            filenames = [args.image_path]
    else:
        filenames = os.listdir(args.image_path)
        filenames = [os.path.join(args.image_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()
    
    for k, filename in enumerate(filenames):
        frame_id = k+1
        # print(' ---- Progress {:}/{:},'.format(k+1, len(filenames)), 'Processing', filename)
        
        raw_frame = Image.open(filename)
        raw_frame = np.array(raw_frame)
        frame_height, frame_width = raw_frame.shape[:2]

        # filename = os.path.basename(filename)
        file_name_with_ext = os.path.basename(filename)
        filename, _ = os.path.splitext(file_name_with_ext)

        if k % 100 == 0:
            print(' ---- Progress {:}/{:},'.format(frame_id, len(filenames)), 'Processing', filename)
            print(f' ---- Frame width: {frame_width}, height: {frame_height}')

        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB) / 255.0
        
        frame = transform({'image': frame})['image']
        frame = torch.from_numpy(frame).unsqueeze(0).to(DEVICE)

        start = time.time()
        with torch.no_grad():
            depth = depth_anything(frame)
        end = time.time()
        # print(f"Inf time: {end - start:.5f} sec")
        
        depth = F.interpolate(depth[None], (frame_height, frame_width), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        
        depth = depth.cpu().numpy().astype(np.uint8)
        if True: #args.save_depth_frame:
            depth_img = Image.fromarray(depth)
            # file_name = os.path.splitext(args.video_path)[0]
            if args.dep_folder_path is None:
                dep_folder = f"{filename}_depth"
            else:
                dep_folder = args.dep_folder_path
            os.makedirs(dep_folder, exist_ok=True)
            depth_img.save(os.path.join(dep_folder, filename + f'.png'))

        depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        if True: #args.save_depth_frame:
            depth_color_rgb = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
            depth_color_img = Image.fromarray(depth_color_rgb)
            # file_name = os.path.splitext(args.video_path)[0]
            if args.rgb_dep_folder_path is None:
                rgb_folder = f"{filename}_depth_rgb"
            else:
                rgb_folder = args.rgb_dep_folder_path
            os.makedirs(rgb_folder, exist_ok=True)
            depth_color_img.save(os.path.join(rgb_folder, filename + f'.png'))
        
        split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
        combined_frame = cv2.hconcat([raw_frame, split_region, depth_color])