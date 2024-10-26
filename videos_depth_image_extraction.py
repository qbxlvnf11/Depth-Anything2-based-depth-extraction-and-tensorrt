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
    parser.add_argument('--video_path', type=str)
    parser.add_argument('--video_out_dir', type=str, default='./vis_video_depth')
    parser.add_argument('--dep_folder_path', type=str, default=None)
    parser.add_argument('--rgb_dep_folder_path', type=str, default=None)
    parser.add_argument('--raw_img_folder_path', type=str, default=None)
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--save_raw_img', action='store_true')
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

    if os.path.isfile(args.video_path):
        if args.video_path.endswith('txt'):
            with open(args.video_path, 'r') as f:
                lines = f.read().splitlines()
        else:
            filenames = [args.video_path]
    else:
        filenames = os.listdir(args.video_path)
        filenames = [os.path.join(args.video_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()
    
    if args.save_video:
        os.makedirs(args.video_out_dir, exist_ok=True)
    
    for k, filename in enumerate(filenames):
        print(' ---- Progress {:}/{:},'.format(k+1, len(filenames)), 'Processing', filename)
        
        raw_video = cv2.VideoCapture(filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        print(f' ---- Frame width: {frame_width}, height: {frame_height}, rate: {frame_rate}')
        output_width = frame_width * 2 + margin_width
        
        # filename = os.path.basename(filename)
        file_name_with_ext = os.path.basename(filename)
        filename, _ = os.path.splitext(file_name_with_ext)

        if args.save_video:
            output_path = os.path.join(args.video_out_dir, filename + '_video_depth.mp4')
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))
        
        frame_id = 1
        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()

            if not ret:
                break

            if args.save_raw_img:
                raw_frame_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                raw_frame_img = Image.fromarray(raw_frame_rgb)
                # file_name = os.path.splitext(args.video_path)[0]
                if args.raw_img_folder_path is None:
                    img_folder = f"{filename}_img"
                else:
                    img_folder = args.raw_img_folder_path
                os.makedirs(img_folder, exist_ok=True)
                raw_frame_img.save(os.path.join(img_folder, filename + f'_{frame_id}.png'))
            
            if frame_id % 100 == 0:
                print(f'ID: {frame_id}')

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
                depth_img.save(os.path.join(dep_folder, filename + f'_{frame_id}.png'))

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
                depth_color_img.save(os.path.join(rgb_folder, filename + f'_{frame_id}.png'))
            
            split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
            combined_frame = cv2.hconcat([raw_frame, split_region, depth_color])
            
            if args.save_video:
                out.write(combined_frame)
            frame_id += 1
            
        raw_video.release()
        out.release()