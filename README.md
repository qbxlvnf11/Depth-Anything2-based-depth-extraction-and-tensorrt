Contents
=============

#### - Depth-Anything2-based generalized depth extraction

#### - Depth-Anything2 TensorRT test


Structures of Project Folders
=============

#### - [Download TensorRT 8.6.1.6](https://developer.nvidia.com/nvidia-tensorrt-8x-download)

```
${CODE_ROOT}
            |   |-- run_video.py
            |   |-- run_video_trt.py
            |   |-- images_depth_image_extraction.py
            |   |-- videos_depth_image_extraction.py
            |   |-- Depth_Anythingv2_TensorRT_python
            |   |   |   |-- TensorRT-8.6.1.6
            |   |   |   |   |   |-- onnx_graphsurgeon
            |   |   |   |   |   |-- ...
            |   |-- TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0
            |   |   |   |-- checkpoints
            |   |   |   |   |   |-- depth_anything_v2_vitl.onnx
            |   |   |   |   |   |-- depth_anything_v2_vitl_fp16.engine
            |   |   |   |   |   |-- ...
            |   |-- ...
```


Docker Environments
=============

#### - Build docker environment
  
```
docker build -t depth_anything_trt_env .
```

#### - Run docker environment

```
docker run -it --gpus all --name {container_name} \
--shm-size=64G -p 8845:8845 -e GRANT_SUDO=yes --user root \
-v {root_folder}:/workspace/Depth-Anything \
-w /workspace/Depth-Anything depth_anything_trt_env:latest bash
```


Depth-Anything2 ONNX & TensorRT Engine
=============

#### - [ONNX Download](https://huggingface.co/yuvraj108c/Depth-Anything-2-Onnx/tree/main)

#### - [TensorRT Engine Sample Download](http://naver.me/GMmjTDNi)
   
   - Password: 1234

#### - Convert ONNX to TensorRT Engine

```
cd Depth_Anythingv2_TensorRT_python
python tools/onnx2trt.py \
--mode {fp16|fp32} \
-o {onnx_path} \
--output {trt_engine_path} \
```


Depth Extraction
=============

#### - Depth extraction from videos

```
python videos_depth_image_extraction.py \
--encoder {vitl|vitb|vits} \
--video_path {input_video_path} \
--dep_folder_path {depth_save_path} \
--rgb_dep_folder_path {rgb_depth_save_path} \
--raw_img_folder_path {raw_image_save_path} \
--video_out_dir {video_save_path} \
--save_raw_img \
--save_video
```

#### - Depth extraction from images

```
python videos_depth_image_extraction.py \
--encoder {vitl|vitb|vits} \
--image_path {input_folder_path} \
--dep_folder_path {depth_save_path} \
--rgb_dep_folder_path {rgb_depth_save_path}
```


Inference
=============

#### - Inference video (Pytorch)
  
```
python run_video.py \
--encoder {vitl|vitb|vits} \
--video_path {input_video_path}
--outdir {video_save_path}
```

#### - Inference video (TensorRT)
  
```
python run_video_trt.py \
--engine {trt_engine_path} \
--video_path {input_video_path}
--outdir {video_save_path}
```


References
=============

#### - [Paper](https://arxiv.org/abs/2406.09414)

#### - [Original Code](https://github.com/DepthAnything/Depth-Anything-V2)

#### - [TensorRT](https://github.com/zhujiajian98/Depth-Anythingv2-TensorRT-python)


Author
=============

#### - LinkedIn: https://www.linkedin.com/in/taeyong-kong-016bb2154

#### - Blog URL: https://blog.naver.com/qbxlvnf11

#### - Email: qbxlvnf11@google.com, qbxlvnf11@naver.com
