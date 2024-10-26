FROM qbxlvnf11docker/depth_anything2:trt_8.6.1.6
ENV PATH="/workspace/Depth-Anything/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0/TensorRT-8.6.1.6/bin:$PATH" 
ENV LD_LIBRARY_PATH="/workspace/Depth-Anything/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH" 
CMD ["bash"]