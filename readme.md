# Speed Up Model

## 1. Setup 
### 1.1. Hardware
- Acer Nitro 5
- Ram 8GB
- GTX 1650 4GB
- Nvidia-driver: 470 - cuda 11.4
### 1.2. Install Some package
Install [cuda-toolkit 11.4](https://developer.nvidia.com/cuda-11-4-1-download-archive)
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.4.1/local_installers/cuda-repo-ubuntu2004-11-4-local_11.4.1-470.57.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-4-local_11.4.1-470.57.02-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-4-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```
Install [cuDNN-8.2.4](https://developer.nvidia.com/cudnn)
```
sudo dpkg -i libcudnn8_8.2.4.15-1+cuda11.4_amd64.deb
sudo dpkg -i libcudnn8-dev_8.2.4.15-1+cuda11.4_amd64.deb
# Verify install cuDNN
cat /usr/include/x86_64-linux-gnu/cudnn_v*.h | grep CUDNN_MAJOR -A 2 
```
Install [TensorRT-8.2.0](https://developer.nvidia.com/tensorrt)
```
sudo dpkg -i nv-tensorrt-repo-ubuntu2004-cuda11.4-trt8.2.0.6-ea-20210922_1-1_amd64.deb
sudo apt update
sudo apt install tensorrt
pip install nvidia-pyindex
pip install nvidia-tensorrt
```
Add PATH to /.bashrc
```
nano ~/.bashrc
export CUDA_HOME=/usr/local/cuda
export DYLD_LIBRARY_PATH=$CUDA_HOME/lib64:$DYLD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
export C_INCLUDE_PATH=$CUDA_HOME/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_RUN_PATH=$CUDA_HOME/lib64:$LD_RUN_PATH
```
## 2. Torch model 
### 2.1. Test inference with image
```
    python3 torch_model/inference.py --imgpath <image path>
``` 
### 2.2. Test performance with batch size
```
    python3 torch_model/test.py \
    --batchsize 4 \
    --n_loop 100
```

## 3. Using ONNX
### 3.1. Convert model pytorch to ONNX 
```
    python3 torch_model/convert.py \
    --path_torch <image torch model> \
    --path_onnx <name onnx model>
``` 
### 3.2. Test performance with batch size
```
    python3 torch_model/infernce.py --imgpath <image path>
```

### 3.3. Test performance with batch size
```
    python3 onnx_model/test.py \
    --batchsize 4 \
    --n_loop 100
```