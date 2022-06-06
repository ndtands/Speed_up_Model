# Speed Up Model

## 1. Setup 

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