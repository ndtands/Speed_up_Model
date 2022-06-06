from tensorrt_model.model import Resnet
from configs.config import *
import argparse

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description="Convert onnx to tensorRT")
    # Add an argument
    parser.add_argument('--path_trt', type=str, default=str(PATH_MODEL_TRT), help="path of trt model", required=False)
    parser.add_argument('--path_onnx', type=str, default=str(PATH_MODEL_ONNX), help="path of onnx model", required=False)
    parser.add_argument('--batch_size', type=int, default=32, help="batch size", required=False)
    # Parse the argument
    args = parser.parse_args()

    logger.info("Build tensorrt engine...")
    dynamic_shapes = {"input": ((1, 3, 224, 224), (args.batch_size, 3, 224, 224), (args.batch_size, 3, 224, 224))}
    Resnet.build_engine(
        onnx_file_path=args.path_onnx, 
        engine_file_path=args.path_trt,
        dynamic_shapes=dynamic_shapes,
        dynamic_batch_size=args.batch_size
        )
