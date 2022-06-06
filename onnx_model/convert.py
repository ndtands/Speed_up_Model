from onnx_model import ONNX_MODEL
from configs.config import PATH_MODEL_ONNX, PATH_MODEL_TORCH
import argparse


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--path_torch', type=str, default=str(PATH_MODEL_TORCH), help="path of pretrain model", required=False)
    parser.add_argument('--path_onnx', type=str, default=str(PATH_MODEL_ONNX), help="path of pretrain model", required=False)


    # Parse the argument
    args = parser.parse_args()
    onnx = ONNX_MODEL(
        path_torch=args.path_torch,
        path_onnx=args.path_onnx
    )
    onnx.convert()