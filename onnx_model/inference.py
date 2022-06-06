from onnx_model import ONNX_MODEL
from configs.config import PATH_IMAGE_DEMO
import argparse


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description="inference with model converted to ONNX")
    # Add an argument
    parser.add_argument('--path_image', type=str, default=str(PATH_IMAGE_DEMO), help="path of pretrain model", required=False)


    # Parse the argument
    args = parser.parse_args()

    # Create object onnx
    onnx = ONNX_MODEL()
    # Load model
    onnx.load()
    # Run inference
    onnx.inference(path_image=args.path_image)