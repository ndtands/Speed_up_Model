from utils import *
from configs.config import PATH_IMAGE_DEMO, PATH_MODEL_TORCH
from configs.config import logger
import argparse
import time

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description="Inference with pretrain model in pytorch")
    # Add an argument
    parser.add_argument('--imgpath', type=str, default=str(PATH_IMAGE_DEMO), help="path of image", required=False)

    # Parse the argument
    args = parser.parse_args()
    # Create device
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(">>>>> Pretrain model <<<<<")
    # preprocess_image
    input = preprocess_image(args.imgpath).to(device)
    # loading model
    model = torch.load(PATH_MODEL_TORCH)
    model.eval().to(device)

    # Inference
    start = time.time()
    output = model(input)
    postprocess(output_data=output)
    logger.info(f'Time for inference: {time.time() - start: 0.6f} s')