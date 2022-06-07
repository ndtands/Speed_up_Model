import time
from configs.config import logger, PATH_IMAGE_DEMO
from triton_model import TritonClient
from triton_model.model_client import RESNET
from utils import *
import argparse

triton_client = TritonClient().triton_client

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference for tensorRT")
    # Add an argument
    parser.add_argument('--path_image', type=str, default=str(PATH_IMAGE_DEMO), help="path of image", required=False)
    # Parse the argument
    args = parser.parse_args()
    
    logger.info(">>>>> Triton server <<<<<")
    resnet_client = RESNET(
        triton_client=triton_client,
        model_name='model_onnx'
        )
    logger.info("✅ Load triton client done")
    input = preprocess_image(args.path_image).numpy()

    #First time for warm up
    resnet_client.infer(input0_data=input)

    start = time.time()
    output = resnet_client.infer(input0_data=input)
    total_time = time.time() - start
    postprocess(torch.tensor(output))
    logger.info(f'Time for inference: {time.time() - start: 0.6f} s')