from tensorrt_model.model import Resnet
from configs.config import *
import argparse
import numpy as np
from utils import *
import time


if __name__ =='__main__':
    parser = argparse.ArgumentParser(description="Inference for tensorRT")
    # Add an argument
    parser.add_argument('--path_image', type=str, default=str(PATH_IMAGE_DEMO), help="path of image", required=False)
    # Parse the argument
    args = parser.parse_args()

    net = Resnet(PATH_MODEL_TRT)
    logger.info(">>>>> TensorRt engine <<<<<")
    logger.info("✅ Load tensorrt engine done")
    input = preprocess_image(args.path_image).numpy()
    
    start = time.time()
    outputs = net(np.ascontiguousarray(input))
    result = Resnet.postprocess(outputs)
    postprocess(torch.from_numpy(result[0]).reshape(-1, 1000))
    logger.info(f'Time for inference: {time.time() - start: 0.6f} s')