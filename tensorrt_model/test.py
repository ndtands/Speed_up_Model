from tensorrt_model.model import Resnet
from configs.config import *
import argparse
import numpy as np
import time


if __name__ =='__main__':
    parser = argparse.ArgumentParser(description="Test perfomance for tensorRT")
    # Add an argument
    parser.add_argument('--path_trt', type=str, default=str(PATH_MODEL_TRT), help="path of trt model", required=False)
    parser.add_argument('--batch_size', type=int, default=32, help="batch size", required=False)
    parser.add_argument('--n_loop', type=int, default=100, help="num loop", required=False)
    # Parse the argument
    args = parser.parse_args()

    net = Resnet(args.path_trt)
    logger.info("✅ Load tensorrt engine done")

    input = np.float32(np.random.rand(args.batch_size,3,224,224))
    start = time.time()
    for _ in range(args.n_loop):
        net(input)
    total_time = time.time() - start
    logger.info(f"""
    Batch size: {args.batch_size} 
    Time inferene with {args.n_loop} loop is {total_time:0.4f} s
    >>>> About: {total_time/args.n_loop:.6f} s per 1 loop
    """)