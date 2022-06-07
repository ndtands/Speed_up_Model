import time
from configs.config import logger
from triton_model import TritonClient
from triton_model.model_client import RESNET
import numpy as np
import argparse

triton_client = TritonClient().triton_client

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test perfomance for triton server")
    # Add an argument
    parser.add_argument('--batch_size', type=int, default=4, help="batch size", required=False)
    parser.add_argument('--n_loop', type=int, default=100, help="num loop", required=False)
    
    # Parse the argument
    args = parser.parse_args()

    input = np.float32(np.random.rand(args.batch_size,3,224,224))
    resnet_client = RESNET(
        triton_client=triton_client,
        batch_size=args.batch_size,
        model_name='model_onnx'
        )
    logger.info("✅ Load triton client done")
    start = time.time()
    for _ in range(args.n_loop):
        resnet_client.infer(input0_data=input)
    total_time = time.time() - start
    logger.info(f"""
    Batch size: {args.batch_size} 
    Time inferene with {args.n_loop} loop is {total_time:0.4f} s
    >>>> About: {total_time/args.n_loop:.6f} s per 1 loop
    """)