import torch
import numpy as np
from configs.config import *
from onnx_model import ONNX_MODEL
import time
import argparse

# Load model
onnx_model = ONNX_MODEL()
onnx_model.load()

def test_speed(batch: int, loop: int) -> None:
    x = np.float32(np.random.rand(batch,3,224,224))
    ort_inputs =  {onnx_model.ort_session.get_inputs()[0].name: x}
    start = time.time()
    for _ in range(loop):
        onnx_model.ort_session.run(None, ort_inputs)
    total_time = time.time() - start
    logger.info(f"""
    Batch size: {batch} 
    Time inferene with {loop} loop is {total_time:0.4f} s
    >>>> About: {total_time/loop:.6f} s per 1 loop
    """)
if __name__ == '__main__':
     # Create the parser
    parser = argparse.ArgumentParser(description="Test performance model with model converted to ONNX")
    # Add an argument
    parser.add_argument('--batchsize', type=int, default=4, help="batch size")
    parser.add_argument('--n_loop', type=int, default=100, help="batch size")
    
    # Parse the argument
    args = parser.parse_args()
    test_speed(args.batchsize,args.n_loop)