import torch
import numpy as np
from configs.config import *
import time
import argparse

 # Create device
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = torch.load(PATH_MODEL_TORCH)
model.eval().to(device)

def test_speed(batch: int, loop: int) -> None:
    input = torch.tensor(np.random.rand(batch,3,224,224)).float().cuda()
    start = time.time()
    for _ in range(loop):
        model(input)
    total_time = time.time() - start
    logger.info(f"""
    Batch size: {batch} 
    Time inferene with {loop} loop is {total_time:0.4f} s
    >>>> About: {total_time/loop:.6f} s per 1 loop
    """)
if __name__ == '__main__':
     # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--batchsize', type=int, default=4, help="batch size")
    parser.add_argument('--n_loop', type=int, default=100, help="batch size")
    
    # Parse the argument
    args = parser.parse_args()
    test_speed(args.batchsize,args.n_loop)