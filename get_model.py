import torch
from torchvision import models
from configs.config import *

if __name__ == "__main__":
    model = models.resnet50(pretrained=True)
    torch.save(model,PATH_MODEL_TORCH)
    logging.info('✅ Save pretrain done')
