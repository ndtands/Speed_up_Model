
# configs/config.py
# Configurations.

import logging
import sys
from pathlib import Path
from pyrsistent import v
from rich.logging import RichHandler
import logging.config as lcfg

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
LOGS_DIR = Path(BASE_DIR, "logs")
CONFIG_DIR = Path(BASE_DIR, "configs")
MODELS_TORCH_DIR = Path(BASE_DIR, 'model/model_torch/')
MODELS_ONNX_DIR = Path(BASE_DIR, 'model_repository/model_onnx/1')
MODELS_TRT_DIR = Path(BASE_DIR, 'model/model_trt/')

# Create dirs
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_ONNX_DIR.mkdir(parents=True, exist_ok=True)
MODELS_TORCH_DIR.mkdir(parents=True, exist_ok=True)
MODELS_TRT_DIR.mkdir(parents=True, exist_ok=True)

#Path
PATH_CLASSES = Path(BASE_DIR, "imagenet_classes.txt")
PATH_MODEL_TORCH = Path(MODELS_TORCH_DIR,'model.pt')
PATH_MODEL_ONNX = Path(MODELS_ONNX_DIR,'model.onnx')
PATH_MODEL_TRT = Path(MODELS_TRT_DIR,'model.plan')
PATH_IMAGE_DEMO = Path(BASE_DIR,'img/dog.png')



# Logger
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "loggers": {
        "root": {
            "handlers": ["console", "info", "error"],
            "level": logging.INFO,
            "propagate": True,
        },
    },
}
lcfg.dictConfig(logging_config)
logger = logging.getLogger("root")
logger.handlers[0] = RichHandler(markup=True)

