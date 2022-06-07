from utils import *
from configs.config import *
import onnxruntime
import numpy as np
import time

class ONNX_MODEL:
    def __init__(self, path_image: str=str(PATH_IMAGE_DEMO), path_onnx: str=str(PATH_MODEL_ONNX), path_torch: str=str(PATH_MODEL_TORCH)) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path_image = str(path_image)
        self.path_torch = path_torch
        self.path_onnx = path_onnx
        self.input = preprocess_image(path_image).to(self.device)
        self.ort_session = None

    def convert(self) -> None:
        model = torch.load(self.path_torch)
        model.eval().to(self.device)
        logger.info("===> build onnx ...")
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}
        torch.onnx.export(
            model,
            self.input,
            self.path_onnx,
            input_names=["input"],
            output_names=["output"],
            export_params=True,
            dynamic_axes=dynamic_axes
        )
        logger.info('✅ Convert model pytorch to onnx complete')

    def load(self) -> None:
        if self.device == 'cpu':
            self.ort_session = onnxruntime.InferenceSession(self.path_onnx, providers=["CPUExecutionProvider"])
        else:
            self.ort_session = onnxruntime.InferenceSession(self.path_onnx, providers=["CUDAExecutionProvider"])
        logger.info(f'Load onnx done with {self.device}')
    
    def inference(self, path_image: str) -> None:
        if self.ort_session is None:
            print('You need load onnx with onnx.load()')
        else:
            input = preprocess_image(path_image)
            input = to_numpy(input)
            ort_inputs =  {self.ort_session.get_inputs()[0].name: input}
            # Because the firt run have time-consuming for warm up system
            ort_outs = self.ort_session.run(None, ort_inputs)
            start = time.time()
            ort_outs = self.ort_session.run(None, ort_inputs)
            postprocess_onnx(ort_outs[0])
            logger.info(f'Time for inference: {time.time() - start: 0.6f} s')



def to_numpy(tensor: torch.tensor) -> np.array:
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def softmax(x: np.array) -> np.array:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def postprocess_onnx(output: np.array) -> None:
    # get class names
    with open(PATH_CLASSES) as f:
        classes = [line.strip() for line in f.readlines()]
    
    # calculate human-readable value by softmax
    confidences = softmax(output[0])*100

    # find top predicted classes
    indices = np.argsort(confidences)[::-1]
    i = 0
    # print the top classes predicted by the model
    while confidences[indices[i]] > 0.5:
        class_idx = indices[i]
        logger.info(
            f"class: {classes[class_idx]} , confidence: {confidences[class_idx].item(): 0.4f} %, index: {class_idx.item()}",
        )
        i += 1