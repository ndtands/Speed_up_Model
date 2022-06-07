import tritonclient.grpc as grpcclient
import numpy as np

class RESNET:
    def __init__(
        self,
        triton_client,
        batch_size = 1,
        model_name='model_onnx',
        inputs_name=None,
        outputs_name=None,
        inputs_shape=None,
        inputs_type=None            
    ):
        if inputs_name is None:
            inputs_name = ['input']
        
        if outputs_name is None:
            outputs_name = ['output']
        
        if inputs_shape is None:
            inputs_shape = [[batch_size, 3, 224, 224]]

        if inputs_type is None:
            inputs_type = ['FP32']
        

        self.triton_client = triton_client
        self.model_name = model_name
        self.inputs_name = inputs_name
        self.outputs_name = outputs_name
        self.inputs_shape = inputs_shape
        self.inputs_type = inputs_type

        self.inputs = []
        self.outputs = []
        for i in range(len(inputs_name)):
            self.inputs.append(grpcclient.InferInput(inputs_name[i], inputs_shape[i], inputs_type[i]))
        self.outputs.append(grpcclient.InferRequestedOutput(outputs_name[0]))

    def infer(self, input0_data: np.ndarray) -> np.ndarray:
        # Initialize the data
        self.inputs[0].set_data_from_numpy(input0_data)

        # Test with outputs
        results = self.triton_client.infer(
                    model_name=self.model_name,
                    inputs=self.inputs,
                    outputs=self.outputs,
                    headers={'test': '1'}
                )
        logit = results.as_numpy('output')
        return logit

    def predict(self, image_path: str) -> None:
        pass