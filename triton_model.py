import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype
import os

triton_ip = '0.0.0.0'
triton_port = 8000


class TritonInference:
    def __init__(self, url, model_name):
        self.triton_client = httpclient.InferenceServerClient(
            url=url, verbose=False, concurrency=4, max_greenlets=1)
        self.model_name = model_name
        self.model_config = self.triton_client.get_model_metadata(
            model_name=self.model_name, model_version='1')
        self.inputs, self.outputs, self.outputs_name = self.create_io()
    def create_io(self):
        inputs = [httpclient.InferInput(i.get('name'), [j  for j in i.get('shape')],
                                        i.get('datatype')) for i in self.model_config.get('inputs')]
        outputs = [httpclient.InferRequestedOutput(
            i.get('name')) for i in self.model_config.get('outputs')]
        outputs_name = [i.get('name')
                        for i in self.model_config.get('outputs')]
        return inputs, outputs, outputs_name

    def inference(self, images):
        triton_client = httpclient.InferenceServerClient(
            url=f'{triton_ip}:{triton_port}', verbose=False, concurrency=4)
        inputs = []
        inputs.append(httpclient.InferInput('images', images.shape, 'FP32'))
        inputs[0].set_data_from_numpy(images)
        result = triton_client.async_infer(self.model_name,
                                           inputs,
                                           model_version='1',
                                           outputs=self.outputs)
        result = result.get_result()
        return [result.as_numpy(name) for name in self.outputs_name]
