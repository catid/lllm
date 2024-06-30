import requests
import io
import numpy as np

import torch
import numpy as np

def copy_model_weights_to_numpy(model):
    numpy_weights = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            numpy_weights[name] = param.detach().cpu().float().numpy().copy()

    return numpy_weights



class Client:
    def __init__(self, orchestrator_url: str, cert_path: str):
        self.orchestrator_url = orchestrator_url
        self.cert_path = cert_path
        self.session = requests.Session()

    def PostJson(self,
                 endpoint: str,
                 data: Dict[str, Any] = None) -> requests.Response:
        url = f"{self.orchestrator_url}/{endpoint}"
        try:
            response = self.session.post(url, json=data)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return None

    def PostBinary(self,
             endpoint: str,
             binary_data: np.ndarray = None) -> requests.Response:
        url = f"{self.orchestrator_url}/{endpoint}"
        try:
            # Convert numpy array to bytes
            buffer = io.BytesIO()
            np.save(buffer, binary_data, allow_pickle=False)
            binary_payload = buffer.getvalue()
            
            headers = {'Content-Type': 'application/octet-stream'}
            response = self.session.post(url, data=binary_payload, headers=headers)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return None
