# worker_node.py

import requests
import json
import time
import ssl
import torch
import multiprocessing as mp
from typing import Dict, Any
import subprocess
import sys
import uuid

class WorkerNode:
    def __init__(self, orchestrator_url: str, cert_path: str):
        self.worker_id = str(uuid.uuid4())
        self.orchestrator_url = orchestrator_url
        self.cert_path = cert_path
        self.session = self._create_secure_session()
        self.current_work_item = None
        self.training_process = None
        self.ipc_queue = mp.Queue()
        self.abort_event = mp.Event()

    def _create_secure_session(self):
        session = requests.Session()
        session.verify = self.cert_path
        return session

    def _make_request(self, endpoint: str, method: str = 'GET', data: Dict[str, Any] = None) -> requests.Response:
        url = f"{self.orchestrator_url}/{endpoint}"
        try:
            if method == 'GET':
                response = self.session.get(url)
            elif method == 'POST':
                response = self.session.post(url, json=data)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return None

    def register_capabilities(self) -> Dict[str, Any]:
        gpu_list = [str(gpu) for gpu in torch.cuda.get_device_properties(torch.cuda.current_device())]
        data = {
            "worker_id": self.worker_id,
            "gpu_list": gpu_list,
            "current_work_item": self.current_work_item,
            "progress": self.get_progress()
        }
        response = self._make_request("register", method='POST', data=data)
        if response:
            return response.json()
        return None

    def get_progress(self):
        if not self.training_process or not self.training_process.is_alive():
            return 0
        try:
            self.ipc_queue.put("GET_PROGRESS")
            return self.ipc_queue.get(timeout=1)
        except mp.queues.Empty:
            return 0

    def get_work_item(self) -> Dict[str, Any]:
        response = self._make_request("get_work")
        if response:
            self.current_work_item = response.json()
            return self.current_work_item
        return None

    def submit_result(self, weight_updates: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        if not self.current_work_item:
            raise ValueError("No current work item")
        
        data = {
            "worker_id": self.worker_id,
            "work_item_id": self.current_work_item["work_item_id"],
            "weight_updates": {k: v.tolist() for k, v in weight_updates.items()}
        }
        response = self._make_request("submit_result", method='POST', data=data)
        if response:
            new_work_item = response.json()
            if new_work_item.get("status") == "hold":
                return {"status": "hold"}
            self.current_work_item = new_work_item
            return self.current_work_item
        return None

    def run(self):
        while True:
            try:
                if not self.current_work_item:
                    work_item = self.get_work_item()
                    if not work_item:
                        print("Failed to get work item, retrying in 60 seconds")
                        time.sleep(60)
                        continue

                self.start_training_process()

                while self.training_process.is_alive():
                    response = self.register_capabilities()
                    if response and response.get("abort"):
                        self.abort_training()
                        break
                    time.sleep(5)

                if not self.abort_event.is_set():
                    weight_updates = self.ipc_queue.get()
                    result = self.submit_result(weight_updates)
                    if not result:
                        print("Failed to submit results, retrying in 60 seconds")
                        time.sleep(60)
                        continue

                    if result.get("status") == "hold":
                        print("Received hold signal, waiting for new work")
                        self.current_work_item = None
                        continue

                    self.current_work_item = result
                else:
                    self.abort_event.clear()
                    self.current_work_item = None

            except Exception as e:
                print(f"Worker script crashed: {e}")
                self.abort_training()
                time.sleep(60)

    def start_training_process(self):
        self.abort_event.clear()
        self.training_process = mp.Process(target=self._run_training_script, args=(self.current_work_item, self.ipc_queue, self.abort_event))
        self.training_process.start()

    def abort_training(self):
        if self.training_process and self.training_process.is_alive():
            self.abort_event.set()
            self.training_process.join(timeout=10)
            if self.training_process.is_alive():
                self.training_process.terminate()

    @staticmethod
    def _run_training_script(work_item, ipc_queue, abort_event):
        import training_script
        training_script.train(work_item, ipc_queue, abort_event)

if __name__ == "__main__":
    worker = WorkerNode(
        orchestrator_url="https://127.0.0.1:8000",
        cert_path="async/ed25519_cert.pem"
    )
    worker.run()
