import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import torch
import torch.nn as nn
from collections import deque
import random

class WorkerCapabilities(BaseModel):
    worker_id: str
    gpu_list: List[str]

class WorkItem(BaseModel):
    work_item_id: str
    model: Dict[str, Any]  # Simplified for this example
    tokens: List[int]
    num_steps: int
    learning_rate: float
    batch_size: int

class WeightUpdate(BaseModel):
    worker_id: str
    work_item_id: str
    weight_updates: Dict[str, List[float]]

class AsyncDiLoCo:
    def __init__(self, model, num_workers, buffer_size, outer_lr, inner_lr):
        self.model = model
        self.num_workers = num_workers
        self.buffer_size = buffer_size
        self.outer_optimizer = torch.optim.SGD(self.model.parameters(), lr=outer_lr, momentum=0.9, nesterov=True)
        self.inner_lr = inner_lr
        self.buffer = deque(maxlen=buffer_size)
        self.global_step = 0

    def apply_update(self, weight_update):
        self.buffer.append(weight_update)
        if len(self.buffer) == self.buffer_size:
            self.apply_delayed_nesterov_update()
        else:
            self.apply_sgd_update(weight_update)
        self.global_step += 1

    def apply_delayed_nesterov_update(self):
        avg_update = self.average_updates(self.buffer)
        self.outer_optimizer.zero_grad()
        for name, param in self.model.named_parameters():
            param.grad = avg_update[name]
        self.outer_optimizer.step()
        self.buffer.clear()

    def apply_sgd_update(self, weight_update):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.add_(torch.tensor(weight_update[name]), alpha=-self.inner_lr)

    @staticmethod
    def average_updates(updates):
        avg_update = {}
        for name in updates[0].keys():
            avg_update[name] = torch.mean(torch.stack([torch.tensor(u[name]) for u in updates]), dim=0)
        return avg_update

class Orchestrator:
    def __init__(self, model, num_workers, buffer_size, outer_lr, inner_lr):
        self.app = FastAPI()
        self.diloco = AsyncDiLoCo(model, num_workers, buffer_size, outer_lr, inner_lr)
        self.workers = {}
        self.work_items = {}
        self.setup_routes()

    def setup_routes(self):
        @self.app.post("/register")
        async def register_worker(capabilities: WorkerCapabilities):
            self.workers[capabilities.worker_id] = capabilities
            return {"status": "registered"}

        @self.app.get("/get_work")
        async def get_work():
            work_item = self.create_work_item()
            return work_item

        @self.app.post("/submit_result")
        async def submit_result(weight_update: WeightUpdate):
            self.diloco.apply_update(weight_update.weight_updates)
            new_work_item = self.create_work_item()
            return new_work_item

        @self.app.get("/ping")
        async def ping():
            return {"status": "pong"}

    def create_work_item(self):
        work_item_id = f"work_{random.randint(0, 1000000)}"
        work_item = WorkItem(
            work_item_id=work_item_id,
            model=self.diloco.model.state_dict(),
            tokens=self.get_next_tokens(),  # Implement this method
            num_steps=50,  # You might want to make this dynamic
            learning_rate=self.diloco.inner_lr,
            batch_size=32  # You might want to make this configurable
        )
        self.work_items[work_item_id] = work_item
        return work_item

    def get_next_tokens(self):
        # Implement this method to return the next batch of tokens
        pass

    def run(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8000, ssl_keyfile="ed25519_key.pem", ssl_certfile="ed25519_cert.pem")

# Example usage
model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 2))  # Example model
orchestrator = Orchestrator(model, num_workers=4, buffer_size=4, outer_lr=0.1, inner_lr=0.01)
orchestrator.run()
