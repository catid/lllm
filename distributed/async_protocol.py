from pydantic import BaseModel
from typing import List, Dict, Any

class Register(BaseModel):
    version: int
    worker_uuid: str
    gpu_list: List[str]

class RegisterResponse(BaseModel):
    accepted: bool

class WorkItem(BaseModel):
    work_item_id: str
    tokens: List[int]
    num_steps: int
    learning_rate: float
    batch_size: int

class WeightUpdate(BaseModel):
    worker_id: str
    work_item_id: str
    weight_updates: Dict[str, List[float]]

class Ping(BaseModel):
    text: str

class Pong(BaseModel):
    text: str
