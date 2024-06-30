import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException

from async_protocol import WorkerCapabilities, WorkItem, WeightUpdate

class Orchestrator:
    def __init__(self):
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

    def run(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8000, ssl_keyfile="ed25519_key.pem", ssl_certfile="ed25519_cert.pem")

orchestrator = Orchestrator()
orchestrator.run()
