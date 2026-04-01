import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel
from env import TaskSchedulerEnv

app = FastAPI()
env = TaskSchedulerEnv()

class StepRequest(BaseModel):
    action: int

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Task Scheduler is running"}

@app.post("/reset")
def reset():
    state = env.reset()
    return state

@app.get("/state")
def get_state():
    return env.state()

@app.post("/step")
def step(request: StepRequest):
    state, reward, done, info = env.step(request.action)
    return {"state": state, "reward": reward, "done": done, "info": info}

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()