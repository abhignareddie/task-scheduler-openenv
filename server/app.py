import sys
import os
from fastapi import FastAPI
from pydantic import BaseModel

# Crucial: This tells Python to look in the folder above for env.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import TaskSchedulerEnv

app = FastAPI()
env = TaskSchedulerEnv()

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    return env.reset()

@app.post("/step")
def step(action: int):
    state, reward, done, info = env.step(action)
    return {"state": state, "reward": reward, "done": done, "info": info}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)