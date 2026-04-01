import sys
import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Crucial: This tells Python to look in the folder above for env.py if app.py is in a subfolder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your environment logic
try:
    from env import TaskSchedulerEnv
except ImportError:
    # Fallback if env.py is in the same directory
    from env import TaskSchedulerEnv

app = FastAPI()
env = TaskSchedulerEnv()

class ActionRequest(BaseModel):
    action: int

@app.get("/")
def health():
    return {"status": "ok", "message": "OpenEnv Task Scheduler is Running"}

@app.post("/reset")
def reset():
    # OpenEnv expects the initial state/observation on reset
    return env.reset()

@app.post("/step")
def step(request: ActionRequest):
    # Using a Pydantic model (ActionRequest) makes the API more robust
    state, reward, done, info = env.step(request.action)
    return {
        "state": state, 
        "reward": reward, 
        "done": done, 
        "info": info
    }

# The OpenEnv Validator specifically looks for this main() function
def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
