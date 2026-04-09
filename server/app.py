import sys
import os
import re
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, Any
from openai import OpenAI

# Fix import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks import easy, medium, hard

# Initialize app
app = FastAPI(title="Task Scheduler OpenEnv")

# Store multiple environments
environments: Dict[str, Any] = {}

# Initialize OpenAI client using injected variables
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

# Validate API configuration
if not API_BASE_URL or not API_KEY:
    print("[ERROR] API_BASE_URL and API_KEY must be set", flush=True)
    # Don't exit, just log for server mode

print(f"[INFO] API_BASE_URL: {API_BASE_URL}", flush=True)
print(f"[INFO] MODEL_NAME: {MODEL_NAME}", flush=True)

# Initialize client if credentials exist
client = None
if API_BASE_URL and API_KEY:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )

# Request models
class StepRequest(BaseModel):
    action: int

class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"

# LLM function
def get_action(state):
    if client is None:
        # Fallback if no client
        tasks = state.get("tasks", [])
        if tasks:
            best_idx = 0
            best_score = -999
            for i, task in enumerate(tasks):
                score = task.get('priority', 0) / max(task.get('duration', 1), 1)
                if score > best_score:
                    best_score = score
                    best_idx = i
            return best_idx
        return 0
    
    try:
        print("Calling LLM...", flush=True)
        
        tasks = state.get("tasks", [])
        if not tasks:
            return 0
            
        num_tasks = len(tasks)

        # Build task description
        task_desc = []
        for i, task in enumerate(tasks):
            task_desc.append(f"Task {i}: priority={task.get('priority', 0)}, deadline={task.get('deadline', 0)}, duration={task.get('duration', 0)}")
        
        prompt = f"""Current tasks:
{chr(10).join(task_desc)}

Choose the best task index (0 to {num_tasks - 1}) to schedule next.
Return ONLY the number, nothing else."""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a task scheduling agent. Return only a number."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.0
        )

        text = response.choices[0].message.content.strip()
        
        # Safe parsing
        numbers = re.findall(r'\d+', text)
        action = int(numbers[0]) if numbers else 0
        
        # Ensure valid range
        action = max(0, min(action, num_tasks - 1))

        print(f"LLM Action: {action}", flush=True)
        return action

    except Exception as e:
        print(f"LLM error: {e}", flush=True)
        # Fallback heuristic
        tasks = state.get("tasks", [])
        if tasks:
            best_idx = 0
            best_score = -999
            for i, task in enumerate(tasks):
                score = task.get('priority', 0) / max(task.get('duration', 1), 1)
                if score > best_score:
                    best_score = score
                    best_idx = i
            return best_idx
        return 0

# Health check endpoints
@app.get("/")
def root():
    return {"status": "ok", "message": "OpenEnv Task Scheduler is Running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Reset endpoint
@app.post("/reset")
def reset(request: ResetRequest = None):
    if request is None:
        request = ResetRequest()
    
    env_id = request.task_id or "easy"
    
    # Create environment based on task_id
    if env_id == "easy":
        env = easy()
    elif env_id == "medium":
        env = medium()
    elif env_id == "hard":
        env = hard()
    else:
        env = easy()
    
    # Store it
    environments[env_id] = env
    
    state = env.state()
    
    return {
        "env_id": env_id,
        "state": state
    }

# Step endpoint with env_id
@app.post("/step/{env_id}")
def step_endpoint(env_id: str, request: StepRequest):
    if env_id not in environments:
        raise HTTPException(status_code=404, detail="Environment not found")
    
    env = environments[env_id]
    
    # Take step with provided action
    next_state, reward, done, info = env.step(request.action)
    
    return {
        "state": next_state,
        "reward": reward,
        "done": done,
        "info": info
    }

# State endpoint - REQUIRED by OpenEnv
@app.get("/state/{env_id}")
def get_state(env_id: str):
    if env_id not in environments:
        raise HTTPException(status_code=404, detail="Environment not found")
    
    env = environments[env_id]
    state = env.state()
    
    return {"state": state}

# Default step endpoint (backward compatible)
@app.post("/step")
def step_default(request: StepRequest):
    env_id = "easy"
    if env_id not in environments:
        env = easy()
        environments[env_id] = env
    
    return step_endpoint(env_id, request)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
