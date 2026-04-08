import sys
import os
import re
import uvicorn
from fastapi import FastAPI
from openai import OpenAI

# Fix import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import TaskSchedulerEnv

# ✅ Initialize app and environment
app = FastAPI()
env = TaskSchedulerEnv()

# ✅ Initialize OpenAI client using injected variables
client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"]
)

# ✅ LLM function
def get_action(state):
    try:
        print("Calling LLM from /step...", flush=True)

        num_tasks = len(state["tasks"])

        response = client.chat.completions.create(
            model=os.environ.get("MODEL_NAME", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a task scheduling agent."},
                {"role": "user", "content": f"State: {state}. Return action index (0 to {num_tasks - 1})."}
            ],
            max_tokens=10
        )

        text = response.choices[0].message.content.strip()

        # ✅ Safe parsing
        numbers = re.findall(r'\d+', text)
        action = int(numbers[0]) if numbers else 0

        # ✅ Ensure valid range
        action = max(0, min(action, num_tasks - 1))

        print(f"LLM Action: {action}", flush=True)

        return action

    except Exception as e:
        print("LLM error:", e, flush=True)
        return 0  # fallback safe action


# ✅ Health check
@app.get("/")
def health():
    return {"status": "ok", "message": "OpenEnv Task Scheduler is Running"}


# ✅ Reset endpoint
@app.post("/reset")
def reset():
    return env.reset()


# ✅ FINAL FIXED STEP ENDPOINT (NO INPUT PARAMETER)
@app.post("/step")
def step():
    state = env.state()

    # 🔥 LLM CALL (REQUIRED)
    action = get_action(state)

    state, reward, done, info = env.step(action)

    return {
        "state": state,
        "reward": reward,
        "done": done,
        "info": info
    }


# ✅ Entry point
def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()