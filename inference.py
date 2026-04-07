import os
from openai import OpenAI
from tasks import easy, medium, hard
from grader import grade

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "dummy-token")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

def run_task(task_func, name):
    env = task_func()
    state = env.state()
    done = False
    total_reward = 0
    step = 0

    print(f"[START] task={name}", flush=True)

    while not done:
        action = 0
        state, reward, done, _ = env.step(action)
        total_reward += reward
        step += 1
        print(f"[STEP] step={step} reward={reward}", flush=True)

    score = grade(total_reward)
    print(f"[END] task={name} score={score} steps={step}", flush=True)

if __name__ == "__main__":
    print("Running Inference on All Tasks...\n")
    run_task(easy, "Easy")
    run_task(medium, "Medium")
    run_task(hard, "Hard")