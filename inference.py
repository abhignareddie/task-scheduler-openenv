import os
from openai import OpenAI
from tasks import easy, medium, hard
from grader import grade

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = os.getenv("API_KEY", HF_TOKEN)
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

def get_action(state):
    try:
        prompt = f"You are a task scheduler agent. Given the current state: {state}, choose an action as an integer (0 or 1). Reply with only the integer."
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10
        )
        action = int(response.choices[0].message.content.strip())
        return action
    except Exception as e:
        print(f"LLM call failed: {e}, using default action 0", flush=True)
        return 0

def run_task(task_func, name):
    env = task_func()
    state = env.state()
    done = False
    total_reward = 0
    step = 0

    print(f"[START] task={name}", flush=True)

    while not done:
        action = get_action(state)
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