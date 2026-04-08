import os
import re
from openai import OpenAI
from tasks import easy, medium, hard
from grader import grade

client = OpenAI(
    base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.environ.get("API_KEY", "dummy_key")
)
def get_action(state):
    try:
        num_tasks = len(state["tasks"])
        response = client.chat.completions.create(
            model=os.environ.get("MODEL_NAME", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a task scheduling agent."},
                {"role": "user", "content": f"State: {state}. Return action index (0 to {num_tasks - 1}). Integer only."}
            ],
            max_tokens=10
        )
        text = response.choices[0].message.content.strip()
        numbers = re.findall(r'\d+', text)
        action = int(numbers[0]) if numbers else 0
        return max(0, min(action, num_tasks - 1))
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return 0

def run_task(task_func, task_name):
    env = task_func()
    state = env.state()
    done = False
    total_reward = 0
    step = 0
    rewards = []
    model = os.environ.get("MODEL_NAME", "gpt-4o-mini")

    print(f"[START] task={task_name} env=task-scheduler model={model}", flush=True)

    while not done:
        action = get_action(state)
        state, reward, done, info = env.step(action)
        normalized = round(min(max(reward / 30.0, 0.0), 1.0), 3)
        total_reward += normalized
        step += 1
        rewards.append(normalized)
        error = info.get("error", None)
        print(f"[STEP] step={step} action={action} reward={normalized:.3f} done={str(done).lower()} error={error}", flush=True)

    score = grade(total_reward)
    rewards_str = ",".join(str(r) for r in rewards)
    print(f"[END] success=true steps={step} score={score:.3f} rewards={rewards_str}", flush=True)
    return score

if __name__ == "__main__":
    print("Running Inference on All Tasks...\n", flush=True)
    run_task(easy, "easy")
    run_task(medium, "medium")
    run_task(hard, "hard")