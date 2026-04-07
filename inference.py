import os
from openai import OpenAI
from tasks import easy, medium, hard
from grader import grade

# Initialize OpenAI client using environment variables
client = OpenAI(
    base_url=os.getenv("API_BASE_URL"),
    api_key=os.getenv("HF_TOKEN")
)

# Function to get action from LLM
def get_action(state):
    try:
        response = client.chat.completions.create(
            model=os.getenv("MODEL_NAME"),
            messages=[
                {"role": "system", "content": "You are a task scheduling agent."},
                {"role": "user", "content": f"Current state: {state}. Return the best action index (integer only)."}
            ]
        )
        return int(response.choices[0].message.content.strip())
    except:
        return 0  # fallback safe action


# Function to run each task
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

        # Normalize reward to 0.0–1.0
        normalized_reward = min(max(reward / 20, 0), 1)

        total_reward += normalized_reward
        step += 1

        print(f"[STEP] step={step} action={action} reward={normalized_reward}", flush=True)

    score = grade(total_reward)

    print(f"[END] task={name} score={score} steps={step}", flush=True)


# Main execution
if __name__ == "__main__":
    print("Running Inference on All Tasks...\n")

    run_task(easy, "Easy")
    run_task(medium, "Medium")
    run_task(hard, "Hard")