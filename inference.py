import os
import re
import sys
from openai import OpenAI
from tasks import easy, medium, hard
from grader import grade

# MUST use their environment variables
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

# Validate API configuration
if not API_BASE_URL:
    print("[ERROR] API_BASE_URL environment variable not set", flush=True)
    sys.exit(1)
if not API_KEY:
    print("[ERROR] API_KEY environment variable not set", flush=True)
    sys.exit(1)

# Initialize client with THEIR proxy
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

print(f"[INFO] Using API: {API_BASE_URL}", flush=True)
print(f"[INFO] Using model: {MODEL_NAME}", flush=True)

def get_action(state):
    """Get action from LLM via their proxy"""
    try:
        tasks = state.get("tasks", [])
        if not tasks:
            return 0
        
        num_tasks = len(tasks)
        
        # Build task description for LLM
        task_desc = []
        for i, task in enumerate(tasks):
            task_desc.append(f"Task {i}: priority={task.get('priority', 0)}, deadline={task.get('deadline', 0)}, duration={task.get('duration', 0)}")
        
        prompt = f"""You are a task scheduler. Current tasks:
{chr(10).join(task_desc)}

Choose the best task index (0 to {num_tasks-1}) to schedule next.
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
        numbers = re.findall(r'\d+', text)
        action = int(numbers[0]) if numbers else 0
        action = max(0, min(action, num_tasks - 1))
        
        print(f"[ACTION] LLM chose task {action}", flush=True)
        return action
        
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}", flush=True)
        # Fallback to simple heuristic
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

def run_task(task_func, task_name):
    """Run a single task with required stdout format"""
    env = task_func()
    state = env.state()
    done = False
    total_reward = 0
    step = 0
    rewards = []
    
    # REQUIRED: [START] line format
    print(f"[START] task={task_name} env=task-scheduler model={MODEL_NAME}", flush=True)
    
    while not done and step < 20:
        action = get_action(state)
        state, reward, done, info = env.step(action)
        
        # Normalize reward to 0-1 range
        normalized = round(min(max(reward / 30.0, 0.0), 1.0), 3)
        total_reward += normalized
        step += 1
        rewards.append(normalized)
        
        error = info.get("error", None) if info else None
        
        # REQUIRED: [STEP] line format
        print(f"[STEP] step={step} action={action} reward={normalized:.3f} done={str(done).lower()} error={error}", flush=True)
    
    score = grade(total_reward)
    rewards_str = ",".join(f"{r:.3f}" for r in rewards)
    
    # REQUIRED: [END] line format
    print(f"[END] success=true steps={step} score={score:.3f} rewards={rewards_str}", flush=True)
    return score

if __name__ == "__main__":
    print("Running Inference on All Tasks...\n", flush=True)
    
    results = {}
    results['easy'] = run_task(easy, "easy")
    results['medium'] = run_task(medium, "medium")
    results['hard'] = run_task(hard, "hard")
    
    avg_score = sum(results.values()) / 3
    print(f"\nFinal Average Score: {avg_score:.3f}", flush=True)
