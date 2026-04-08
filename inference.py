import os
import re
import sys
from tasks import easy, medium, hard
from grader import grade

HAS_OPENAI = False
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    pass

def get_action(state):
    try:
        tasks = state.get("tasks", [])
        if not tasks:
            return 0
        
        best_idx = 0
        best_score = -999999
        
        for i, task in enumerate(tasks):
            priority = task.get("priority", 1)
            deadline = task.get("deadline", 10)
            duration = task.get("duration", 1)
            score = priority / (deadline * duration)
            if score > best_score:
                best_score = score
                best_idx = i
        
        print(f"[ACTION] Chosen task {best_idx}", flush=True)
        return best_idx
    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        return 0

def run_task(task_func, task_name):
    env = task_func()
    state = env.state()
    done = False
    total_reward = 0
    step = 0
    rewards = []
    
    print(f"[START] task={task_name}", flush=True)
    
    while not done and step < 20:
        action = get_action(state)
        state, reward, done, info = env.step(action)
        normalized = round(min(max(reward / 30.0, 0.0), 1.0), 3)
        total_reward += normalized
        step += 1
        rewards.append(normalized)
        print(f"[STEP] step={step} action={action} reward={normalized:.3f} done={str(done).lower()}", flush=True)
    
    score = grade(total_reward)
    rewards_str = ",".join(str(r) for r in rewards)
    print(f"[END] steps={step} score={score:.3f} rewards={rewards_str}", flush=True)
    return score

if __name__ == "__main__":
    print("Running Inference...\n", flush=True)
    easy_score = run_task(easy, "easy")
    medium_score = run_task(medium, "medium")
    hard_score = run_task(hard, "hard")
    avg = (easy_score + medium_score + hard_score) / 3
    print(f"\nAverage Score: {avg:.3f}", flush=True)