from tasks import easy, medium, hard
from grader import grade

def run_task(task_func, name):
    env = task_func()
    state = env.state()
    done = False
    total_reward = 0
    step = 0

    print(f"[START] task={name}", flush=True)  # ADD THIS

    while not done:
        action = 0
        state, reward, done, _ = env.step(action)
        total_reward += reward
        step += 1
        print(f"[STEP] step={step} reward={reward}", flush=True)  # ADD THIS

    score = grade(total_reward)
    print(f"[END] task={name} score={score} steps={step}", flush=True)  # CHANGE THIS

if __name__ == "__main__":
    print("Running Inference on All Tasks...\n")
    run_task(easy, "Easy")
    run_task(medium, "Medium")
    run_task(hard, "Hard")
