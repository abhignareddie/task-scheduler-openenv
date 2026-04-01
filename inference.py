from tasks import easy, medium, hard
from grader import grade

def run_task(task_func, name):
    env = task_func()
    state = env.state()
    done = False
    total_reward = 0

    while not done:
        action = 0
        state, reward, done, _ = env.step(action)
        total_reward += reward

    score = grade(total_reward)
    print(f"{name} Task -> Reward: {total_reward}, Score: {score}")

if __name__ == "__main__":
    print("Running Inference on All Tasks...\n")
    run_task(easy, "Easy")
    run_task(medium, "Medium")
    run_task(hard, "Hard")