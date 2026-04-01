import random

class TaskSchedulerEnv:
    def __init__(self):
        self.tasks = []
        self.current_time = 0

    def reset(self):
        self.current_time = 0
        self.tasks = self._generate_tasks()
        return self.state()

    def _generate_tasks(self):
        num_tasks = random.randint(5, 8)
        tasks = []
        for i in range(num_tasks):
            duration = random.randint(1, 3)
            deadline = random.randint(duration + 1, 10)
            task = {"id": i, "deadline": deadline, "priority": random.randint(1,10), "duration": duration}
            tasks.append(task)
        return tasks

    def state(self):
        return {"tasks": self.tasks, "current_time": self.current_time}

    def step(self, action):
        if action < 0 or action >= len(self.tasks):
            return self.state(), -10, False, {"error": "Invalid action"}

        task = self.tasks.pop(action)
        self.current_time += task["duration"]

        if self.current_time <= task["deadline"]:
            reward = task["priority"] * 2
        else:
            reward = task["priority"] - 5

        reward += (task["deadline"] - self.current_time)
        done = len(self.tasks) == 0
        return self.state(), reward, done, {}