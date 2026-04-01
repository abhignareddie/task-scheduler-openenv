from env import TaskSchedulerEnv

# Easy Task
def easy():
    env = TaskSchedulerEnv()
    env.tasks = [
        {"id": 1, "deadline": 5, "priority": 3, "duration": 1},
        {"id": 2, "deadline": 6, "priority": 4, "duration": 2}
    ]
    env.current_time = 0
    return env

# Medium Task
def medium():
    env = TaskSchedulerEnv()
    env.tasks = [
        {"id": 1, "deadline": 4, "priority": 6, "duration": 2},
        {"id": 2, "deadline": 3, "priority": 8, "duration": 1},
        {"id": 3, "deadline": 7, "priority": 5, "duration": 2}
    ]
    env.current_time = 0
    return env

# Hard Task
def hard():
    env = TaskSchedulerEnv()
    env.tasks = [
        {"id": 1, "deadline": 2, "priority": 10, "duration": 2},
        {"id": 2, "deadline": 3, "priority": 9, "duration": 1},
        {"id": 3, "deadline": 4, "priority": 8, "duration": 2},
        {"id": 4, "deadline": 6, "priority": 7, "duration": 3}
    ]
    env.current_time = 0
    return env