def grade(total_reward):
    max_possible = 30.0
    score = total_reward / max_possible
    score = max(0.0, min(score, 1.0))
    return round(score, 3)