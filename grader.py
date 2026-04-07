def grade(total_reward):
    # Expected max total reward ~3–5 depending on steps
    max_possible = 5.0  

    score = total_reward / max_possible

    # Clamp between 0 and 1
    score = max(0.0, min(score, 1.0))

    return round(score, 2)