def grade(total_reward):
    if total_reward >= 30:
        return 1.0
    elif total_reward >= 15:
        return 0.5
    else:
        return 0.0
#hd