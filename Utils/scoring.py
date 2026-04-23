def calculate_score(counts):
    fish = counts.get("fish", 0)
    coral = counts.get("coral", 0)
    plastic = counts.get("plastic", 0)

    score = (fish + coral) - (plastic * 2)
    return score