def add(a, b):
    return a + b


def mean(values):
    if not values:
        raise ValueError("mean of empty list")
    return sum(values) / len(values)
