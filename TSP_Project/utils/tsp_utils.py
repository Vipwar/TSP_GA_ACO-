import random
import math

def generate_cities(n):
    return [(random.randint(50, 450), random.randint(50, 450)) for _ in range(n)]

def total_distance(path, cities):
    dist = 0
    for i in range(len(path)):
        x1, y1 = cities[path[i]]
        x2, y2 = cities[path[(i + 1) % len(path)]]
        dist += math.hypot(x1 - x2, y1 - y2)
    return dist
