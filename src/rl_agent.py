import random

def reward(demand):
    return -demand

print("Running RL stub...")

for episode in range(10):
    demand = random.uniform(1, 5)
    r = reward(demand)
    print(f"Episode {episode} | Demand: {demand:.2f} | Reward: {r:.2f}")