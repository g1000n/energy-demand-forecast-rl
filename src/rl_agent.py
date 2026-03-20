import numpy as np
import random
import matplotlib.pyplot as plt

NUM_STATES = 5
NUM_ACTIONS = 3
EPISODES = 100

ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.2

q_table = np.zeros((NUM_STATES, NUM_ACTIONS))

def get_state(demand):
    if demand < 1.5:
        return 0
    elif demand < 2.5:
        return 1
    elif demand < 3.5:
        return 2
    elif demand < 4.5:
        return 3
    return 4

def choose_action(state):
    if random.random() < EPSILON:
        return random.randint(0, NUM_ACTIONS - 1)
    return np.argmax(q_table[state])

def environment_step(demand, action):
    # action 0 = do nothing
    # action 1 = shift some load
    # action 2 = aggressively reduce load

    base_cost = demand * 10

    if action == 0:
        new_cost = base_cost
    elif action == 1:
        new_cost = base_cost * 0.9
    else:
        new_cost = base_cost * 0.8

    reward = base_cost - new_cost
    next_demand = random.uniform(1, 5)
    next_state = get_state(next_demand)
    return next_state, reward

episode_rewards = []

for episode in range(EPISODES):
    demand = random.uniform(1, 5)
    state = get_state(demand)
    total_reward = 0

    for step in range(20):
        action = choose_action(state)
        next_state, reward = environment_step(demand, action)

        q_table[state, action] = q_table[state, action] + ALPHA * (
            reward + GAMMA * np.max(q_table[next_state]) - q_table[state, action]
        )

        state = next_state
        total_reward += reward
        demand = random.uniform(1, 5)

    episode_rewards.append(total_reward)

print("Final Q-table:")
print(q_table)

plt.plot(episode_rewards)
plt.title("RL Learning Curve")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()