import random
import numpy as np
class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        if len(self.memory) == self.capacity:
            del self.memory[0]
        self.memory.append(transition)

    def push_batch(self, transition):
        if len(self.memory) == self.capacity:
            del self.memory[0]
        self.memory.append(transition)

    def sample(self, batch_size):
        random_ints = np.random.randint(0, len(self.memory), size=batch_size)
        sample = [self.memory[random_int] for random_int in random_ints]
        return sample

    def __len__(self):
        return len(self.memory)
