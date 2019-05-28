from collections import deque
import random 
import numpy as np 

class replayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.num_exp = 0

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.num_exp < self.buffer_size:
            self.buffer.append(experience)
            self.num_exp += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
    
    def size(self):
        return self.buffer_size
    
    def count(self):
        return self.num_exp 

    def sample(self, batch_size):
        if batch_size > self.num_exp:
            batch = random.sample(self.buffer, self.num_exp)
        else:
            batch = random.sample(self.buffer, batch_size)
        # each is (batch_size, feature_size)
        s, a, r, t, s2 = map(np.stack, zip(*batch))
        return s, a, r, t, s2 

    def clear(self):
        self.buffer = deque()
        self.num_exp = 0