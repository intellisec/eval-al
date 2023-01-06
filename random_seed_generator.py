import random


class RandomSeedGenerator:
    def __init__(self, seed=None):
        self.min_rand_value = 0
        self.max_rand_value = 1000000
        self.seed = seed
        if seed:
            random.seed(seed)

    def get_k_random_numbers(self, k):
        if self.seed is None:
            return [None] * 5
        return random.choices(range(self.max_rand_value), k=k)