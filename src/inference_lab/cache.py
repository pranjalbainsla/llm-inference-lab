from transformers import DynamicCache

class KVCache:
    def __init__(self):
        self.cache = DynamicCache()

    def get(self):
        return self.cache