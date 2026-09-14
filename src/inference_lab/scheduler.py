from collections import deque
from .request import Request

class Scheduler:
    def __init__(self, max_batch_size: int = 8):
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.max_batch_size = max_batch_size

    def add_request(self, request: Request):
        self.waiting.append(request)

    def schedule(self) -> list[Request]:
        """Admit waiting requests into the running batch (FCFS), up to capacity."""
        while self.waiting and len(self.running) < self.max_batch_size:
            self.running.append(self.waiting.popleft())
        return self.running

    def finish_request(self, request: Request):
        request.finished = True
        self.running.remove(request)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)