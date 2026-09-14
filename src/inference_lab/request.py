import time
import uuid
from dataclasses import dataclass, field

import torch

from cache import KVCache

@dataclass
class Request:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = ""
    max_new_tokens: int = 20

    input_ids: torch.Tensor | None = None
    input_tokens: int = 0

    output_ids: list[int] = field(default_factory=list)

    finished: bool = False
    finish_reason: str | None = None  # "eos" | "length" | "stop_str" | "abort"; useful for logging/debugging and API responses

    # Runtime generation state
    cache: KVCache | None = None
    outputs: object | None = None

    # sampling params almost always live per-request, not global
    # temperature: float = 1.0
    # top_p: float = 1.0

    # needed later for scheduling policies (FCFS, priority, timeout)
    arrival_time: float = field(default_factory=time.time)

    def num_generated(self) -> int:
        return len(self.output_ids)

    def is_finished(self) -> bool:
        return self.finished