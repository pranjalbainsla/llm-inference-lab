import torch
from torch.nn.utils.rnn import pad_sequence
from model import Model

model = Model()

class ModelExecutor:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def prefill(self, requests):
        """
        Run the prompt of every request as one batched model invocation.

        Each request is expected to have:
            request.input_ids: 1D tensor of prompt token IDs

        The latest logits for each request are stored as:
            request.logits
        """
        if not requests:
            return

        input_ids = [
            request.input_ids.to(self.device)
            for request in requests
        ]

        # Different prompts can have different lengths, so pad them.
        batch_input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        # Attention mask tells the model which positions are real tokens.
        attention_mask = (
            batch_input_ids != self.tokenizer.pad_token_id
        ).long()

        # ONE model invocation for the entire batch.
        outputs = self.model(
            input_ids=batch_input_ids,
            attention_mask=attention_mask,
        )

        # Each request needs the logits corresponding to its
        # final (non-padding) prompt token.
        for i, request in enumerate(requests):
            prompt_length = input_ids[i].shape[0]
            request.logits = outputs.logits[i, prompt_length - 1]

    @torch.no_grad()
    def decode(self, requests):
        """
        Run one decode step for every request as one batched model invocation.

        Each request is expected to have:
            request.next_token_id: the token generated previously

        The resulting logits are stored back on each request.
        """
        if not requests:
            return

        # One token per request.
        input_ids = torch.stack(
            [request.next_token_id.to(self.device) for request in requests]
        )

        # [batch] -> [batch, 1]
        input_ids = input_ids.unsqueeze(1)

        attention_mask = torch.ones(
            input_ids.shape,
            dtype=torch.long,
            device=self.device,
        )

        # ONE model invocation for the entire batch.
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        for i, request in enumerate(requests):
            request.logits = outputs.logits[i, -1]