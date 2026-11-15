"""
The model. Anything with a `chat(messages) -> str` method can drive the agent:
OllamaLLM for real use, or a scripted fake in tests. The loop never imports
ollama directly, which is what makes it testable without a GPU.
"""

from urllib.parse import urlparse

import ollama


class ModelNotAvailable(Exception):
    pass


class OllamaLLM:
    def __init__(self, model: str, host: str, think: bool, num_ctx: int):
        # Data never leaves this machine:
        # 1. Cloud models run on Ollama's servers.
        if "cloud" in model:
            raise ModelNotAvailable(f"{model} is a cloud model; only local models are allowed.")
        # 2. Only a local Ollama. host.docker.internal is the host itself, as
        #    seen from the UI container.
        if urlparse(host).hostname not in ("127.0.0.1", "localhost", "::1", "host.docker.internal"):
            raise ModelNotAvailable(f"Ollama host {host} is not on this machine.")

        self.model = model
        self.think = think
        self.options = {"num_ctx": num_ctx}
        self.client = ollama.Client(host=host)
        self.last_usage: dict = {}   # token counts from the most recent call

    def check_available(self) -> None:
        """Fail early, with a clear fix, if Ollama isn't running or the model isn't pulled."""
        try:
            self.client.show(self.model)
        except ollama.ResponseError as e:
            if e.status_code == 404:
                raise ModelNotAvailable(f"Model not downloaded. Run: ollama pull {self.model}") from e
            raise
        except ConnectionError as e:
            raise ModelNotAvailable("Can't reach Ollama. Is it running?") from e

    def chat(self, messages: list[dict]) -> str:
        response = self.client.chat(
            model=self.model,
            messages=messages,
            think=self.think,
            options=self.options,
        )
        # prompt_eval_count is the whole input in tokens, so it shows how full
        # the num_ctx window is; Ollama silently drops the oldest input when it
        # doesn't fit. eval_count is the reply, thinking included.
        self.last_usage = {"prompt_tokens": response.prompt_eval_count or 0,
                           "reply_tokens": response.eval_count or 0}
        # Keep only the answer, not message.thinking: old reasoning would fill
        # the context window and the model doesn't need it back.
        return response.message.content or ""