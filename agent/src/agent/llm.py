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
        # Sensitive-data rules from the plan, enforced in code:
        # 1. Cloud models run on Ollama's servers, so your data would leave this PC.
        if "cloud" in model:
            raise ModelNotAvailable(f"{model} is a cloud model; only local models are allowed.")
        # 2. Only talk to an Ollama running on this PC.
        if urlparse(host).hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ModelNotAvailable(f"Ollama host {host} is not on this PC.")

        self.model = model
        self.think = think
        self.options = {"num_ctx": num_ctx}
        self.client = ollama.Client(host=host)

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
        # With think on, the reasoning arrives separately in message.thinking.
        # We keep only the final answer: the reasoning would fill the context
        # window fast and the model doesn't need its old reasoning back.
        return response.message.content or ""