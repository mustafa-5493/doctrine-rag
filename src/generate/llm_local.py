"""
llm_local.py — thin wrapper around llama-cpp-python for the local generator.

Runs on the GTX 1050 laptop. Requires llama-cpp-python built with CUDA
support (cuBLAS) for GPU offload — the plain `pip install llama-cpp-python`
often installs a CPU-only wheel. See README note in benchmark.py for the
CMAKE_ARGS install command if n_gpu_layers > 0 has no effect on VRAM usage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class GenResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    gen_seconds: float

    @property
    def tokens_per_sec(self) -> float:
        return self.completion_tokens / self.gen_seconds if self.gen_seconds > 0 else 0.0


class LocalLLM:
    """Loads one GGUF model at a given GPU-layer offload level."""

    def __init__(self, model_path: str, n_gpu_layers: int = -1,
                n_ctx: int = 4096, verbose: bool = False):
        from llama_cpp import Llama  # imported here so module loads w/o the lib
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self._llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,   # -1 = offload all layers to GPU
            n_ctx=n_ctx,
            verbose=verbose,
        )

    def generate(self, prompt: str, max_tokens: int = 512,
                temperature: float = 0.0) -> GenResult:
        t0 = time.perf_counter()
        out = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            echo=False,
        )
        dt = time.perf_counter() - t0
        usage = out.get("usage", {})
        return GenResult(
            text=out["choices"][0]["text"].strip(),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            gen_seconds=dt,
        )

    def close(self):
        # llama-cpp-python frees GPU memory on garbage collection; explicit
        # del + gc helps ensure VRAM is released between benchmark configs.
        del self._llm
        import gc
        gc.collect()
