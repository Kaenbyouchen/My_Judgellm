"""
vLLM model client (optional dependency).
"""
import os
import multiprocessing as mp
from typing import Optional, Dict, Any
from loguru import logger

from .base import BaseModel


class VLLMModel(BaseModel):
    """vLLM model client."""

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        config = config or {}
        self.tensor_parallel_size = config.get("tensor_parallel_size", 1)
        self.max_model_len = config.get("max_model_len", 2048)
        self.dtype = config.get("dtype", "auto")
        self.gpu_memory_utilization = config.get("gpu_memory_utilization", 0.9)
        self.trust_remote_code = config.get("trust_remote_code", False)
        self.quantization = config.get("quantization")
        self.download_dir = config.get("download_dir") or config.get("local_cache_dir")
        self.seed = config.get("seed")
        self.enforce_eager = config.get("enforce_eager", False)
        # Prevent CUDA re-init crash in forked subprocesses on Linux HPC.
        self.worker_multiproc_method = config.get("worker_multiproc_method", "spawn")

        self._vllm = None
        self._llm = None

        try:
            self._ensure_vllm_multiprocessing_mode()
            import vllm
            self._vllm = vllm
            logger.info(f"vLLM client initialized for {model_name}")
        except ImportError:
            logger.warning("vllm not installed. vLLM client will not work.")

    def _ensure_vllm_multiprocessing_mode(self) -> None:
        """
        Force vLLM worker multiprocessing mode to spawn.
        This avoids: 'Cannot re-initialize CUDA in forked subprocess'.
        """
        method = str(self.worker_multiproc_method).strip().lower() or "spawn"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = method
        try:
            current = mp.get_start_method(allow_none=True)
            if current != method:
                mp.set_start_method(method, force=True)
                logger.info(f"Set multiprocessing start method to '{method}' for vLLM")
        except RuntimeError:
            # Start method may already be set by parent process; env var still helps vLLM.
            pass

    def _load_model(self) -> None:
        """Lazy load the vLLM model."""
        if self._llm is not None:
            return
        if self._vllm is None:
            raise RuntimeError("vllm package not installed")

        llm_kwargs = {
            "model": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "dtype": self.dtype,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": self.trust_remote_code,
            "quantization": self.quantization,
            "download_dir": self.download_dir,
            "seed": self.seed,
            "enforce_eager": self.enforce_eager,
        }
        llm_kwargs = {k: v for k, v in llm_kwargs.items() if v is not None}

        logger.info(f"Loading vLLM model: {self.model_name}")
        self._llm = self._vllm.LLM(**llm_kwargs)
        logger.info("vLLM model loaded successfully")

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Generate text using vLLM model.
        """
        if not self.is_available():
            raise RuntimeError("vLLM client is not available. Check package installation.")

        self._load_model()

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        sampling_params = self._vllm.SamplingParams(
            temperature=kwargs.get("temperature", self.config.get("temperature", 0.7)),
            max_tokens=kwargs.get("max_new_tokens", self.config.get("max_new_tokens", 512)),
            top_p=kwargs.get("top_p", self.config.get("top_p", 1.0)),
            top_k=kwargs.get("top_k", self.config.get("top_k", -1)),
            stop=kwargs.get("stop", self.config.get("stop")),
        )

        try:
            outputs = self._llm.generate([full_prompt], sampling_params=sampling_params)
            if not outputs or not outputs[0].outputs:
                return ""
            return outputs[0].outputs[0].text.strip()
        except Exception as e:
            logger.error(f"vLLM generation error: {e}")
            raise

    def is_available(self) -> bool:
        """Check if vLLM client is available."""
        return self._vllm is not None
