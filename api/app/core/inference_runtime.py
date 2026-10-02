"""Shared concurrency budget for audio inference routes."""

import asyncio
import os

DEFAULT_MAX_CONCURRENT_INFERENCES = 2
_inference_semaphore: asyncio.Semaphore | None = None


def get_inference_semaphore() -> asyncio.Semaphore:
    global _inference_semaphore
    if _inference_semaphore is None:
        max_concurrent = int(
            os.getenv("MAX_CONCURRENT_INFERENCES", str(DEFAULT_MAX_CONCURRENT_INFERENCES))
        )
        _inference_semaphore = asyncio.Semaphore(max_concurrent)
    return _inference_semaphore


def reset_inference_semaphore() -> None:
    global _inference_semaphore
    _inference_semaphore = None


