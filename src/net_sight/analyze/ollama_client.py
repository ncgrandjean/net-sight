"""Async client wrapping the Ollama Python SDK for VLM image analysis."""

from __future__ import annotations

import asyncio
import logging

import cv2
import numpy as np
from ollama import AsyncClient, RequestError, ResponseError

logger = logging.getLogger(__name__)


class OllamaClient:
    """Thin async wrapper around the Ollama chat API for vision-language models.

    Handles image encoding (np.ndarray -> PNG bytes) and concurrency control
    via a semaphore when processing batches.
    """

    def __init__(
        self,
        model: str = "qwen3-vl:8b",
        base_url: str = "http://localhost:11434",
        timeout: float = 300,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._client = AsyncClient(host=base_url, timeout=timeout)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image(image: np.ndarray) -> bytes:
        """Convert a numpy BGR image to PNG bytes suitable for the Ollama API."""
        success, buf = cv2.imencode(".png", image)
        if not success:
            raise ValueError("Failed to encode image as PNG")
        return buf.tobytes()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze_image(self, image: np.ndarray, prompt: str) -> str:
        """Send a single image with a text prompt to the VLM and return the response.

        Parameters
        ----------
        image:
            BGR numpy array (as returned by ``cv2.imread``).
        prompt:
            Textual instruction sent alongside the image.

        Returns
        -------
        str
            The model's text response.

        Raises
        ------
        ConnectionError
            When the Ollama server is unreachable.
        TimeoutError
            When the request exceeds the configured timeout.
        RuntimeError
            On any other Ollama API error.
        """
        png_bytes = self._encode_image(image)

        try:
            response = await self._client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [png_bytes],
                    },
                ],
            )
        except RequestError as exc:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.base_url}: {exc}"
            ) from exc
        except ResponseError as exc:
            if "timeout" in str(exc).lower():
                raise TimeoutError(
                    f"Ollama request timed out after {self.timeout}s: {exc}"
                ) from exc
            raise RuntimeError(f"Ollama API error: {exc}") from exc
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"Ollama request timed out after {self.timeout}s"
            ) from exc

        return response.message.content or ""

    async def analyze_image_batch(
        self,
        tasks: list[tuple[np.ndarray, str]],
        workers: int = 4,
    ) -> list[str]:
        """Process multiple (image, prompt) pairs with bounded concurrency.

        Parameters
        ----------
        tasks:
            List of ``(image, prompt)`` tuples to process.
        workers:
            Maximum number of concurrent Ollama requests.

        Returns
        -------
        list[str]
            Responses in the same order as *tasks*.
        """
        semaphore = asyncio.Semaphore(workers)

        async def _run(idx: int, image: np.ndarray, prompt: str) -> tuple[int, str]:
            async with semaphore:
                result = await self.analyze_image(image, prompt)
                return idx, result

        coros = [_run(i, img, prompt) for i, (img, prompt) in enumerate(tasks)]
        results = await asyncio.gather(*coros)

        # Re-order by original index
        ordered = [""] * len(tasks)
        for idx, text in results:
            ordered[idx] = text
        return ordered
