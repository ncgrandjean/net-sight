"""Async client wrapping the Ollama Python SDK for VLM image analysis."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable

import cv2
import numpy as np
from ollama import AsyncClient, RequestError, ResponseError

logger = logging.getLogger(__name__)


class OllamaClient:
    """Thin async wrapper around the Ollama chat API for vision-language models.

    Handles image encoding (np.ndarray -> PNG bytes) and supports streaming
    responses with thinking-token awareness for reasoning models.
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

        Convenience wrapper around :meth:`analyze_image_stream` without a
        streaming callback, kept for backward compatibility.

        Parameters
        ----------
        image:
            BGR numpy array (as returned by ``cv2.imread``).
        prompt:
            Textual instruction sent alongside the image.

        Returns
        -------
        str
            The model's text response (thinking tokens stripped).

        Raises
        ------
        ConnectionError
            When the Ollama server is unreachable.
        TimeoutError
            When the request exceeds the configured timeout.
        RuntimeError
            On any other Ollama API error.
        """
        return await self.analyze_image_stream(image, prompt)

    async def analyze_image_stream(
        self,
        image: np.ndarray,
        prompt: str,
        stream_callback: Callable[[str, bool], None] | None = None,
    ) -> str:
        """Send a single image with a text prompt and stream the response.

        When *stream_callback* is provided it is called for every received
        token fragment as ``stream_callback(fragment, is_thinking)`` where
        *is_thinking* is ``True`` while the model emits its internal
        reasoning between ``<think>`` and ``</think>`` tags.

        Parameters
        ----------
        image:
            BGR numpy array (as returned by ``cv2.imread``).
        prompt:
            Textual instruction sent alongside the image.
        stream_callback:
            Optional synchronous callback invoked with each text fragment.

        Returns
        -------
        str
            The model's final text response with thinking blocks removed.

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
            stream = await self._client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [png_bytes],
                    },
                ],
                stream=True,
            )

            full_text: list[str] = []
            is_thinking = False

            async for chunk in stream:
                fragment: str = chunk.message.content or ""
                if not fragment:
                    continue

                # Track <think> / </think> boundaries across fragments.
                # A single fragment may contain an opening or closing tag,
                # or even both, so we process the text incrementally.
                remaining = fragment
                while remaining:
                    if is_thinking:
                        # Look for the closing tag
                        close_idx = remaining.find("</think>")
                        if close_idx != -1:
                            thinking_part = remaining[:close_idx]
                            if thinking_part and stream_callback is not None:
                                stream_callback(thinking_part, True)
                            full_text.append(remaining[: close_idx + len("</think>")])
                            remaining = remaining[close_idx + len("</think>"):]
                            is_thinking = False
                        else:
                            if stream_callback is not None:
                                stream_callback(remaining, True)
                            full_text.append(remaining)
                            remaining = ""
                    else:
                        # Look for an opening tag
                        open_idx = remaining.find("<think>")
                        if open_idx != -1:
                            before = remaining[:open_idx]
                            if before and stream_callback is not None:
                                stream_callback(before, False)
                            full_text.append(remaining[: open_idx + len("<think>")])
                            remaining = remaining[open_idx + len("<think>"):]
                            is_thinking = True
                        else:
                            if stream_callback is not None:
                                stream_callback(remaining, False)
                            full_text.append(remaining)
                            remaining = ""

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

        # Strip all <think>...</think> blocks from the accumulated text.
        raw = "".join(full_text)
        return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
