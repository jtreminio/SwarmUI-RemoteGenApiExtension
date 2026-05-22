"""ComfyUI node that forwards a prompt to a remote generation API server."""

from __future__ import annotations

import base64
import io as _io
import json
import time
import urllib.error
import urllib.request
from typing import Final

import numpy as np
import torch
from comfy_api.latest import io
from PIL import Image, ImageOps


DEFAULT_SERVER_URL: Final[str] = "http://localhost:8000/generate"
DEFAULT_DIMENSION: Final[int] = 1024
DEFAULT_STEPS: Final[int] = 20
DEFAULT_CFG: Final[float] = 7.0
DEFAULT_TIMEOUT_SECONDS: Final[float] = 120.0

MIN_DIMENSION: Final[int] = 1
MAX_DIMENSION: Final[int] = 16384
MIN_STEPS: Final[int] = 1
MAX_STEPS: Final[int] = 1000
MIN_CFG: Final[float] = 0.0
MAX_CFG: Final[float] = 100.0
MIN_TIMEOUT_SECONDS: Final[float] = 1.0
MAX_TIMEOUT_SECONDS: Final[float] = 3600.0
MIN_SEED: Final[int] = -1
MAX_SEED: Final[int] = 0xFFFFFFFFFFFFFFFF

RESPONSE_IMAGE_KEYS: Final[tuple[str, ...]] = ("image", "image_base64", "data")


def _b64_to_image_tensor(image_b64: str) -> torch.Tensor:
    image_bytes = base64.b64decode(image_b64)
    img = Image.open(_io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None,]


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def _extract_image_b64(result: object) -> str | None:
    if not isinstance(result, dict):
        return None
    for key in RESPONSE_IMAGE_KEYS:
        value = result.get(key)
        if isinstance(value, str) and value:
            return value
    return None


class SwarmRemoteGenApi(io.ComfyNode):
    """Call a remote generation API server and return the resulting image."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SwarmRemoteGenApi",
            display_name="Swarm Remote Gen API",
            category="SwarmUI/remote_gen_api",
            description=(
                "Calls a remote generation API server with a prompt and returns the generated "
                "image. The server must accept POST JSON with {prompt, negative_prompt, width, "
                "height, seed, steps, cfg, thinking} and respond with JSON containing a "
                "base64-encoded image under one of: 'image', 'image_base64', or 'data'."
            ),
            inputs=[
                io.String.Input("server_url", default=DEFAULT_SERVER_URL),
                io.String.Input("prompt", multiline=True, default=""),
                io.String.Input("negative_prompt", multiline=True, default=""),
                io.Int.Input("width", default=DEFAULT_DIMENSION, min=MIN_DIMENSION, max=MAX_DIMENSION, step=1),
                io.Int.Input("height", default=DEFAULT_DIMENSION, min=MIN_DIMENSION, max=MAX_DIMENSION, step=1),
                io.Int.Input("seed", default=0, min=MIN_SEED, max=MAX_SEED),
                io.Int.Input("steps", default=DEFAULT_STEPS, min=MIN_STEPS, max=MAX_STEPS),
                io.Float.Input("cfg", default=DEFAULT_CFG, min=MIN_CFG, max=MAX_CFG, step=0.1),
                io.Boolean.Input("thinking", default=False),
                io.Float.Input(
                    "timeout_seconds",
                    optional=True,
                    default=DEFAULT_TIMEOUT_SECONDS,
                    min=MIN_TIMEOUT_SECONDS,
                    max=MAX_TIMEOUT_SECONDS,
                    step=1.0,
                ),
                # pixels: not currently implemented (reserved for future image edit support)
                io.Image.Input("pixels", optional=True),
            ],
            outputs=[
                io.Image.Output("image"),
            ],
        )

    @classmethod
    def execute(
        cls,
        server_url: str,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: int,
        steps: int,
        cfg: float,
        thinking: bool = False,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        pixels: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        _ = pixels
        if not server_url or not server_url.strip():
            raise ValueError("SwarmRemoteGenApi: server_url is empty")
        url = server_url.strip()
        payload = {
            "prompt": prompt or "",
            "negative_prompt": negative_prompt or "",
            "width": int(width),
            "height": int(height),
            "seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "thinking": bool(thinking),
        }
        try:
            result = _post_json(url, payload, timeout=timeout_seconds)
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(
                f"SwarmRemoteGenApi: server returned HTTP {e.code}: {detail or e.reason}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"SwarmRemoteGenApi: failed to reach {url}: {e.reason}") from e

        image_b64 = _extract_image_b64(result)
        if image_b64 is None:
            raise RuntimeError(
                "SwarmRemoteGenApi: response did not include a base64 image under "
                "'image', 'image_base64', or 'data'"
            )

        if image_b64.startswith("data:"):
            _, _, image_b64 = image_b64.partition(",")

        tensor = _b64_to_image_tensor(image_b64)
        return io.NodeOutput(tensor)

    @classmethod
    def fingerprint_inputs(cls, **_kwargs) -> float:
        return time.time()
