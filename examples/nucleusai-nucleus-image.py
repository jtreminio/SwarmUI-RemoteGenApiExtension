"""Remote text-to-image generation server for Nucleus-Image.

Loads ``NucleusMoEImagePipeline`` once at startup and serves the standard
``POST /generate`` contract over stdlib ``http.server``. Text KV caching
across denoising steps (the model's headline speedup) is enabled by
default and can be disabled with ``--disable_kv_cache``.

Request body fields (``prompt``, ``width`` and ``height`` are required;
``steps``/``cfg``/``seed`` fall back to the CLI defaults the server was
started with):

    {
        "prompt":          "...",       // required
        "width":           int,         // required
        "height":          int,         // required
        "seed":            int | null,
        "steps":           int,
        "cfg":             float,
        "negative_prompt": "...",       // pass-through if non-empty
        "thinking":        bool         // ignored: model has no thinking mode
    }

Width/height mapping
--------------------
Nucleus-Image was trained with aspect-ratio bucketing at a 1024-base
resolution. Incoming ``width``/``height`` are mapped to the nearest
supported aspect-ratio bucket (defaulting to 1:1); the bucket's
(width, height) pair is what the pipeline actually generates. The
requested dimensions only influence which bucket is chosen.

Supported buckets:

  - 1:1   1024x1024
  - 16:9  1344x768
  - 9:16  768x1344
  - 4:3   1184x896
  - 3:4   896x1184
  - 3:2   1248x832
  - 2:3   832x1248

Model resolution
----------------
``--repo_id`` accepts a HF id (``NucleusAI/Nucleus-Image``), a bare
name (``Nucleus-Image`` -> ``NucleusAI/Nucleus-Image``), or a local
directory. ``from_pretrained`` handles caching/download via the
standard Hugging Face cache.

Example:

    python server.py --repo_id NucleusAI/Nucleus-Image

    curl -X POST http://localhost:7802/generate \\
        -H 'Content-Type: application/json' \\
        -d '{"prompt": "a cat sitting on a roof",
             "width": 1024, "height": 1024,
             "steps": 30, "cfg": 4.5, "seed": 42}'
"""

from __future__ import annotations

import argparse
import base64
import inspect
import io
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch

from diffusers import DiffusionPipeline, TextKVCacheConfig


SUPPORTED_ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3": (1184, 896),
    "3:4": (896, 1184),
    "3:2": (1248, 832),
    "2:3": (832, 1248),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nucleus-Image text-to-image inference HTTP server"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="NucleusAI/Nucleus-Image",
        help=(
            "HF repo id, bare model name, or local directory path. "
            "Bare names default to NucleusAI/<name>. Auto-downloaded "
            "into the standard Hugging Face cache if not local."
        ),
    )
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Bind host for the HTTP server.")
    parser.add_argument("--port", type=int, default=7802,
                        help="Bind port for the HTTP server.")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Enable diffusers model CPU offload to reduce peak VRAM "
             "(instead of moving the whole pipeline to CUDA).",
    )
    parser.add_argument(
        "--disable_kv_cache",
        action="store_true",
        help="Disable text-KV caching across denoising steps. The cache "
             "is enabled by default and is the model's headline speedup.",
    )
    parser.add_argument("--steps", type=int, default=50,
                        help="Per-request fallback when 'steps' is omitted.")
    parser.add_argument("--cfg", type=float, default=4.0,
                        help="Per-request fallback when 'cfg' is omitted.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Per-request fallback when 'seed' is omitted. Omit for a "
             "fresh random seed each call.",
    )
    return parser.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[name]


def _resolve_model(repo_id: str) -> str:
    if os.path.isdir(repo_id):
        return repo_id
    return repo_id if "/" in repo_id else f"NucleusAI/{repo_id}"


def _nearest_aspect_ratio(width: int, height: int) -> str:
    target = width / height
    best, best_diff = "1:1", float("inf")
    for ar, (w_s, h_s) in SUPPORTED_ASPECT_RATIOS.items():
        diff = abs(w_s / h_s - target)
        if diff < best_diff:
            best, best_diff = ar, diff
    return best


def _generate(
    pipe,
    prompt: str,
    *,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int | None,
    negative_prompt: str | None,
    supports_negative: bool,
) -> str:
    generator = (
        torch.Generator(device=pipe._execution_device).manual_seed(int(seed))
        if seed is not None
        else None
    )

    call_kwargs: dict = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": cfg,
        "num_images_per_prompt": 1,
        "generator": generator,
    }
    if supports_negative and negative_prompt:
        call_kwargs["negative_prompt"] = negative_prompt

    out = pipe(**call_kwargs)

    images = list(out.images)
    if len(images) != 1:
        raise RuntimeError(
            f"Pipeline returned {len(images)} images; expected 1."
        )

    buf = io.BytesIO()
    images[0].save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _resolve_request_params(data: dict, args: argparse.Namespace) -> dict:
    """Merge per-request overrides with CLI defaults, validating types.

    Raises ValueError with a user-facing message on bad input.
    """
    def _get(key, expected_type, default, required=False):
        if key not in data:
            if required:
                raise ValueError(f"Missing required field '{key}'.")
            return default
        val = data[key]
        if expected_type is float and isinstance(val, int) and not isinstance(val, bool):
            return float(val)
        if not isinstance(val, expected_type) or isinstance(val, bool):
            raise ValueError(
                f"'{key}' must be {expected_type.__name__}, got {type(val).__name__}."
            )
        return val

    req_width = _get("width", int, None, required=True)
    req_height = _get("height", int, None, required=True)
    if req_width < 1 or req_height < 1:
        raise ValueError("'width' and 'height' must be positive integers.")

    ar = _nearest_aspect_ratio(req_width, req_height)
    width, height = SUPPORTED_ASPECT_RATIOS[ar]

    steps = _get("steps", int, args.steps)
    if steps < 1:
        raise ValueError("'steps' must be >= 1.")

    cfg = _get("cfg", float, args.cfg)

    if "seed" in data and data["seed"] is not None:
        if not isinstance(data["seed"], int) or isinstance(data["seed"], bool):
            raise ValueError("'seed' must be an integer or null.")
        seed = data["seed"]
    else:
        seed = args.seed

    negative_prompt = None
    if "negative_prompt" in data:
        np_val = data["negative_prompt"]
        if not isinstance(np_val, str) or isinstance(np_val, bool):
            raise ValueError(
                f"'negative_prompt' must be str, got {type(np_val).__name__}."
            )
        np_val = np_val.strip()
        if np_val:
            negative_prompt = np_val

    if "thinking" in data:
        thinking_val = data["thinking"]
        if not isinstance(thinking_val, bool):
            raise ValueError(
                f"'thinking' must be bool, got {type(thinking_val).__name__}."
            )

    return {
        "width": width,
        "height": height,
        "steps": steps,
        "cfg": cfg,
        "seed": seed,
        "negative_prompt": negative_prompt,
    }


def _make_handler(pipe, args: argparse.Namespace, supports_negative: bool):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, body: dict) -> None:
            payload = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:
            if self.path.rstrip("/") not in ("", "/generate"):
                self._send_json(404, {"error": f"Unknown path: {self.path}"})
                return

            length = int(self.headers.get("Content-Length", "0") or "0")
            try:
                raw = self.rfile.read(length).decode("utf-8") if length else ""
                data = json.loads(raw) if raw else {}
                if not isinstance(data, dict):
                    raise ValueError("Request body must be a JSON object.")
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
                self._send_json(400, {"error": f"Invalid JSON body: {e}"})
                return

            prompt = data.get("prompt")
            if not isinstance(prompt, str) or isinstance(prompt, bool) or not prompt.strip():
                self._send_json(400, {"error": "Missing or empty 'prompt' string."})
                return
            prompt = prompt.strip()

            try:
                gen_kwargs = _resolve_request_params(data, args)
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return

            try:
                image = _generate(
                    pipe, prompt,
                    supports_negative=supports_negative,
                    **gen_kwargs,
                )
            except Exception as e:
                print(f"[generate] failed: {e!r}")
                self._send_json(500, {"error": f"Generation failed: {e}"})
                return

            print(f"[generate] returning image :: {prompt!r}")
            self._send_json(200, {"image": image})

        def do_GET(self) -> None:
            self._send_json(404, {"error": f"Unknown path: {self.path}"})

        def log_message(self, format: str, *log_args) -> None:
            sys.stderr.write("[http] " + (format % log_args) + "\n")

    return Handler


def main() -> None:
    args = parse_args()

    dtype = _torch_dtype(args.dtype)
    model_source = _resolve_model(args.repo_id)
    print(f"[load] using model source: {model_source}")

    pipe = DiffusionPipeline.from_pretrained(model_source, torch_dtype=dtype)

    if args.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    if not args.disable_kv_cache:
        pipe.transformer.enable_cache(TextKVCacheConfig())
        print("[load] text KV cache enabled")
    else:
        print("[load] text KV cache disabled")

    supports_negative = "negative_prompt" in inspect.signature(pipe.__call__).parameters
    print(f"[load] negative_prompt supported: {supports_negative}")

    handler_cls = _make_handler(pipe, args, supports_negative)
    server = HTTPServer((args.host, args.port), handler_cls)

    print(
        f"\nReady. Listening on http://{args.host}:{args.port}/generate. "
        "POST {\"prompt\": \"...\", \"width\": W, \"height\": H} to generate. "
        "Ctrl-C to exit."
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
