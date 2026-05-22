"""Example HTTP server variant of Microsoft Lens' ``inference.py``.

Source: https://github.com/microsoft/Lens

Loads the Lens pipeline once at startup and stays running. On each POST
to ``/generate`` with a JSON body, generates a single image and responds
with ``{"image": "<base64 PNG>"}``.

Request body fields (``prompt``, ``width`` and ``height`` are required;
``steps``/``cfg``/``seed`` fall back to the CLI defaults the server was
started with):

    {
        "prompt":          "...",     // required
        "width":           int,       // required
        "height":          int,       // required
        "seed":            int | null,
        "steps":           int,
        "cfg":             float,
        "negative_prompt": "..."      // accepted but unused by this model
    }

``width``/``height`` are mapped to the closest supported bucket: the
base resolution is 1440 if either dimension is 1440, 1024 if either is
1024, otherwise 1024; the aspect ratio is the nearest match in
SUPPORTED_ASPECT_RATIOS (defaulting to 1:1).

Example:

    python microsoft-lens.py --model Lens-Turbo

    curl -X POST http://localhost:7802/generate \\
        -H 'Content-Type: application/json' \\
        -d '{"prompt": "A scenic landscape with a serene lake",
             "width": 1936, "height": 1088,
             "steps": 30, "cfg": 4.5, "seed": 42}'
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch

from lens import LensGptOssEncoder, LensPipeline
from lens.resolution import SUPPORTED_ASPECT_RATIOS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lens text-to-image inference HTTP server"
    )
    parser.add_argument("--model", type=str, default="Lens-Turbo",
                        help="One of 'Lens-Turbo' or 'Lens', or absolute path to model directory. Auto-downloads from Hugging Face if not local.")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Bind host for the HTTP server.")
    parser.add_argument("--port", type=int, default=7802,
                        help="Bind port for the HTTP server.")
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of denoising steps.")
    parser.add_argument("--cfg", type=float, default=5.0,
                        help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=None,
                        help="If set, reused for every request. Otherwise "
                             "each request uses a fresh random seed.")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--disable_mxfp4", action="store_true",
                        help="Disable dequantization of GPT-OSS text encoder.")
    parser.add_argument("--reasoner", action="store_true",
                        help="Enable prompt reasoner (uses local GPT-OSS unless "
                             "--api_url is set).")
    parser.add_argument("--api_url", type=str, default=None,
                        help="OpenAI-compatible base URL for the API reasoner.")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key for the OpenAI-compatible endpoint.")
    parser.add_argument("--api_model", type=str, default=None,
                        help="Model name to send to the OpenAI-compatible endpoint.")
    parser.add_argument("--offload", action="store_true",
                        help="Enable diffusers model CPU offload "
                             "(text_encoder->transformer->vae) to reduce peak VRAM.")
    return parser.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[name]


def _resolve_model(model: str) -> str:
    if os.path.isdir(model):
        return model
    local = os.path.join("./models", model)
    if os.path.isdir(local):
        return local
    return model if "/" in model else f"microsoft/{model}"


def _generate(pipe: LensPipeline, prompt: str, *,
              base_resolution: int, aspect_ratio: str,
              steps: int, cfg: float, seed: int | None,
              enable_reasoner: bool) -> str:
    generator = (
        torch.Generator(device=pipe._execution_device).manual_seed(int(seed))
        if seed is not None
        else None
    )

    out = pipe(
        prompt=[prompt],
        base_resolution=base_resolution,
        aspect_ratio=aspect_ratio,
        num_inference_steps=steps,
        guidance_scale=cfg,
        num_images_per_prompt=1,
        generator=generator,
        enable_reasoner=enable_reasoner,
    )

    images = list(out.images)
    if len(images) != 1:
        raise RuntimeError(
            f"Pipeline returned {len(images)} images; expected 1."
        )

    buf = io.BytesIO()
    images[0].save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _nearest_aspect_ratio(width: int, height: int) -> str:
    target = width / height
    best, best_diff = "1:1", float("inf")
    for ar in SUPPORTED_ASPECT_RATIOS:
        w_s, h_s = ar.split(":")
        diff = abs(int(w_s) / int(h_s) - target)
        if diff < best_diff:
            best, best_diff = ar, diff
    return best


def _pick_base_resolution(width: int, height: int) -> int:
    if 1440 in (width, height):
        return 1440
    if 1024 in (width, height):
        return 1024
    return 1024


def _resolve_request_params(data: dict, args: argparse.Namespace) -> dict:
    """Merge per-request overrides with CLI defaults, validating types/choices.

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

    width = _get("width", int, None, required=True)
    height = _get("height", int, None, required=True)
    if width < 1 or height < 1:
        raise ValueError("'width' and 'height' must be positive integers.")

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

    return {
        "base_resolution": _pick_base_resolution(width, height),
        "aspect_ratio": _nearest_aspect_ratio(width, height),
        "steps": steps,
        "cfg": cfg,
        "seed": seed,
        "enable_reasoner": args.reasoner,
    }


def _make_handler(pipe: LensPipeline, args: argparse.Namespace):
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
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                self._send_json(400, {"error": f"Invalid JSON body: {e}"})
                return

            prompt = data.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                self._send_json(400, {"error": "Missing or empty 'prompt' string."})
                return

            prompt = prompt.strip()

            try:
                gen_kwargs = _resolve_request_params(data, args)
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return

            try:
                image = _generate(pipe, prompt, **gen_kwargs)
            except Exception as e:
                print(f"[generate] failed: {e!r}")
                self._send_json(500, {"error": f"Generation failed: {e}"})
                return

            refined = getattr(pipe, "_last_refined_prompts", [prompt])
            if refined and refined[0] != prompt:
                print(f"[generate] refined prompt:\n  {prompt!r}\n    -> {refined[0]!r}")

            print(f"[generate] returning image :: {prompt!r}")
            self._send_json(200, {"image": image})

        def log_message(self, format: str, *log_args) -> None:
            print("[http] " + (format % log_args))

    return Handler


def main() -> None:
    args = parse_args()

    dtype = _torch_dtype(args.dtype)
    model_source = _resolve_model(args.model)
    print(f"[load] using model source: {model_source}")

    # Pre-load the text encoder so we can control MXFP4 dequantization. The
    # gpt-oss-20b weights on the hub are stored in MXFP4 and the loader will
    # try to keep them that way unless we pass an explicit Mxfp4Config that
    # asks for dequantization. MXFP4 kernels need Hopper-or-newer GPUs, so on
    # A100/V100 we may consider dequantizing to bf16/fp16.
    text_encoder_kwargs = {"subfolder": "text_encoder", "dtype": dtype}
    try:
        from transformers import Mxfp4Config
        text_encoder_kwargs["quantization_config"] = Mxfp4Config(
            dequantize=args.disable_mxfp4
        )
    except ImportError:
        # Older transformers without Mxfp4Config: nothing to do, the weights
        # will load as the loader sees fit.
        pass
    text_encoder = LensGptOssEncoder.from_pretrained(
        model_source, **text_encoder_kwargs
    )

    pipe = LensPipeline.from_pretrained(
        model_source, text_encoder=text_encoder, torch_dtype=dtype
    )

    if args.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    if args.api_url or args.api_key or args.api_model:
        pipe.reasoner.openai_base_url = args.api_url
        pipe.reasoner.openai_api_key = args.api_key
        pipe.reasoner.openai_model = args.api_model

    handler_cls = _make_handler(pipe, args)
    server = HTTPServer((args.host, args.port), handler_cls)

    print(f"\nReady. Listening on http://{args.host}:{args.port}/generate. "
          f"POST {{\"prompt\": \"...\"}} to generate. Ctrl-C to exit.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
