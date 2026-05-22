"""Remote text-to-image generation server for Z-Image-Turbo (L2P merge).

Loads ``ZImagePipeline`` once and serves the standard ``POST /generate``
contract over stdlib ``http.server``.

Width/height mapping
--------------------
The DiT requires both dimensions to be divisible by 16 **and** the
patch grid ``(W/16) * (H/16)`` to be a multiple of 32 (otherwise the
internal sequence padding breaks the final feature-map reshape).
Incoming ``width``/``height`` are rounded **up** to the smallest pair
that satisfies both constraints; the adjustment is logged to stderr.
"""

import argparse
import base64
import io
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from math import gcd
from pathlib import Path

import torch

from diffsynth.pipelines.z_image_L2P import ModelConfig, ZImagePipeline


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MAIN_MODEL = str(SCRIPT_DIR / "model-1k-merge.safetensors")
DEFAULT_TEXT_ENCODER_DIR = str(SCRIPT_DIR / "Z-Image-Turbo" / "text_encoder")
DEFAULT_TOKENIZER_DIR = str(SCRIPT_DIR / "Z-Image-Turbo" / "tokenizer")
DEFAULT_TEXT_ENCODER_SHARDS = [
    f"{DEFAULT_TEXT_ENCODER_DIR}/model-00001-of-00003.safetensors",
    f"{DEFAULT_TEXT_ENCODER_DIR}/model-00002-of-00003.safetensors",
    f"{DEFAULT_TEXT_ENCODER_DIR}/model-00003-of-00003.safetensors",
]


def parse_args():
    p = argparse.ArgumentParser(description="Z-Image-Turbo gen API server.")
    p.add_argument("--main_model", default=DEFAULT_MAIN_MODEL,
                   help="Path to the main DiT safetensors file.")
    p.add_argument("--text_encoder", nargs="+", default=DEFAULT_TEXT_ENCODER_SHARDS,
                   help="One or more text-encoder safetensors shard paths.")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_DIR,
                   help="Path to the tokenizer directory.")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="cuda",
                   help="Device to load the pipeline onto.")
    p.add_argument("--rand_device", default="cuda",
                   help="Device used by the pipeline's RNG.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7802)
    p.add_argument("--steps", type=int, default=30,
                   help="Per-request fallback when 'steps' is omitted.")
    p.add_argument("--cfg", type=float, default=2.0,
                   help="Per-request fallback when 'cfg' is omitted.")
    p.add_argument("--seed", type=int, default=None,
                   help="Per-request fallback when 'seed' is omitted (omit for random).")
    return p.parse_args()


def _torch_dtype(name):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


PATCH = 16
SEQ_MULTI_OF = 32


def _align_dimensions(width, height):
    """Round up to the smallest (W, H) where both are %16 and (W/16)*(H/16) is %32."""
    w16 = (width + PATCH - 1) // PATCH
    h16 = (height + PATCH - 1) // PATCH
    if (w16 * h16) % SEQ_MULTI_OF == 0:
        return w16 * PATCH, h16 * PATCH

    need_w_mult = SEQ_MULTI_OF // gcd(h16, SEQ_MULTI_OF)
    new_w16 = ((w16 + need_w_mult - 1) // need_w_mult) * need_w_mult
    need_h_mult = SEQ_MULTI_OF // gcd(w16, SEQ_MULTI_OF)
    new_h16 = ((h16 + need_h_mult - 1) // need_h_mult) * need_h_mult

    cost_w = (new_w16 - w16) * h16
    cost_h = w16 * (new_h16 - h16)
    if cost_w <= cost_h:
        return new_w16 * PATCH, h16 * PATCH
    return w16 * PATCH, new_h16 * PATCH


def _generate(pipe, prompt, *, width, height, steps, cfg, seed, negative_prompt, rand_device):
    kwargs = dict(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        cfg_scale=cfg,
        seed=seed,
        rand_device=rand_device,
    )
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
    image = pipe(**kwargs)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _resolve_request_params(data, args):
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

    prompt = _get("prompt", str, None, required=True)
    if not prompt.strip():
        raise ValueError("'prompt' must be a non-empty string.")

    width = _get("width", int, None, required=True)
    height = _get("height", int, None, required=True)
    if width <= 0 or height <= 0:
        raise ValueError("'width' and 'height' must be positive integers.")

    aligned_w, aligned_h = _align_dimensions(width, height)
    if (aligned_w, aligned_h) != (width, height):
        sys.stderr.write(
            f"[align] {width}x{height} -> {aligned_w}x{aligned_h} "
            f"(grid {aligned_w // PATCH}x{aligned_h // PATCH}, "
            f"product {(aligned_w // PATCH) * (aligned_h // PATCH)})\n"
        )
    width, height = aligned_w, aligned_h

    steps = _get("steps", int, args.steps)
    if steps is not None and steps < 1:
        raise ValueError("'steps' must be >= 1.")

    cfg = _get("cfg", float, args.cfg)

    if "seed" in data and data["seed"] is None:
        seed = None
    else:
        seed = _get("seed", int, args.seed)

    negative_prompt = ""
    if "negative_prompt" in data:
        np_val = data["negative_prompt"]
        if not isinstance(np_val, str):
            raise ValueError(
                f"'negative_prompt' must be str, got {type(np_val).__name__}."
            )
        negative_prompt = np_val

    return {
        "prompt": prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg": cfg,
        "seed": seed,
        "negative_prompt": negative_prompt,
    }


def _make_handler(pipe, args):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *a):
            sys.stderr.write("[http] " + (format % a) + "\n")

        def _send_json(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path.rstrip("/") not in ("", "/generate"):
                self._send_json(404, {"error": f"Unknown path: {self.path}"})
                return

            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                data = json.loads(raw.decode("utf-8")) if raw else {}
                if not isinstance(data, dict):
                    raise ValueError("Request body must be a JSON object.")
            except (ValueError, json.JSONDecodeError) as e:
                self._send_json(400, {"error": f"Invalid JSON body: {e}"})
                return

            try:
                params = _resolve_request_params(data, args)
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return

            try:
                image_b64 = _generate(
                    pipe,
                    params["prompt"],
                    width=params["width"],
                    height=params["height"],
                    steps=params["steps"],
                    cfg=params["cfg"],
                    seed=params["seed"],
                    negative_prompt=params["negative_prompt"],
                    rand_device=args.rand_device,
                )
            except Exception as e:
                sys.stderr.write(f"[generate] failed: {e!r}\n")
                traceback.print_exc()
                self._send_json(500, {"error": f"Generation failed: {e}"})
                return

            sys.stderr.write(f"[generate] returning image :: {params['prompt']!r}\n")
            self._send_json(200, {"image": image_b64})

        def do_GET(self):
            self._send_json(404, {"error": f"Unknown path: {self.path}"})

    return Handler


def main():
    args = parse_args()
    dtype = _torch_dtype(args.dtype)

    print(f"Loading Z-Image pipeline (main_model={args.main_model}, dtype={args.dtype}, device={args.device}) ...")
    pipe = ZImagePipeline.from_pretrained(
        torch_dtype=dtype,
        device=args.device,
        model_configs=[
            ModelConfig(path=[args.main_model]),
            ModelConfig(path=list(args.text_encoder)),
        ],
        tokenizer_config=ModelConfig(path=args.tokenizer),
    )

    handler_cls = _make_handler(pipe, args)
    server = HTTPServer((args.host, args.port), handler_cls)
    print(f"Ready. Listening on http://{args.host}:{args.port}/generate.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
