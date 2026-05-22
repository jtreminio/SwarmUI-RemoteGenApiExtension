"""Remote text-to-image generation server for SenseNova-U1.

Loads the SenseNova-U1 MoT pipeline once and serves the standard
``POST /generate`` contract over stdlib ``http.server``. Mirrors the
single-shot CLI script ``examples/t2i/inference.py`` — same load-time
options, same defaults, same kwargs into ``t2i_generate``.

Model resolution
----------------
``--model_path`` accepts a bare checkpoint name
(``SenseNova-U1-8B-MoT-SFT``), a full HF id
(``sensenova/SenseNova-U1-8B-MoT``), or a local directory path. The
local cache lives at ``./sensenova/<name>``, which is also the HF
namespace for the official checkpoints — so a downloaded model and a
HF-id reference resolve to the same directory.

Supported main checkpoints (sensenova/ namespace):

  - SenseNova-U1-8B-MoT
  - SenseNova-U1-8B-MoT-SFT
  - SenseNova-U1-8B-MoT-Infographic

``--lora_path`` accepts a bare LoRA name
(``SenseNova-U1-8B-MoT-LoRA-8step-V1.0``), a filename with
``.safetensors``, or a path. Bare names are fetched from
``sensenova/SenseNova-U1-8B-MoT-LoRAs`` into ``./sensenova/``.

Width/height mapping
--------------------
``width`` / ``height`` are passed straight through to
``t2i_generate``. The model was trained on a fixed set of buckets
(1:1 -> 2048x2048, 16:9 -> 2720x1536, etc.); requests outside that set
log a one-line stderr warning but are still attempted. See
``SUPPORTED_RESOLUTIONS`` below for the full list.

Thinking mode
-------------
Per-request ``thinking`` (bool) maps to ``think_mode`` of
``t2i_generate``. When true the model first emits a ``<think>...</think>``
block; the decoded reasoning text is included in the response as
``{"image": ..., "think": "..."}``. When false (default) the response
is the usual ``{"image": ...}``. The CLI flag ``--thinking`` sets the
fallback for requests that omit the field.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import secrets
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sensenova_u1
from sensenova_u1.utils import (
    DEFAULT_VRAM_MODE,
    add_offload_args,
    best_available_device,
    load_and_merge_lora_weight_from_safetensors,
    load_model_and_tokenizer,
    make_offload_ctx,
    vram_mode_to_prefetch_count,
)


NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

# Resolution buckets the model was trained on. Requests outside this set
# are still attempted but log a warning. Mirrors examples/t2i/inference.py.
SUPPORTED_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1:1": (2048, 2048),
    "16:9": (2720, 1536),
    "9:16": (1536, 2720),
    "3:2": (2496, 1664),
    "2:3": (1664, 2496),
    "4:3": (2368, 1760),
    "3:4": (1760, 2368),
    "1:2": (1440, 2880),
    "2:1": (2880, 1440),
    "1:3": (1152, 3456),
    "3:1": (3456, 1152),
}


def _denorm(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(NORM_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def _to_pil(batch: torch.Tensor) -> list[Image.Image]:
    arr = _denorm(batch.float()).permute(0, 2, 3, 1).cpu().numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return [Image.fromarray(a) for a in arr]


class SenseNovaU1T2I:
    """Thin wrapper around ``AutoModel.from_pretrained``.

    Mirrors ``examples/t2i/inference.py::SenseNovaU1T2I``. Because
    ``sensenova_u1`` registers the config / model with transformers at import
    time, no ``trust_remote_code=True`` is needed.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        gguf_checkpoint: str | None = None,
        vram_mode: str = DEFAULT_VRAM_MODE,
        device_map: str | None = None,
        max_memory: str | None = None,
    ) -> None:
        self.device = device
        self._last_think_text: str = ""
        self.vram_mode = vram_mode
        self.prefetch_count = vram_mode_to_prefetch_count(vram_mode)
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_path,
            dtype=dtype,
            device=device,
            gguf_checkpoint=gguf_checkpoint,
            for_offload=self.prefetch_count > 0,
            device_map=device_map,
            max_memory=max_memory,
        )

    def _offload_ctx(self):
        return make_offload_ctx(self.model, self.prefetch_count, self.device)

    @property
    def last_think_text(self) -> str:
        return self._last_think_text

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        image_size: tuple[int, int],
        cfg_scale: float,
        cfg_norm: str,
        timestep_shift: float,
        cfg_interval: tuple[float, float],
        num_steps: int,
        seed: int,
        think_mode: bool,
    ) -> list[Image.Image]:
        with self._offload_ctx() as offloaded:
            out = offloaded.t2i_generate(
                self.tokenizer,
                prompt,
                image_size=image_size,
                cfg_scale=cfg_scale,
                cfg_norm=cfg_norm,
                timestep_shift=timestep_shift,
                cfg_interval=cfg_interval,
                num_steps=num_steps,
                batch_size=1,
                seed=seed,
                think_mode=think_mode,
            )
        if think_mode:
            tensor, think_text = out
            self._last_think_text = think_text
        else:
            tensor = out
            self._last_think_text = ""
        return _to_pil(tensor)


MODELS_DIR = Path("sensenova")
DEFAULT_MODEL_HF_OWNER = "sensenova"
DEFAULT_LORA_HF_REPO = "sensenova/SenseNova-U1-8B-MoT-LoRAs"


def _resolve_repo_dir(repo: str) -> str:
    """Locate a model snapshot directory, downloading it if missing.

    Resolution order:
      1. ``repo`` itself is an existing directory.
      2. ``./sensenova/<basename>`` exists.
      3. Download via ``snapshot_download`` into ``./sensenova/<basename>``.
         A name containing ``/`` is used as the HF repo id verbatim; a bare
         name is fetched from ``sensenova/<name>``.
    """
    direct = Path(repo)
    if direct.is_dir():
        return str(direct.resolve())

    basename = direct.name
    cached = MODELS_DIR / basename
    if cached.is_dir():
        return str(cached.resolve())

    from huggingface_hub import snapshot_download

    repo_id = repo if "/" in repo else f"{DEFAULT_MODEL_HF_OWNER}/{repo}"
    cached.mkdir(parents=True, exist_ok=True)
    sys.stderr.write(f"[download] {repo_id} -> {cached}/\n")
    return snapshot_download(repo_id=repo_id, local_dir=str(cached))


def _resolve_lora_file(lora: str) -> str:
    """Locate a LoRA ``.safetensors`` file, downloading it if missing.

    Accepts a bare name (``.safetensors`` is appended), a filename, a
    ``<repo_owner>/<repo_name>/<filename>`` reference, or an existing path.
    Bare names default to ``sensenova/SenseNova-U1-8B-MoT-LoRAs``. Cached
    files live next to model snapshots under ``./sensenova/``.
    """
    direct = Path(lora)
    if direct.is_file():
        return str(direct.resolve())

    filename = lora if lora.endswith(".safetensors") else f"{lora}.safetensors"

    cached = MODELS_DIR / Path(filename).name
    if cached.is_file():
        return str(cached.resolve())

    from huggingface_hub import hf_hub_download

    if "/" in filename:
        repo_id, fname = filename.rsplit("/", 1)
    else:
        repo_id, fname = DEFAULT_LORA_HF_REPO, filename

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    sys.stderr.write(f"[download] {repo_id}/{fname} -> {MODELS_DIR}/\n")
    return hf_hub_download(repo_id=repo_id, filename=fname, local_dir=str(MODELS_DIR))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SenseNova-U1 T2I HTTP gen API server.")
    p.add_argument(
        "--model_path",
        default="SenseNova-U1-8B-MoT",
        help=(
            "Checkpoint name, HF id, or local directory. Bare names are fetched from "
            "sensenova/<name> into ./sensenova/<name>. Known names: "
            "SenseNova-U1-8B-MoT, SenseNova-U1-8B-MoT-SFT, SenseNova-U1-8B-MoT-Infographic."
        ),
    )
    p.add_argument(
        "--lora_path",
        default=None,
        help=(
            "Optional LoRA. Bare names (e.g. SenseNova-U1-8B-MoT-LoRA-8step-V1.0) are "
            f"fetched from {DEFAULT_LORA_HF_REPO}/. Also accepts a .safetensors filename "
            "or a local path."
        ),
    )

    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7802)

    p.add_argument(
        "--device",
        default=str(best_available_device()),
        help="Compute device, e.g. 'cuda', 'cuda:0', 'xpu', 'cpu'.",
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    add_offload_args(p)
    p.add_argument(
        "--gguf_checkpoint",
        default=None,
        help="Optional .gguf quantized checkpoint (requires the [gguf] extra).",
    )
    p.add_argument(
        "--attn_backend",
        default="auto",
        choices=["auto", "flash", "sdpa"],
        help="Attention kernel for Qwen3 layers. 'auto' picks flash-attn when available.",
    )

    p.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Per-request fallback when 'steps' is omitted.",
    )
    p.add_argument(
        "--cfg_scale",
        type=float,
        default=4.0,
        help="Per-request fallback when 'cfg' is omitted.",
    )
    p.add_argument(
        "--cfg_norm",
        default="none",
        choices=["none", "global", "channel", "cfg_zero_star"],
        help="Classifier-free guidance rescaling mode (server-side; not per-request).",
    )
    p.add_argument("--timestep_shift", type=float, default=3.0)
    p.add_argument(
        "--cfg_interval",
        type=float,
        nargs=2,
        default=[0.0, 1.0],
        metavar=("LO", "HI"),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Per-request fallback when 'seed' is omitted. Omit for a fresh random seed each call.",
    )
    p.add_argument(
        "--thinking",
        action="store_true",
        help=(
            "Default value for the per-request 'thinking' field. When true, requests "
            "without an explicit 'thinking' field run with reasoning enabled."
        ),
    )
    return p.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _warn_if_unsupported(width: int, height: int) -> None:
    if (width, height) in SUPPORTED_RESOLUTIONS.values():
        return
    sys.stderr.write(
        f"[warn] ({width}x{height}) is outside the trained resolution set; quality may degrade.\n"
    )


def _generate(
    engine: SenseNovaU1T2I,
    prompt: str,
    *,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
    thinking: bool,
    cfg_norm: str,
    timestep_shift: float,
    cfg_interval: tuple[float, float],
) -> tuple[str, str]:
    """Run one generation and return (base64 PNG, think text).

    ``think text`` is the empty string when ``thinking`` is false.
    """
    images = engine.generate(
        prompt,
        image_size=(width, height),
        cfg_scale=cfg,
        cfg_norm=cfg_norm,
        timestep_shift=timestep_shift,
        cfg_interval=cfg_interval,
        num_steps=steps,
        seed=seed,
        think_mode=thinking,
    )
    if len(images) != 1:
        raise RuntimeError(f"Pipeline returned {len(images)} images; expected 1.")

    buf = io.BytesIO()
    images[0].save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii"), engine.last_think_text


def _resolve_request_params(data: dict, args: argparse.Namespace) -> dict:
    """Validate JSON body, raise ``ValueError`` with a user-facing message on bad input."""

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
    if width < 1 or height < 1:
        raise ValueError("'width' and 'height' must be positive integers.")

    steps = _get("steps", int, args.num_steps)
    if steps < 1:
        raise ValueError("'steps' must be >= 1.")

    cfg = _get("cfg", float, args.cfg_scale)

    if "seed" in data and data["seed"] is not None:
        seed = _get("seed", int, args.seed)
    else:
        seed = args.seed
    if seed is None:
        # t2i_generate requires an int seed; mint a fresh one when the
        # request and server default are both unset.
        seed = secrets.randbits(63)

    if "thinking" in data:
        thinking_val = data["thinking"]
        if not isinstance(thinking_val, bool):
            raise ValueError(
                f"'thinking' must be bool, got {type(thinking_val).__name__}."
            )
        thinking = thinking_val
    else:
        thinking = args.thinking

    return {
        "prompt": prompt.strip(),
        "width": width,
        "height": height,
        "steps": steps,
        "cfg": cfg,
        "seed": seed,
        "thinking": thinking,
    }


def _make_handler(engine: SenseNovaU1T2I, args: argparse.Namespace):
    cfg_interval = tuple(args.cfg_interval)

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

            _warn_if_unsupported(params["width"], params["height"])

            try:
                image_b64, think_text = _generate(
                    engine,
                    params["prompt"],
                    width=params["width"],
                    height=params["height"],
                    steps=params["steps"],
                    cfg=params["cfg"],
                    seed=params["seed"],
                    thinking=params["thinking"],
                    cfg_norm=args.cfg_norm,
                    timestep_shift=args.timestep_shift,
                    cfg_interval=cfg_interval,
                )
            except Exception as e:
                print(f"[generate] failed: {e!r}")
                self._send_json(500, {"error": f"Generation failed: {e}"})
                return

            print(f"[generate] returning image :: {params['prompt']!r}")
            response: dict = {"image": image_b64}
            if params["thinking"]:
                response["think"] = think_text
            self._send_json(200, response)

        def do_GET(self):
            self._send_json(404, {"error": f"Unknown path: {self.path}"})

    return Handler


def main() -> None:
    args = parse_args()
    dtype = _torch_dtype(args.dtype)

    sensenova_u1.set_attn_backend(args.attn_backend)
    print(
        f"[attn] backend={args.attn_backend!r} "
        f"(effective={sensenova_u1.effective_attn_backend()!r})"
    )

    model_path = _resolve_repo_dir(args.model_path)
    print(f"[load] model -> {model_path}")

    engine = SenseNovaU1T2I(
        model_path,
        device=args.device,
        dtype=dtype,
        gguf_checkpoint=args.gguf_checkpoint,
        vram_mode=args.vram_mode,
        device_map=args.device_map,
        max_memory=args.max_memory,
    )

    if args.lora_path is not None:
        lora_path = _resolve_lora_file(args.lora_path)
        print(f"[load] lora -> {lora_path}")
        engine.model = load_and_merge_lora_weight_from_safetensors(engine.model, lora_path)

    handler_cls = _make_handler(engine, args)
    server = HTTPServer((args.host, args.port), handler_cls)

    print(
        f"\nReady. Listening on http://{args.host}:{args.port}/generate. "
        f"POST {{\"prompt\": \"...\", \"width\": W, \"height\": H}} to generate. "
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
