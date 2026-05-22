# Building a Remote Gen API Server for a New Model

Use these instructions to produce a `gen_api_server.py` for a new
text-to-image model. Every server must expose the **same API contract**
described below — only the model-loading and the
width/height-to-pipeline mapping change per model.

## Inputs you'll be given

1. A working **single-shot CLI inference script** for the target model
   (an `inference.py`-style file). Treat this as the source of truth
   for: how to load the pipeline, what its `__call__` accepts, any
   quantization/offload/reasoner setup, and reasonable default values
   for steps/cfg/dtype/etc.
2. Knowledge of how the model's pipeline parameterizes resolution
   (direct `width`/`height`, bucketed, must be divisible by N, etc.)
   and whether its `__call__` supports `negative_prompt`.

## The fixed API contract

All servers expose the same API regardless of the underlying model.

### Endpoint
`POST /generate` — any other path returns 404.

### Request body (JSON)

| Field             | Type        | Required | Notes |
|-------------------|-------------|----------|-------|
| `prompt`          | string      | yes      | Non-empty after `.strip()`. |
| `width`           | int         | yes      | Positive integer. |
| `height`          | int         | yes      | Positive integer. |
| `steps`           | int         | no       | Falls back to server CLI default. Must be ≥ 1. |
| `cfg`             | float       | no       | Falls back to server CLI default. Accept int as float. |
| `seed`            | int \| null | no       | Falls back to server CLI default. Reject bools. |
| `negative_prompt` | string      | no       | Pass through if the pipeline supports it; otherwise silently ignore — **do not log**. |

Unknown fields are ignored. Reject `bool` for any int/float field
(`isinstance(True, int)` is `True` in Python — explicit check needed).

### Responses

- **200** — `{"image": "<base64-encoded PNG>"}`. Exactly one image per request.
- **400** — `{"error": "..."}` for bad JSON, missing/invalid fields.
- **404** — `{"error": "Unknown path: ..."}`.
- **500** — `{"error": "Generation failed: ..."}` when the pipeline raises.

### Logging

- `[generate] returning image :: <prompt!r>` on success.
- `[generate] failed: <repr>` on pipeline error.
- `[http] ...` for default access logs (override `log_message`).

## Building the server

### Skeleton

Mirror the structure of `examples/gen_api_server.py`:

```
parse_args()                       # CLI: server + model load-time options
_torch_dtype(name)
_generate(pipe, prompt, *, ...)    # one call -> base64 PNG string
_resolve_request_params(data, args)# validate JSON body, raise ValueError
_make_handler(pipe, args)          # BaseHTTPRequestHandler subclass
main()                             # load pipeline once, serve_forever
```

Use stdlib `http.server`, `json`, `base64` only. No Flask/FastAPI/uvicorn.

### CLI args

Keep from the source inference script (load-time only):
- `--repo_id` (or whatever points at the model weights)
- `--dtype`
- All model-loading quirks: quantization configs, offload, reasoner
  flags, API keys, etc.

Add:
- `--host` (default `0.0.0.0`)
- `--port` (default `7802`)

Keep as **per-request fallback defaults**:
- `--steps`, `--cfg`, `--seed`

Remove (now per-request or unused for a server):
- `--prompt`, `--n`, `--out`
- Any resolution/aspect args replaced by per-request `width`/`height`.

### Mapping API `width`/`height` to the pipeline

This is the per-model part. Pick the case that fits and document the
mapping in the module docstring so callers know how their dims will be
honored.

- **Direct width/height** — pass straight through.
- **Bucketed (Lens-style)** — pick the nearest supported bucket. The
  Lens server uses:
  - `base_resolution = 1440 if 1440 in (w,h) else 1024 if 1024 in (w,h) else 1024`
  - `aspect_ratio = argmin over SUPPORTED_ASPECT_RATIOS of |W/H − w/h|`,
    fallback `1:1`.
- **Must be divisible by N** — round per the model's rule, mention the
  rounding in the docstring.

### Handling `negative_prompt`

- If the pipeline's `__call__` accepts `negative_prompt`: pass it
  through when present and non-empty; omit otherwise so the pipeline
  default applies.
- If it doesn't: silently ignore. No log line, no warning, no error.

### Generation invariants

- Always `num_images_per_prompt=1`.
- Seed: `None` → no generator (random per call); int → `torch.Generator(device=pipe._execution_device).manual_seed(int(seed))`.
- Encode the PIL image as PNG into a `BytesIO`, then
  `base64.b64encode(buf.getvalue()).decode("ascii")`.

### Validation helper (copy as-is)

```python
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
```

`_resolve_request_params` raises `ValueError`; the handler catches it
and returns 400 with the message.

### Handler

```python
def do_POST(self):
    if self.path.rstrip("/") not in ("", "/generate"):
        self._send_json(404, {"error": f"Unknown path: {self.path}"})
        return
    # read body -> json.loads -> validate prompt -> _resolve_request_params
    # -> _generate -> 200 {"image": ...}
```

Override `log_message` to prefix `[http] ` so access logs are uniform.

### Pipeline lifetime

- Load the pipeline **once** in `main()` before
  `HTTPServer(...).serve_forever()`.
- Move to CUDA **or** enable model CPU offload — never both.
- Wrap `serve_forever()` in `try / except KeyboardInterrupt / finally:
  server.server_close()` for clean shutdown.

## Reference request

```bash
curl -X POST http://localhost:7802/generate \
    -H 'Content-Type: application/json' \
    -d '{"prompt": "a cat sitting on a roof",
         "width": 1024, "height": 1024,
         "steps": 30, "cfg": 4.5, "seed": 42}'
```

## Acceptance checklist

Before declaring the new server done, verify:

- [ ] `POST /generate` with the reference request returns 200 + base64 PNG.
- [ ] Missing `prompt` / `width` / `height` returns 400 with descriptive error.
- [ ] `seed: true` / `steps: "20"` / `cfg: false` all return 400 (bool/type rejection).
- [ ] Unknown path returns 404.
- [ ] Pipeline exception returns 500 with `Generation failed:` prefix.
- [ ] `negative_prompt` in the body never appears in logs.
- [ ] Server prints `Ready. Listening on http://HOST:PORT/generate.` after load.
- [ ] CTRL-C exits cleanly (no traceback).
- [ ] Module docstring documents the width/height → pipeline mapping for this model.
