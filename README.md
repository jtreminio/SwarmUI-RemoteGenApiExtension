# SwarmUI RemoteGenApi

This SwarmUI extension allows you to make an API request to a remote server for image generation.

The extension came about because I wanted to run [Microsoft Lens](https://huggingface.co/microsoft/Lens)
before initial ComfyUI support was merged for it, but the only way to do this was through diffusers.
Now with this extension you can point SwarmUI to a remote server that is running whatever fancy new image
model you want to try out, and get back an image which you can then save as-is via SwarmUI, or use it in a
regular workflow like refining, upscaling, editing, etc.

This extension does require some programming knowledge, at least to get the initial server wrapper built.
I've included the [examples/microsoft-lens.py](examples/microsoft-lens.py) file if you'd like to try out
Microsoft Lens. As I play with more new image models I will continue adding more examples you can use yourself.

## Get Started

You can use `0.0.0.0` instead of `127.0.0.1` in instructions below, if you need to access
the server from a different machine. Note that this is potentially unsafe since it opens
the server to all internet traffic.

Then on SwarmUI under the "Remote Gen API URL" textbox, paste `http://127.0.0.1:7802/generate`.

* This extension replaces ONLY THE BASE IMAGE STAGE. It cannot currently be used for Refiner or Edit image stages
* You must still select an image model for your base stage. IT WILL BE IGNORED. This just makes the extension play nice with SwarmUI
* You can upscale/refine/etc as normal afterward
* seed/steps/cfg/width/height are all applied
    * if you create a custom server, you must implement these yourself

For the server files, some sane defaults have been selected:
* `--host` defaults to `127.0.0.1`
* `--port` defaults to `7802`

### Microsoft Lens

For [Microsoft Lens](https://github.com/microsoft/Lens), follow their installation instructions.
Copy the [examples/microsoft-lens.py](examples/microsoft-lens.py) file to `server.py` repo root.

* For Lens-Turbo, run it with `python server.py --model Lens-Turbo`.
* For Lens (non-Turbo), run it with `python server.py --model Lens`.

## TencentYoutuResearch/T2I-L2P (Z-Image Pixel Space)
For [TencentYoutuResearch/T2I-L2P](https://github.com/TencentYoutuResearch/T2I-L2P), follow their installation instructions.
Copy the [examples/tencentyouturesearch-T2I-L2P.py](examples/tencentyouturesearch-T2I-L2P.py) file to `server.py` repo root.

* Run it with `python server.py`

### OpenSenseNova/SenseNova-U1

For [OpenSenseNova/SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1), follow their installation instructions.
Copy the [examples/opensensenova-sensenova-u1.py](examples/opensensenova-sensenova-u1.py) file to `server.py` repo root.

Run it with any of the following:

* `python server.py --model SenseNova-U1-8B-MoT-Infographic` (default)
* `python server.py --model SenseNova-U1-8B-MoT-SFT`
* `python server.py --model SenseNova-U1-8B-MoT	8B`
* `python server.py --model SenseNova-U1-A3B-MoT-SFT`
* `python server.py --model SenseNova-U1-A3B-MoT`

# Development

## Use ComfyTyped

### Generate node definitions with ComfyTyped
```
cd /path/to/ComfyTyped
dotnet build -c Release ComfyTyped.csproj
cp bin/Release/net8.0/ComfyTyped.dll \
    ../SwarmUI-RemoteGenApiExtension/lib/ComfyTyped.dll

dotnet run --project tools/ComfyTyped.CodeGen -- \
    --comfy-json http://0.0.0.0:7801/ComfyBackendDirect/api/object_info \
    --output ../SwarmUI-RemoteGenApiExtension/src/Generated \
    --namespace RemoteGenApiExtension.Generated \
    --keep-list ../SwarmUI-RemoteGenApiExtension/comfytyped.keep.json \
    --core-assembly ../SwarmUI-RemoteGenApiExtension/lib/ComfyTyped.dll
```

### Once ready to commit, prune unused node definitions
```
cd /path/to/ComfyTyped
dotnet run --project tools/ComfyTyped.CodeGen -- prune \
    --generated-dir ../SwarmUI-RemoteGenApiExtension/src/Generated \
    --source ../SwarmUI-RemoteGenApiExtension
```
