"""ComfyUI node package for SwarmUI Remote Gen API."""

from comfy_api.latest import ComfyExtension, io

from .SwarmRemoteGenApi import SwarmRemoteGenApi


class SwarmRemoteGenApiExtension(ComfyExtension):
    """Extension entrypoint exposing SwarmUI Remote Gen API nodes."""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SwarmRemoteGenApi]


async def comfy_entrypoint() -> ComfyExtension:
    """Create the extension instance for ComfyUI runtime loading."""
    return SwarmRemoteGenApiExtension()


NODE_CLASS_MAPPINGS = {
    "SwarmRemoteGenApi": SwarmRemoteGenApi,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SwarmRemoteGenApi": "Swarm Remote Gen API",
}

__all__ = [
    "SwarmRemoteGenApiExtension",
    "comfy_entrypoint",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
