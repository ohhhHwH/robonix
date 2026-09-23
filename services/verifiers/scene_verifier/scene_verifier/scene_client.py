"""Fetch and validate robot context from the Scene MCP service."""

import asyncio
import logging

from .core import parse_json_object, parse_robot_context, require_object

SCENE_CONTRACT = "robonix/system/scene/get_robot_context"
log = logging.getLogger("scene_verifier")


def decode_mcp_response(result) -> dict:
    """Decode a Scene response and unwrap its optional result envelope."""
    if getattr(result, "isError", False):
        raise RuntimeError(
            "Scene get_robot_context returned an MCP tool error: "
            f"{getattr(result, 'content', None)!r}"
        )

    structured = getattr(result, "structuredContent", None)

    if structured is not None:
        raw = require_object(
            structured,
            "Scene structuredContent",
        )
    else:
        blocks = getattr(result, "content", None)
        if (
            blocks is None
            or len(blocks) != 1
            or getattr(blocks[0], "type", None) != "text"
        ):
            raise ValueError("Scene must return one JSON object")

        raw = parse_json_object(
            blocks[0].text,
            "Scene response",
        )

    # Accept both a direct Scene object and {"result": SceneObject}.
    # Keep field validation in parse_robot_context().
    if "pose_known" not in raw and "result" in raw:
        raw = require_object(
            raw["result"],
            "Scene response.result",
        )

    return raw


async def fetch_robot_context(
    consumer_id: str,
    scene_provider_id: str,
    timeout_s: float,
):
    """Fetch Scene context and parse it after closing the MCP session.

    Imports remain local so pure tests do not require generated modules.
    The timeout covers MCP observation, but not synchronous Atlas calls.
    """
    import httpx
    from robonix_api import ATLAS, Transport
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    def _local_mcp_client(**kwargs):
        # The MCP endpoint Atlas hands us is always local (127.0.0.1).
        # Disable ambient proxy routing so a host shell that exports
        # `all_proxy=socks://…` (which httpx rejects) cannot break the
        # verification handshake. Mirrors the grpc proxy-bypass precedent.
        return httpx.AsyncClient(trust_env=False, **kwargs)

    with ATLAS.connect_capability(
        consumer_id=consumer_id,
        provider_id=scene_provider_id,
        contract_id=SCENE_CONTRACT,
        transport=Transport.MCP,
    ) as channel:

        # Scene declares its MCP endpoint with a trailing slash
        # (e.g. http://127.0.0.1:56279/mcp/), but its streamable-HTTP app
        # mounts at the slash-less path and answers the slashed URL with a
        # 307 redirect.  httpx does not follow redirects by default, so the
        # MCP client's raise_for_status() turns that redirect into a hard
        # error.  Normalise the endpoint once so the POST lands on the real
        # path and never triggers the redirect.
        endpoint = channel.endpoint.rstrip("/")

        async def observe():
            """Return the MCP response after closing its session."""
            async with streamablehttp_client(
                endpoint,
                httpx_client_factory=_local_mcp_client,
            ) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        "get_robot_context", {}
                    )

                    log.info(
                        "Scene MCP response: "
                        "isError=%r structuredContent=%r content=%r",
                        getattr(result, "isError", None),
                        getattr(result, "structuredContent", None),
                        getattr(result, "content", None),
                    )

            return result

        result = await asyncio.wait_for(observe(), timeout=timeout_s)

    raw = decode_mcp_response(result)

    log.info(
        "Scene decoded response: type=%s value=%r",
        type(raw).__name__,
        raw,
    )

    log.info(
        "Scene pose_known: present=%s type=%s value=%r",
        "pose_known" in raw,
        type(raw.get("pose_known")).__name__,
        raw.get("pose_known"),
    )

    return parse_robot_context(raw)