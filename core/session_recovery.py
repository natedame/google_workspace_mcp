"""
Session Recovery Module for MCP Streamable HTTP Transport

This module patches the MCP library's StreamableHTTPSessionManager to handle
unknown session IDs gracefully. Instead of returning a 400 error when a client
sends a stale session ID (e.g., after server restart), it creates a new session.

This is a common issue when:
1. The MCP server restarts (all in-memory sessions are cleared)
2. Clients (like Claude Code) cache session IDs and send stale ones
3. The server rejects the stale ID instead of recovering gracefully

The patch intercepts the session handling and recovers by:
1. Detecting when a request has an unknown session ID
2. Logging a warning about the stale session
3. Creating a new session instead of returning an error
"""

import logging
from http import HTTPStatus
from uuid import uuid4

from anyio.abc import TaskStatus
import anyio
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import Receive, Scope, Send

logger = logging.getLogger(__name__)

_original_handle_stateful_request = None


async def _patched_handle_stateful_request(
    self,
    scope: Scope,
    receive: Receive,
    send: Send,
) -> None:
    """
    Patched version of _handle_stateful_request that handles unknown session IDs
    gracefully by creating new sessions instead of returning 400 errors.
    """
    from mcp.server.streamable_http import (
        MCP_SESSION_ID_HEADER,
        StreamableHTTPServerTransport,
    )

    request = Request(scope, receive)
    request_mcp_session_id = request.headers.get(MCP_SESSION_ID_HEADER)

    # Existing session case
    if (
        request_mcp_session_id is not None
        and request_mcp_session_id in self._server_instances
    ):
        transport = self._server_instances[request_mcp_session_id]
        logger.debug("Session already exists, handling request directly")
        await transport.handle_request(scope, receive, send)
        return

    # NEW: Handle unknown session ID gracefully - create a new session
    if request_mcp_session_id is not None:
        logger.warning(
            f"Unknown session ID received: {request_mcp_session_id[:8]}... "
            "(possibly from before server restart). Creating new session."
        )
        # Fall through to new session creation below

    # New session case (or recovery from unknown session)
    logger.debug("Creating new transport")
    async with self._session_creation_lock:
        new_session_id = uuid4().hex
        http_transport = StreamableHTTPServerTransport(
            mcp_session_id=new_session_id,
            is_json_response_enabled=self.json_response,
            event_store=self.event_store,  # May be None (no resumability)
            security_settings=self.security_settings,
            retry_interval=self.retry_interval,
        )

        assert http_transport.mcp_session_id is not None
        self._server_instances[http_transport.mcp_session_id] = http_transport
        logger.info(f"Created new transport with session ID: {new_session_id}")

        # Define the server runner
        async def run_server(
            *, task_status: TaskStatus[None] = anyio.TASK_STATUS_IGNORED
        ) -> None:
            async with http_transport.connect() as streams:
                read_stream, write_stream = streams
                task_status.started()
                try:
                    await self.app.run(
                        read_stream,
                        write_stream,
                        self.app.create_initialization_options(),
                        stateless=False,  # Stateful mode
                    )
                except Exception as e:
                    logger.error(
                        f"Session {http_transport.mcp_session_id} crashed: {e}",
                        exc_info=True,
                    )
                finally:
                    # Only remove from instances if not terminated
                    if (
                        http_transport.mcp_session_id
                        and http_transport.mcp_session_id in self._server_instances
                        and not http_transport.is_terminated
                    ):
                        logger.info(
                            "Cleaning up crashed session "
                            f"{http_transport.mcp_session_id} from "
                            "active instances."
                        )
                        del self._server_instances[http_transport.mcp_session_id]

        # Assert task group is not None for type checking
        assert self._task_group is not None
        # Start the server task
        await self._task_group.start(run_server)

        # Handle the HTTP request and return the response
        await http_transport.handle_request(scope, receive, send)


def apply_session_recovery_patch():
    """
    Apply the session recovery patch to the MCP StreamableHTTPSessionManager.

    This should be called early in the server startup, before any requests
    are handled.
    """
    global _original_handle_stateful_request

    try:
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

        # Store original for potential restoration
        _original_handle_stateful_request = (
            StreamableHTTPSessionManager._handle_stateful_request
        )

        # Apply the patched version
        StreamableHTTPSessionManager._handle_stateful_request = (
            _patched_handle_stateful_request
        )

        logger.info(
            "Applied session recovery patch to StreamableHTTPSessionManager - "
            "unknown session IDs will now create new sessions instead of 400 errors"
        )
        return True

    except ImportError as e:
        logger.warning(f"Could not apply session recovery patch: {e}")
        return False
    except Exception as e:
        logger.error(f"Error applying session recovery patch: {e}", exc_info=True)
        return False


def remove_session_recovery_patch():
    """
    Remove the session recovery patch and restore original behavior.
    """
    global _original_handle_stateful_request

    if _original_handle_stateful_request is None:
        logger.warning("Session recovery patch was not applied, nothing to remove")
        return False

    try:
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

        StreamableHTTPSessionManager._handle_stateful_request = (
            _original_handle_stateful_request
        )
        _original_handle_stateful_request = None

        logger.info("Removed session recovery patch from StreamableHTTPSessionManager")
        return True

    except Exception as e:
        logger.error(f"Error removing session recovery patch: {e}", exc_info=True)
        return False
