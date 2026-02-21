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

Session limits prevent unbounded memory growth from polling clients that
repeatedly create new sessions without reusing existing ones.
"""

import logging
import time
from http import HTTPStatus
from uuid import uuid4

from anyio.abc import TaskStatus
import anyio
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import Receive, Scope, Send

logger = logging.getLogger(__name__)

_original_handle_stateful_request = None

# Maximum number of concurrent sessions before we start rejecting new ones.
# This prevents unbounded memory growth from polling clients.
# Reduced from 100: single-user deployment rarely needs >20 concurrent sessions.
# OOM crashes (SIGKILL/exit-9) were caused by sessions accumulating to 100
# before cleanup triggered, each holding transport/stream state in memory.
MAX_SESSIONS = 30

# Track session creation times for cleanup of idle sessions
_session_last_active: dict[str, float] = {}

# Sessions older than this (seconds) with no activity are eligible for cleanup.
# Reduced from 300s: health-check polling creates ~7 sessions per 30s cycle,
# so 300s allowed ~70 sessions to accumulate before expiry. At 120s, idle
# sessions are cleaned before they pile up to OOM-inducing levels.
SESSION_IDLE_TIMEOUT = 120  # 2 minutes

# Proactive cleanup interval — run cleanup periodically, not just when MAX_SESSIONS hit.
# This prevents the sawtooth pattern where sessions grow to MAX then bulk-purge.
_PROACTIVE_CLEANUP_INTERVAL = 60  # seconds
_last_proactive_cleanup: float = 0.0


def _cleanup_idle_sessions(manager) -> int:
    """Remove sessions that have been idle longer than SESSION_IDLE_TIMEOUT.

    Args:
        manager: The StreamableHTTPSessionManager instance

    Returns:
        Number of sessions cleaned up
    """
    now = time.monotonic()
    to_remove = []
    for sid, last_active in list(_session_last_active.items()):
        if now - last_active > SESSION_IDLE_TIMEOUT:
            to_remove.append(sid)

    cleaned = 0
    for sid in to_remove:
        _session_last_active.pop(sid, None)
        transport = manager._server_instances.pop(sid, None)
        if transport is not None:
            cleaned += 1
            # Terminate the transport to free resources
            try:
                transport.is_terminated = True
            except Exception:
                pass

    if cleaned > 0:
        logger.info(
            f"Cleaned up {cleaned} idle sessions (>{SESSION_IDLE_TIMEOUT}s inactive). "
            f"Active sessions: {len(manager._server_instances)}"
        )
    return cleaned


async def _patched_handle_stateful_request(
    self,
    scope: Scope,
    receive: Receive,
    send: Send,
) -> None:
    """
    Patched version of _handle_stateful_request that handles unknown session IDs
    gracefully by creating new sessions instead of returning 400 errors.

    Includes session limits and idle cleanup to prevent unbounded memory growth.
    """
    from mcp.server.streamable_http import (
        MCP_SESSION_ID_HEADER,
        StreamableHTTPServerTransport,
    )

    global _last_proactive_cleanup

    request = Request(scope, receive)
    request_mcp_session_id = request.headers.get(MCP_SESSION_ID_HEADER)

    # Proactive cleanup: run periodically on every request to prevent session buildup.
    # This avoids the sawtooth pattern where sessions accumulate to MAX_SESSIONS
    # before any cleanup happens, which was the root cause of OOM crashes.
    now = time.monotonic()
    if now - _last_proactive_cleanup > _PROACTIVE_CLEANUP_INTERVAL:
        _last_proactive_cleanup = now
        _cleanup_idle_sessions(self)

    # Existing session case
    if (
        request_mcp_session_id is not None
        and request_mcp_session_id in self._server_instances
    ):
        transport = self._server_instances[request_mcp_session_id]
        # Update last-active time
        _session_last_active[request_mcp_session_id] = time.monotonic()
        logger.debug("Session already exists, handling request directly")
        await transport.handle_request(scope, receive, send)
        return

    # Handle unknown session ID gracefully - create a new session
    # but also map the old session ID to the new transport for continuity
    stale_session_id = None
    if request_mcp_session_id is not None:
        logger.warning(
            f"Unknown session ID received: {request_mcp_session_id[:8]}... "
            "(possibly from before server restart). Creating new session."
        )
        stale_session_id = request_mcp_session_id
        # Fall through to new session creation below

    # New session case (or recovery from unknown session)
    logger.debug("Creating new transport")
    async with self._session_creation_lock:
        # Enforce session limit: clean up idle sessions first, then check limit
        current_count = len(self._server_instances)
        if current_count >= MAX_SESSIONS:
            cleaned = _cleanup_idle_sessions(self)
            current_count = len(self._server_instances)

            if current_count >= MAX_SESSIONS:
                logger.error(
                    f"Session limit reached ({MAX_SESSIONS} active sessions, "
                    f"cleaned {cleaned} idle). Rejecting new session request."
                )
                response = Response(
                    content="Too many active sessions. Please retry later.",
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                )
                await response(scope, receive, send)
                return

        # For stale session recovery, reuse the client's session ID
        # This allows the client to continue using their existing session ID
        # without needing to handle session ID changes
        if stale_session_id is not None:
            new_session_id = stale_session_id
            logger.info(
                f"Reusing client's stale session ID for continuity: {new_session_id[:8]}..."
            )
        else:
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
        _session_last_active[http_transport.mcp_session_id] = time.monotonic()
        logger.info(
            f"Created new transport with session ID: {new_session_id[:8]}... "
            f"(active sessions: {len(self._server_instances)})"
        )

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
                    # Clean up from instances and tracking
                    sid = http_transport.mcp_session_id
                    if sid:
                        _session_last_active.pop(sid, None)
                        if (
                            sid in self._server_instances
                            and not http_transport.is_terminated
                        ):
                            logger.info(
                                f"Cleaning up crashed session {sid} from active instances."
                            )
                            del self._server_instances[sid]

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
