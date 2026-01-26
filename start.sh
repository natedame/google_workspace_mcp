#!/bin/bash
cd /Users/natedame/local-ai/google-workspace-mcp
source .env
export GOOGLE_OAUTH_CLIENT_ID GOOGLE_OAUTH_CLIENT_SECRET WORKSPACE_MCP_PORT OAUTHLIB_INSECURE_TRANSPORT
uv run main.py --transport streamable-http --single-user --tools gmail drive docs sheets slides calendar tasks
