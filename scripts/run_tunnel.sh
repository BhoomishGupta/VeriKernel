#!/usr/bin/env bash
# ==============================================================================
# VeriKernel: Zero-Configuration Cloudflare Tunnel Helper
# Exposes the local port 8000 backend via an encrypted public HTTPS endpoint
# to eliminate Mixed Content blocking when connected to GitHub Pages or Netlify.
# ==============================================================================
set -e

PORT="${PORT:-8000}"

echo "=================================================================="
echo "   VeriKernel: Cloudflare HTTPS Tunnel Launcher"
echo "=================================================================="

if ! command -v cloudflared &> /dev/null; then
    echo "cloudflared CLI is not installed on this host."
    echo ""
    echo "To install on Linux:"
    echo "  curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb"
    echo "  sudo dpkg -i cloudflared.deb"
    echo ""
    echo "Or run via Docker without installing locally:"
    echo "  docker run --rm -it --network host cloudflare/cloudflared:latest tunnel --url http://127.0.0.1:${PORT}"
    echo "=================================================================="
    exit 1
fi

echo "Starting ad-hoc HTTPS tunnel to http://localhost:${PORT}..."
echo "Copy the https://...trycloudflare.com URL below into the VeriKernel Lab Instrument modal."
echo "=================================================================="
cloudflared tunnel --url "http://127.0.0.1:${PORT}"
