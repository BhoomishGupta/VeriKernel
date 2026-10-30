#!/usr/bin/env bash
# ==============================================================================
# VeriKernel: Zero-Configuration Cloudflare HTTPS Tunnel Helper
# Exposes the local port 8000 backend via an encrypted public HTTPS endpoint
# to eliminate Mixed Content blocking when connected to GitHub Pages or Netlify.
# ==============================================================================
set -e

PORT="${PORT:-8000}"
JUDGE_TOKEN="${JUDGE_SECRET_TOKEN:-verikernel-judge-2026}"

echo "=================================================================="
echo "   VeriKernel: Cloudflare HTTPS Ingress Tunnel Launcher"
echo "=================================================================="
echo "Target Local Service: http://127.0.0.1:${PORT}"
echo "Judge VIP Passkey:    ${JUDGE_TOKEN}"
echo "=================================================================="

# Check if local backend is responding
if ! curl -s -f -m 2 "http://127.0.0.1:${PORT}/api/health" &> /dev/null; then
    echo "⚠️ Warning: Local backend does not appear to be running on http://127.0.0.1:${PORT}."
    echo "  Start it with 'make app' or 'make docker-run' in another terminal."
    echo ""
fi

# Determine whether to use local binary or Docker container
if command -v cloudflared &> /dev/null; then
    echo "✓ Found local cloudflared binary."
    echo "Starting ad-hoc HTTPS tunnel to http://127.0.0.1:${PORT}..."
    echo ""
    echo "💡 Once running, copy the https://...trycloudflare.com URL below."
    echo "   Submission format for judges:"
    echo "   https://<YOUR-GITHUB-USERNAME>.github.io/VeriKernel/?backend=https://<TUNNEL-URL>&judge_token=${JUDGE_TOKEN}"
    echo "=================================================================="
    exec cloudflared tunnel --url "http://127.0.0.1:${PORT}"
elif command -v docker &> /dev/null; then
    echo "✓ Local cloudflared not found. Launching via official Docker image (cloudflare/cloudflared:latest)..."
    echo ""
    echo "💡 Once running, copy the https://...trycloudflare.com URL below."
    echo "   Submission format for judges:"
    echo "   https://<YOUR-GITHUB-USERNAME>.github.io/VeriKernel/?backend=https://<TUNNEL-URL>&judge_token=${JUDGE_TOKEN}"
    echo "=================================================================="
    exec docker run --rm -it --network host cloudflare/cloudflared:latest tunnel --url "http://127.0.0.1:${PORT}"
else
    echo "✗ Neither cloudflared CLI nor Docker is available on this host."
    echo ""
    echo "To install cloudflared on Linux (Debian/Ubuntu):"
    echo "  curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb"
    echo "  sudo dpkg -i cloudflared.deb"
    echo ""
    echo "Or start Docker service to use the containerized tunnel launcher."
    echo "=================================================================="
    exit 1
fi
