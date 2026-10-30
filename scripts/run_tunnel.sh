#!/usr/bin/env bash
# ==============================================================================
# VeriKernel: Persistent HTTPS Ingress Tunnel Launcher (ngrok & cloudflared)
# Exposes the local port 8000 backend via an encrypted public HTTPS endpoint
# to eliminate Mixed Content blocking when connected to GitHub Pages or Netlify.
# ==============================================================================
set -e

PORT="${PORT:-8000}"
JUDGE_TOKEN="${JUDGE_SECRET_TOKEN:-verikernel-judge-2026}"
NGROK_DOMAIN="${NGROK_DOMAIN:-thank-regulate-swell.ngrok-free.dev}"

echo "=================================================================="
echo "   VeriKernel: HTTPS Ingress Tunnel Launcher (ngrok)"
echo "=================================================================="
echo "Target Local Service: http://127.0.0.1:${PORT}"
echo "Registered Domain:    https://${NGROK_DOMAIN}"
echo "Judge VIP Passkey:    ${JUDGE_TOKEN}"
echo "=================================================================="

# Check if local backend is responding
if ! curl -s -f -m 2 "http://127.0.0.1:${PORT}/api/health" &> /dev/null; then
    echo "⚠️ Warning: Local backend does not appear to be running on http://127.0.0.1:${PORT}."
    echo "  Start it with 'make app' or 'make docker-run' in another terminal."
    echo ""
fi

# 1. Prefer native ngrok CLI if installed or configured
if command -v ngrok &> /dev/null; then
    echo "✓ Found ngrok CLI at $(command -v ngrok)."
    
    # Check if authtoken is present in environment or ngrok config
    if [ -n "$NGROK_AUTHTOKEN" ]; then
        ngrok config add-authtoken "$NGROK_AUTHTOKEN" &> /dev/null || true
    fi

    echo "Launching persistent tunnel on https://${NGROK_DOMAIN}..."
    echo ""
    echo "💡 Devpost & Judge Submission URL:"
    echo "   https://bhoomishgupta.github.io/VeriKernel/?backend=https://${NGROK_DOMAIN}&judge_token=${JUDGE_TOKEN}"
    echo "=================================================================="
    exec ngrok http "${PORT}" --url="${NGROK_DOMAIN}"

# 2. Docker container fallback for ngrok
elif command -v docker &> /dev/null && [ -n "$NGROK_AUTHTOKEN" ]; then
    echo "✓ Launching ngrok via Docker container..."
    echo ""
    echo "💡 Devpost & Judge Submission URL:"
    echo "   https://bhoomishgupta.github.io/VeriKernel/?backend=https://${NGROK_DOMAIN}&judge_token=${JUDGE_TOKEN}"
    echo "=================================================================="
    exec docker run --rm -it --network host -e NGROK_AUTHTOKEN="${NGROK_AUTHTOKEN}" ngrok/ngrok:latest http "${PORT}" --url="${NGROK_DOMAIN}"

# 3. Fallback to Cloudflare quick tunnel if requested
elif command -v cloudflared &> /dev/null; then
    echo "✓ Launching via cloudflared..."
    exec cloudflared tunnel --url "http://127.0.0.1:${PORT}"

else
    echo "✗ ngrok authtoken not yet configured."
    echo ""
    echo "Please configure your ngrok authtoken:"
    echo "  1. Copy your authtoken from: https://dashboard.ngrok.com/get-started/your-authtoken"
    echo "  2. Run: ngrok config add-authtoken <YOUR_TOKEN>"
    echo "  3. Re-run: ./scripts/run_tunnel.sh"
    echo "=================================================================="
    exit 1
fi
