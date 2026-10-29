#!/usr/bin/env bash
# ==============================================================================
# VeriKernel Quick Evaluator Verification Script
# Executes the 5-Stage Physics-Aware Referee locally in < 3 seconds.
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
KERNEL="${1:-${REPO_ROOT}/candidate_kernel.hpp}"

# Styling
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BOLD}${BLUE}====================================================================${NC}"
echo -e "${BOLD}${CYAN}   VeriKernel: 5-Stage Physics-Aware C++ Referee Quick Verifier    ${NC}"
echo -e "${BOLD}${BLUE}====================================================================${NC}"
echo -e "Evaluating Kernel: ${BOLD}${KERNEL}${NC}"

if [ ! -f "${KERNEL}" ]; then
    echo -e "${RED}[ERROR] Target kernel header not found: ${KERNEL}${NC}"
    exit 1
fi

# 1. Check SHA-256 Seal
echo -e "\n${YELLOW}[Stage 0] Cryptographic Seal Calculation...${NC}"
if command -v sha256sum &> /dev/null; then
    SHA=$(sha256sum "${KERNEL}" | awk '{print $1}')
elif command -v shasum &> /dev/null; then
    SHA=$(shasum -a 256 "${KERNEL}" | awk '{print $1}')
else
    SHA="Unavailable (no sha256 tool)"
fi
echo -e "  SHA-256: ${BOLD}${SHA}${NC}"

# 2. Run 5-Stage Referee
echo -e "\n${YELLOW}[Executing Referee Gates 1 to 5]...${NC}"
cd "${REPO_ROOT}"

if command -v python3 &> /dev/null && [ -f "${REPO_ROOT}/referee.py" ]; then
    python3 "${REPO_ROOT}/referee.py" --kernel "${KERNEL}" --local --json
    REF_EXIT=$?
else
    echo -e "Running direct C++ test harness build..."
    g++ -O3 -march=native -fopenmp -DKERNEL_HEADER="\"${KERNEL}\"" "${REPO_ROOT}/harness_cavity.cpp" -o /tmp/quick_verify_harness
    /tmp/quick_verify_harness
    REF_EXIT=$?
fi

echo -e "\n${BOLD}${BLUE}====================================================================${NC}"
if [ ${REF_EXIT} -eq 0 ]; then
    echo -e "${BOLD}${GREEN}✓ REFEREE VERDICT: ALL 5 STAGES MATHEMATICALLY VERIFIED SOUND${NC}"
    echo -e "${GREEN}  Certified zero data race, strict conservation, and pinned speedup.${NC}"
else
    echo -e "${BOLD}${RED}✗ REFEREE VERDICT: REJECTED — SAFETY/PHYSICS INVARIANT BREACHED${NC}"
fi
echo -e "${BOLD}${BLUE}====================================================================${NC}"
exit ${REF_EXIT}
