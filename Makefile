# ==============================================================================
# VeriKernel: Physics-Aware AI Code Optimization Referee
# Build Automation & Verification Makefile
# ==============================================================================

CXX          ?= g++
CXXFLAGS     ?= -O3 -march=native -fopenmp -Wall -std=c++17
BUILD_DIR    := build
PYTHON       ?= python3

# --- Workload 1: 2D Lid-Driven Cavity (Navier-Stokes) ---
CAVITY_TARGET   := $(BUILD_DIR)/harness_cavity
CAVITY_SRC      := harness_cavity.cpp
CAVITY_BASELINE := baseline_kernel.hpp
CAVITY_REF_DATA := reference_cavity_129.bin
KERNEL_DEF      := -DKERNEL_HEADER='"'$(abspath $(CAVITY_BASELINE))'"'

# Backward-compatibility aliases
TARGET          := $(CAVITY_TARGET)
REF_DATA        := $(CAVITY_REF_DATA)

# --- Workload 2: 2D Acoustic Wave Equation (Hyperbolic Stencil) ---
WAVE_TARGET     := $(BUILD_DIR)/harness_wave
WAVE_SRC        := harness_wave.cpp
WAVE_BASELINE   := wave_baseline.hpp

.PHONY: all clean clean-all test test-local bench check verify-all ref gallery wave test-wave bench-wave help static test-guards test-static preflight tunnel docker-build docker-run docker-stop

# Default target compiles both computational harnesses
all: $(CAVITY_TARGET) $(WAVE_TARGET)

# Ensure build directory exists
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# ==============================================================================
# Workload 1: Cavity Targets
# ==============================================================================

# Compile cavity harness with baseline kernel
$(CAVITY_TARGET): $(CAVITY_SRC) $(CAVITY_BASELINE) | $(BUILD_DIR)
	@echo "> Compiling cavity harness with $(CAVITY_BASELINE)..."
	$(CXX) $(CXXFLAGS) $(KERNEL_DEF) $(CAVITY_SRC) -o $(CAVITY_TARGET)
	@echo "> Build complete: $(CAVITY_TARGET)"

# Concrete rule to build the golden reference binary using a dedicated baseline build
$(CAVITY_REF_DATA): $(CAVITY_SRC) $(CAVITY_BASELINE) | $(BUILD_DIR)
	@echo "==> Compiling dedicated baseline to generate $(CAVITY_REF_DATA)..."
	$(CXX) $(CXXFLAGS) -DKERNEL_HEADER='"$(abspath $(CAVITY_BASELINE))"' $(CAVITY_SRC) -o $(BUILD_DIR)/harness_baseline_ref
	@echo "==> Generating ground-truth reference binary ($(CAVITY_REF_DATA))..."
	./$(BUILD_DIR)/harness_baseline_ref --generate-ref
	@rm -f $(BUILD_DIR)/harness_baseline_ref

# Alias target for explicitly generating or updating reference data
ref: $(CAVITY_REF_DATA)

# Run full 5-stage referee test suite using containerized sandbox (or fallback)
test: $(CAVITY_REF_DATA)
	@echo "==> Running VeriKernel 5-Stage Referee against $(CAVITY_BASELINE)..."
	$(PYTHON) referee.py --kernel $(CAVITY_BASELINE)

# Force local compilation without Docker
test-local: $(CAVITY_REF_DATA)
	@echo "==> Running VeriKernel Referee locally against $(CAVITY_BASELINE)..."
	$(PYTHON) referee.py --kernel $(CAVITY_BASELINE) --local

# Stage 5 Pinned Benchmark runner (513x513 out-of-L2 grid)
bench: $(CAVITY_TARGET)
	@echo "==> Executing Stage 5 Out-of-L2 Bandwidth Benchmark (513x513, 100 sweeps)..."
	@OMP_PROC_BIND=close OMP_PLACES=cores ./$(CAVITY_TARGET) --stage5

# Verify the complete Break-It Gallery (6 Adversarial Defect Kernels)
gallery:
	@echo ""
	@echo "  Running VeriKernel Break-It Gallery Verification Suite (6 Defects)"
	@echo ""
	@echo "\n[1/6] Testing Bug 1: Fast-Math Cheat Pragma (Expect Stage 1 fail)"
	-$(PYTHON) referee.py --kernel gallery/bug_1_fast_mathcheat.hpp --local
	@echo "\n[2/6] Testing Bug 2: Naive OpenMP Race Condition (Expect Stage 3 fail)"
	-$(PYTHON) referee.py --kernel gallery/bug_2_naive_omp_race.hpp --local
	@echo "\n[3/6] Testing Bug 3: Boundary Off-By-One (Expect Stage 4 fail)"
	-$(PYTHON) referee.py --kernel gallery/bug_3_boundary_off_by_one.hpp --local
	@echo "\n[4/6] Testing Bug 4: Float32 Precision Stalling (Expect Stage 4 fail)"
	-$(PYTHON) referee.py --kernel gallery/bug_4_fp32_downgrade.hpp --local
	@echo "\n[5/6] Testing Bug 5: Parity Strided Checkerboard Bug (Expect Stage 4 fail)"
	-$(PYTHON) referee.py --kernel gallery/bug_5_wrong_redblack_index.hpp --local
	@echo "\n[6/6] Testing Bug 6: Wave Naive Startup Order Collapse (Expect Stage 4 fail)"
	-$(PYTHON) referee.py --kernel gallery/bug_6_wave_naive_startup.hpp --workload wave --local
	@echo "\n==> Break-It Gallery suite finished (6/6 adversarial defect kernels evaluated)."

# One-command comprehensive reproducibility & verification suite
check: all $(CAVITY_REF_DATA)
	@echo "======================================================================"
	@echo "   VeriKernel Complete Automated Verification & Reproducibility Suite "
	@echo "======================================================================"
	@echo "\n==> [1/3] Verifying Baseline Cavity Stencil (5-Stage Referee)..."
	$(PYTHON) referee.py --kernel $(CAVITY_BASELINE) --local
	@echo "\n==> [2/3] Verifying 2D Wave Multi-Grid Convergence Order (p >= 1.90)..."
	./$(WAVE_TARGET)
	@echo "\n==> [3/3] Running Complete Break-It Gallery (6 Adversarial Tests)..."
	@$(MAKE) gallery
	@echo "\n======================================================================"
	@echo "✓ ALL VERIKERNEL VERIFICATION CHECKS COMPLETED SUCCESSFULLY."
	@echo "======================================================================"

verify-all: check

# ==============================================================================
# Workload 2: 2D Acoustic Wave Targets
# ==============================================================================

# Compile wave harness with baseline kernel
$(WAVE_TARGET): $(WAVE_SRC) $(WAVE_BASELINE) | $(BUILD_DIR)
	@echo "> Compiling wave harness with $(WAVE_BASELINE)..."
	$(CXX) $(CXXFLAGS) $(WAVE_SRC) -o $(WAVE_TARGET)
	@echo "> Build complete: $(WAVE_TARGET)"

wave: $(WAVE_TARGET)

# Multi-grid convergence test (verifies order p >= 1.90 and thread determinism)
test-wave: $(WAVE_TARGET)
	@echo "==> Running 2D Wave Multi-Grid Convergence Order & Determinism Check..."
	./$(WAVE_TARGET)

# Stage 5 Pinned Benchmark for Wave (512x512 grid, 200 sweeps)
bench-wave: $(WAVE_TARGET)
	@echo "==> Executing 2D Wave Stage 5 Pinned Bandwidth Benchmark (512x512, 200 sweeps)..."
	@OMP_PROC_BIND=close OMP_PLACES=cores ./$(WAVE_TARGET) --bench

# ==============================================================================
# Housekeeping & Help Targets
# ==============================================================================

# Clean build artifacts and temporary files
clean:
	@echo "> Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR) test test_bad_lint.hpp test_race.hpp test_bad_physics.hpp __pycache__ .pytest_cache
	@echo "> Clean complete."

# Clean all artifacts including generated ground-truth binaries
clean-all: clean
	@echo "> Removing generated reference binaries..."
	@rm -f $(CAVITY_REF_DATA)
	@echo "> Full clean complete."

# Run FastAPI Lab Instrument Dashboard
app:
	@echo "==> Launching VeriKernel Execution Server on http://0.0.0.0:8000..."
	$(PYTHON) server.py

# ==============================================================================
# Step 9: 3-Tier Deployment & Availability Protection Targets
# ==============================================================================

# Build standalone static distribution for GitHub Pages & Netlify
static:
	@echo "==> Materializing standalone static distribution in static/..."
	$(PYTHON) scripts/build_static.py

# Test strict API guards (IP rate limiter, daily cap, cloud spend limit)
test-guards:
	@echo "==> Running VeriKernel API Guards and Availability Protection test suite..."
	$(PYTHON) scripts/test_api_guards.py

# Test zero-backend static fallback layer
test-static: static
	@echo "==> Verifying Zero-Backend Static Fallback on local HTTP server..."
	$(PYTHON) scripts/verify_static_fallback.py

# Pre-flight 3-tier readiness audit (verifies Tier 1 static, Tier 2 guards, Tier 3 physics)
preflight: static
	@echo "==> Running Step 9 Pre-Flight 3-Tier Readiness Audit..."
	$(PYTHON) scripts/judge_preflight_check.py

# Launch HTTPS Cloudflare tunnel for port 8000 (eliminates mixed content)
tunnel:
	@bash scripts/run_tunnel.sh

# Build production Docker container
docker-build:
	@echo "==> Building production Docker image (verikernel:latest)..."
	docker build -t verikernel:latest .

# Run production Docker container with 8 dedicated pinned CPU cores and security limits
docker-run:
	@echo "==> Running VeriKernel container on port 8000 (8 pinned CPU cores, 2GB RAM, OpenMP Pinned)..."
	docker run -d --name verikernel-backend --rm \
		-p 8000:8000 \
		--cpuset-cpus="0-7" \
		--memory=2g \
		--pids-limit=128 \
		--security-opt no-new-privileges:true \
		--tmpfs /tmp:rw,size=256m,mode=1777 \
		--env-file .env \
		-v $(abspath cache):/app/cache \
		verikernel:latest

# Stop production Docker container
docker-stop:
	@echo "==> Stopping VeriKernel container..."
	-docker stop verikernel-backend

# Help menu
help:
	@echo "VeriKernel Build, Test & Deployment Commands:"
	@echo "  make app          - Launches the VeriKernel FastAPI execution server and lab dashboard"
	@echo "  make check        - One-command execution: runs baseline test, wave test, and all 6 gallery defect tests"
	@echo "  make              - Compiles both Cavity and Wave test harnesses into $(BUILD_DIR)/"
	@echo "  make ref          - Generates $(CAVITY_REF_DATA) on 1 thread using clean baseline"
	@echo "  make test         - Runs referee.py on baseline_kernel.hpp (All 5 Stages)"
	@echo "  make test-local   - Runs referee.py forcing native g++ compilation"
	@echo "  make bench        - Runs Stage 5 pinned memory bandwidth benchmark for Cavity (513x513)"
	@echo "  make gallery      - Executes all 6 Break-It Gallery adversarial defect tests"
	@echo "  make wave         - Compiles the 2D Wave test harness into $(BUILD_DIR)/"
	@echo "  make test-wave    - Runs 2D Wave multi-grid convergence & determinism check"
	@echo "  make bench-wave   - Runs Stage 5 pinned memory bandwidth benchmark for Wave (512x512)"
	@echo "  make static       - Builds standalone static distribution for GitHub Pages/Netlify"
	@echo "  make test-guards  - Tests IP rate limiting (5 calls/hr), daily cap (30), spend & judge VIP"
	@echo "  make test-static  - Validates zero-backend static fallback layer resolution"
	@echo "  make preflight    - End-to-end audit of 3-Tier deployment (Static + Guards + Physics)"
	@echo "  make tunnel       - Launches Cloudflare HTTPS tunnel for port 8000"
	@echo "  make docker-build - Builds production Docker image (verikernel:latest)"
	@echo "  make docker-run   - Runs containerized backend with 8 pinned cores on http://localhost:8000"
	@echo "  make docker-stop  - Stops running container"
	@echo "  make clean        - Deletes compiled binaries and temporary test files"
	@echo "  make clean-all    - Deletes binaries AND $(CAVITY_REF_DATA)"

