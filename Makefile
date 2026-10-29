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
KERNEL_DEF      := -DKERNEL_HEADER="$(abspath $(CAVITY_BASELINE))"

# Backward-compatibility aliases
TARGET          := $(CAVITY_TARGET)
REF_DATA        := $(CAVITY_REF_DATA)

# --- Workload 2: 2D Acoustic Wave Equation (Hyperbolic Stencil) ---
WAVE_TARGET     := $(BUILD_DIR)/harness_wave
WAVE_SRC        := harness_wave.cpp
WAVE_BASELINE   := wave_baseline.hpp

.PHONY: all clean clean-all test test-local bench ref gallery wave test-wave bench-wave help

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

# Concrete rule to build the golden reference binary
$(CAVITY_REF_DATA): $(CAVITY_TARGET)
	@echo "==> Generating ground-truth reference binary ($(CAVITY_REF_DATA))..."
	./$(CAVITY_TARGET) --generate-ref

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

# Verify the complete Break-It Gallery (5 Adversarial Defect Kernels)
gallery:
	@echo ""
	@echo "  Running VeriKernel Break-It Gallery Verification Suite"
	@echo ""
	@echo "\n[1/5] Testing Bug 1: Fast-Math Cheat Pragma (Expect Stage 1 fail)"
	-$(PYTHON) referee.py --kernel gallery/bug_1_fast_mathcheat.hpp --local
	@echo "\n[2/5] Testing Bug 2: Naive OpenMP Race Condition (Expect Stage 3 fail)"
	-$(PYTHON) referee.py --kernel gallery/bug_2_naive_omp_race.hpp --local
	@echo "\n[3/5] Testing Bug 3: Boundary Off-By-One (Expect Stage 4 fail)"
	-$(PYTHON) referee.py --kernel gallery/bug_3_boundary_off_by_one.hpp --local
	@echo "\n[4/5] Testing Bug 4: Float32 Precision Stalling (Expect Stage 4 fail)"
	-$(PYTHON) referee.py --kernel gallery/bug_4_fp32_downgrade.hpp --local
	@echo "\n[5/5] Testing Bug 5: Parity Strided Checkerboard Bug (Expect Stage 4 fail)"
	-$(PYTHON) referee.py --kernel gallery/bug_5_wrong_redblack_index.hpp --local
	@echo "\n==> Break-It Gallery suite finished."

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

# Help menu
help:
	@echo "VeriKernel Build and Test Commands:"
	@echo "  make              - Compiles both Cavity and Wave test harnesses into $(BUILD_DIR)/"
	@echo "  make ref          - Generates $(CAVITY_REF_DATA) on 1 thread"
	@echo "  make test         - Runs referee.py on baseline_kernel.hpp (All 5 Stages)"
	@echo "  make test-local   - Runs referee.py forcing native g++ compilation"
	@echo "  make bench        - Runs Stage 5 pinned memory bandwidth benchmark for Cavity (513x513)"
	@echo "  make gallery      - Executes the 5 Break-It Gallery adversarial tests"
	@echo "  make wave         - Compiles the 2D Wave test harness into $(BUILD_DIR)/"
	@echo "  make test-wave    - Runs 2D Wave multi-grid convergence & determinism check"
	@echo "  make bench-wave   - Runs Stage 5 pinned memory bandwidth benchmark for Wave (512x512)"
	@echo "  make clean        - Deletes compiled binaries and temporary test files"
	@echo "  make clean-all    - Deletes binaries AND $(CAVITY_REF_DATA)"
