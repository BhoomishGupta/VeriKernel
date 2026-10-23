# ==============================================================================

# VeriKernel: Physics-Aware AI Code Optimization Referee

# Build Automation & Verification Makefile

# ==============================================================================

CXX          ?= g++
CXXFLAGS     ?= -O3 -march=native -fopenmp -Wall -std=c++17
BUILD_DIR    := build
TARGET       := $(BUILD_DIR)/harness_cavity
SRC_HARNESS  := harness_cavity.cpp
BASELINE     := baseline_kernel.hpp
REF_DATA     := reference_cavity_129.bin
PYTHON       ?= python3

# Compiler definition for baseline kernel

KERNEL_DEF   := -DKERNEL_HEADER="$(abspath$(BASELINE))"

.PHONY: all clean clean-all test test-local bench ref gallery help

all: $(TARGET)

# Ensure build directory exists

$(BUILD_DIR):
@mkdir -p $(BUILD_DIR)

# Compile harness with baseline kernel directly via local compiler

$(TARGET):$(SRC_HARNESS) $(BASELINE) \vert{}$(BUILD_DIR)
@echo "> Compiling harness with $(BASELINE)..."
$(CXX) $(CXXFLAGS)$(KERNEL_DEF) $(SRC_HARNESS) -o$(TARGET)
@echo "> Build complete: $(TARGET)"

# Concrete rule to build the golden reference binary

$(REF_DATA):$(TARGET)
@echo "==> Generating ground-truth reference binary ($(REF_DATA))..."
./$(TARGET) --generate-ref

# Alias target for explicitly generating or updating reference data

ref: $(REF_DATA)

# Run full 5-stage referee test suite using containerized sandbox (or fallback)

test: $(REF_DATA)
@echo "==> Running VeriKernel 5-Stage Referee against $(BASELINE)..."
$(PYTHON) referee.py --kernel$(BASELINE)

# Force local compilation without Docker

test-local: $(REF_DATA)
@echo "==> Running VeriKernel Referee locally against $(BASELINE)..."
$(PYTHON) referee.py --kernel$(BASELINE) --local

# Stage 5 Pinned Benchmark runner (513x513 out-of-L2 grid)

bench: $(TARGET)
@echo "==> Executing Stage 5 Out-of-L2 Bandwidth Benchmark (513x513, 100 sweeps)..."
@OMP_PROC_BIND=close OMP_PLACES=cores ./$(TARGET) --stage5

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

# Clean build artifacts and temporary files

clean:
@echo "> Cleaning build artifacts..."
@rm -rf $(BUILD_DIR) test test_bad_lint.hpp test_race.hpp test_bad_physics.hpp **pycache** .pytest_cache
@echo "> Clean complete."

# Clean all artifacts including generated ground-truth binaries

clean-all: clean
@echo "> Removing generated reference binaries..."
@rm -f $(REF_DATA)
@echo "> Full clean complete."

# Help menu

help:
@echo "VeriKernel Build and Test Commands:"
@echo "  make            - Compiles the baseline test harness into $(BUILD_DIR)/"
@echo "  make ref        - Generates $(REF_DATA) on 1 thread"
@echo "  make test       - Runs referee.py on baseline_kernel.hpp (All 5 Stages)"
@echo "  make test-local - Runs referee.py forcing native g++ compilation"
@echo "  make bench      - Runs Stage 5 pinned memory bandwidth benchmark"
@echo "  make gallery    - Executes the 5 Break-It Gallery adversarial tests"
@echo "  make clean      - Deletes compiled binaries and temporary test files"
@echo "  make clean-all  - Deletes binaries AND $(REF_DATA)"