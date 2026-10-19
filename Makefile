CXX ?= g++
CXXFLAGS ?= -O3 -march=native -fopenmp -std=c++17 -Wall -Wextra -pedantic
LDFLAGS ?= -fopenmp

BIN_DIR := build
HARNESS := $(BIN_DIR)/harness_cavity

.PHONY: bench test clean

$(HARNESS): harness_cavity.cpp baseline_kernel.hpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) harness_cavity.cpp -o $(HARNESS) $(LDFLAGS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

bench: $(HARNESS)
	$(HARNESS)

test: $(HARNESS)
	python3 referee.py --harness harness_cavity.cpp --binary $(HARNESS)

clean:
	rm -rf $(BIN_DIR)
