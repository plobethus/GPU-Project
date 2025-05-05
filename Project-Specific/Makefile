# Makefile

# Compilers
CC        := g++
CFLAGS    := -O2 -std=c++11
NVCC      := nvcc
NVCCFLAGS := -O2 -std=c++11

# Default problem size (override with e.g. `make run N=2000`)
N         ?= 1024

# Executable names
GEN       := matrix_generator
CPU       := cpu_solver
CUDA      := cuda_solver

# Time command
TIME_CMD  := /usr/bin/time -p

.PHONY: all run clean

all: $(GEN) $(CPU) $(CUDA)

# 1) Build the random matrix generator
$(GEN): matrix_generator.c
	$(CC) $(CFLAGS) $< -o $@

# 2) Build the CPU LU-solver
$(CPU): CPU_solver.cpp
	$(CC) $(CFLAGS) $< -o $@

# 3) Build the CUDA LU-solver
$(CUDA): lu_cuda_extended.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# 4) Generate, then benchmark both solvers
run: all
	@echo ">>> Generating a $(N)x$(N) system"
	@./$(GEN) $(N)
	@echo
	@echo ">>> Running CPU solver on Matrix.txt / b.txt"
	@$(TIME_CMD) ./$(CPU)
	@echo
	@echo ">>> Running CUDA solver on Matrix.txt / b.txt"
	@$(TIME_CMD) ./$(CUDA)

# 5) Cleanup everything
clean:
	@rm -f $(GEN) $(CPU) $(CUDA) Matrix.txt b.txt
	@echo "Cleaned."