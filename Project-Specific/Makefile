# Compilers
CC        := g++
CFLAGS    := -O2 -std=c++11
NVCC      := nvcc
NVCCFLAGS := -O2 -std=c++11

# Default problem size (override with e.g. `make run N=2000`)
N         ?= 1024

# Executable names
GEN           := matrix_generator
CPU           := cpu_solver
CUDA_NAIVE    := cuda_solver
CUDA_OPT      := cuda_solver_opt

# Time command
time_cmd    := /usr/bin/time -p

.PHONY: all run clean

all: $(GEN) $(CPU) $(CUDA_NAIVE) $(CUDA_OPT)

# 1) Build the random matrix generator
$(GEN): matrix_generator.c
	$(CC) $(CFLAGS) $< -o $@

# 2) Build the CPU LU-solver
$(CPU): CPU_solver.cpp
	$(CC) $(CFLAGS) $< -o $@

# 3) Build the naive CUDA LU-solver
$(CUDA_NAIVE): lu_cuda_naieve.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# 4) Build the optimized CUDA LU-solver
$(CUDA_OPT): lu_cuda_opt.cu
	$(NVCC) $(NVCCFLAGS) -lcublas $< -o $@

# 5) Generate, then benchmark both solvers
run: all
	@echo ">>> Generating a $(N)x$(N) system"
	@./$(GEN) $(N)
	@echo
	@echo ">>> Running CPU solver on Matrix.txt / b.txt"
	@$(time_cmd) ./$(CPU)
	@echo
	@echo ">>> Running naive CUDA solver on Matrix.txt / b.txt"
	@$(time_cmd) ./$(CUDA_NAIVE)
	@echo
	@echo ">>> Running optimized CUDA solver on Matrix.txt / b.txt"
	@$(time_cmd) ./$(CUDA_OPT)

# 6) Cleanup everything
clean:
	@rm -f $(GEN) $(CPU) $(CUDA_NAIVE) $(CUDA_OPT) Matrix.txt b.txt
	@echo "Cleaned."
