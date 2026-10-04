# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QUALEX-MS (QUick ALmost EXact maximum weight clique solver based on Motzkin-Straus formulation) is a C/C++ solver for the maximum weight clique/independent set problem. It uses a trust region technique with a generalized Motzkin-Straus formulation and has O(n³) complexity.

The solver is a command line front end (`main.cc`) to a library in `lib/` (`libqms.a`), which other programs (the SAT01 solver) link for MIN and the spectral stages.

## Build Commands

```bash
# Build the solver on the CPU backend with OpenBLAS (LAPACK and CBLAS), the default
make

# The same with Intel MKL (from MKLROOT, default ~/opt/intel-mkl-2026.1)
make BLAS=mkl

# Build the solver on the GPU backend (CUDA toolkit: cuSOLVER, cuBLAS; plus a CBLAS)
make GPU=1

# Clean build artifacts
make clean
```

Objects, `libqms.a` and the binary go to `build/<cpu|gpu>-<openblas|mkl>`; `./qualex-ms` is a copy of the binary of the configuration last asked for. The code is C++20 (`-std=gnu++20`) and depends on nothing but the standard library and the BLAS/LAPACK/CUDA backends; do not add third-party libraries.

## Usage

```bash
# Find maximum clique (default)
./qualex-ms <dimacs_binary_file>

# Find maximum independent set
./qualex-ms -c <dimacs_binary_file>

# With vertex weights
./qualex-ms <dimacs_binary_file> -w<weights_file>

# Vertex numbering from 1 in output
./qualex-ms +1 <dimacs_binary_file>
```

Output is written to a `.sol` file with the same base name as the input.

## Architecture

The solver pipeline consists of three stages:

1. **Preprocessing** (`lib/preproc_clique.h/.cc`): Reduces the graph by preselecting vertices that must be in any maximum clique and removing vertices that cannot contribute.

2. **Greedy Heuristic** (`lib/greedy_clique.h/.cc`): MIN heuristic starting from each vertex to find an initial lower bound.

3. **QUALEX-MS Core** (`lib/qualex.h/.cc`): Trust region optimization on the Motzkin-Straus quadratic formulation with refinement to extract cliques.

The library has three layers:

- **Combinatorial core**, no BLAS: `bool_vector`, `graph`, `greedy_clique`, `refiner` (VO and MIN), `preproc_clique`.
- **Linear algebra** (`lib/linalg.h`): one interface, two backends chosen at build time: `linalg_cpu.c` (LAPACK DSYEVR and CBLAS, from OpenBLAS or MKL; the default) and `linalg_gpu.c` (cuSOLVER/cuBLAS, eigenvectors kept on the GPU).
- **Spectral stages**: `wrapper.h/.cc` (the clique wrapper and its experimental modifications: perturbation, Theorem 8 anchoring, the lambda_max step), `qualex.h/.cc` (the trust region core), `dr.h/.cc` (generic Douglas-Rachford between a sphere and a set given by its projection, used by `try_dr_points()`; it calls CBLAS on either backend).

**Key Data Structures**:
- `Graph` (`lib/graph.h`): Undirected graph with weighted vertices, adjacency stored as `bool_vector` bit matrices
- `MaxCliqueInfo` (`lib/graph.h`): Tracks solver state including current best clique, bounds, and weight scaling factors; `verbose` turns the progress line off for library use
- `bool_vector` (`lib/bool_vector.h`): Bit string in 64-bit words; `for(int j : v.ones())` visits the true entries, skipping zero words and finding each entry with `std::countr_zero`

**Experimental switches** are `QMS_*` environment variables (listed in `README.md`); `bench/variants` names the combinations the bench runs.

## Code Style

Prefix increments (`++i`, `--i`) wherever the value is not used. Keep the existing comment density: the long comments record what each experiment measured.

## Input Format

DIMACS binary format for graphs. Weights file is optional plain text with one weight per line (must be ≥ 1.0).
