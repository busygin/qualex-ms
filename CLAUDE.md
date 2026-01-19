# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QUALEX-MS (QUick ALmost EXact maximum weight clique solver based on Motzkin-Straus formulation) is a C/C++ solver for the maximum weight clique/independent set problem. It uses a trust region technique with a generalized Motzkin-Straus formulation and has O(n³) complexity.

## Build Commands

```bash
# Build the solver (requires BLAS and LAPACK)
make

# Clean build artifacts
make clean
```

**Dependencies**: BLAS and LAPACK libraries. The Makefile is configured for CUDA-accelerated BLAS (`cublas`) with static LAPACK. Adjust `BLASLIB` and `LAPACKLIB` in the Makefile for your environment.

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

1. **Preprocessing** (`preproc_clique.h/.cc`): Reduces the graph by preselecting vertices that must be in any maximum clique and removing vertices that cannot contribute.

2. **Greedy Heuristic** (`greedy_clique.h/.cc`): MIN heuristic starting from each vertex to find an initial lower bound.

3. **QUALEX-MS Core** (`qualex.h/.cc`): Trust region optimization on the Motzkin-Straus quadratic formulation with refinement to extract cliques.

**Key Data Structures**:
- `Graph` (`graph.h`): Undirected graph with weighted vertices, adjacency stored as `bool_vector` bit matrices
- `MaxCliqueInfo` (`graph.h`): Tracks solver state including current best clique, bounds, and weight scaling factors
- `bool_vector` (`bool_vector.h`): Bit-packed boolean vector with efficient iteration via `bit_iterator`

**Numerical Core**:
- `eigen.c`: LAPACK interface for eigendecomposition (DSYEVR)
- `mdv.c`: Sparse matrix-diagonal-vector products
- `refiner.h/.cc`: VO (Vertex Order) and MIN procedures to extract maximal cliques from continuous solutions

## Input Format

DIMACS binary format for graphs. Weights file is optional plain text with one weight per line (must be ≥ 1.0).
