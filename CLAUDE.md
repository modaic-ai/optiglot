# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DS-RPC (DSPyGlot) is a library that makes DSPy optimizers multi-lingual by converting them into RPC calls. This allows DSPy ports in other languages (TypeScript, Rust, Go, Ruby, etc.) to leverage Python-based optimizers while running programs in their native implementation language.

## Architecture

### Core Concept

The system operates on a client-server RPC model:
- **Python side (this repo)**: Contains optimizer implementations that communicate via RPC
- **Server side (language-agnostic)**: Runs the actual DS program in its native language and returns results

### Data Flow

1. Optimizer (Python) sends `RPCRolloutRequest` to server
2. Server executes the program using native DS implementation
3. Server returns `RPCRolloutResponse` with predictions, scores, and optional traces
4. Optimizer uses results to compile/improve the program
5. Final optimized `RPCModule` is sent back to server via `/send` endpoint

### Key Components

**models.py** - Core data models (all Pydantic):
- `RPCModule`: A collection of `RPCPredict` predictors that form a program
- `RPCPredict`: Individual predictor with signature, demos, and LM config
- `RPCSignature`: Defines input/output fields and instructions
- `RPCRolloutRequest`/`RPCRolloutResponse`: Request/response for running programs
- `RPCTraceStep`: Captures predictor execution in a trace (used by bootstrap optimizers)
- `RPCUsage`: Token consumption tracking

**utils.py** - Utility functions:
- `evaluate()`: Sends rollout requests to server and returns results
- `create_minibatch()`: Creates random minibatches from trainset
- `eval_candidate_program()`: Evaluates candidate on full trainset or minibatch

**optimizers/** - Optimizer implementations:
- Base class: `RPCTeleprompter` with `compile()` and `compile_and_send()` methods
- `LabeledFewShot`: Simple few-shot with labeled examples
- `bootstrap.py`: Original DSPy bootstrap implementation (reference only)
- New optimizers should follow the RPC pattern: no direct LM calls, use `evaluate()` for rollouts

### Important Constraints

When implementing RPC-style optimizers:
- **No direct LM access**: Use `evaluate()` to trigger server-side execution
- **Stateless**: Optimizers should not maintain internal state that can't be serialized
- **Trace-based**: Bootstrap optimizers need `trace=True` to collect execution traces
- The server's `/rollout` endpoint handles all program execution
- The server's `/send` endpoint receives the final optimized module

## Environment Configuration

- `HOST_URL`: Server endpoint (default: `http://localhost:8000`, set via env var)

## Project Setup

This project uses `uv` for dependency management (Python >=3.11):
```bash
uv sync
```

## Reference: Original DSPy Bootstrap

The file `optimizers/bootstrap.py` contains the original DSPy `BootstrapFewShot` implementation. When creating RPC versions:
- Replace `teacher(**example.inputs())` with `evaluate()` calls
- Request traces via `trace=True` in `RPCRolloutRequest`
- Use `RPCTraceStep` instead of DSPy's internal trace format
- Metric evaluation happens server-side, returned in `score` field
