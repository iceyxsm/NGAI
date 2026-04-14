# NGAI — Next Gen Artificial Intelligence

A research project exploring novel methods for training and running AI models. The goal: 100B+ parameter models that can run on 4-8GB RAM.

## Vision

Current AI models rely on algorithms and architectures that are decades old. NGAI rethinks the fundamentals — new training methods, new inference approaches, and radically better memory efficiency.

## Getting Started

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (fast Python package manager)

### Setup
```bash
uv sync
```

### Run tests
```bash
uv run pytest
```

## Project Structure
```
src/ngai/
├── core/       # Core engine and novel algorithms
├── models/     # Model architectures
├── data/       # Data loading and processing
└── utils/      # Shared utilities
experiments/    # Experiment scripts
tests/          # Tests
```

## License

TBD
