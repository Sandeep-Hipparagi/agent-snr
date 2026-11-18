# Contributing to Agent SNR

Thank you for your interest in contributing to `agent-snr`!

`agent-snr` is an open-source, lightweight, framework-agnostic Agent Signal Firewall in Python.

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/agent-snr.git
   cd agent-snr
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e ".[dev,embeddings]"
   ```

## Development Workflow

Before submitting a pull request, ensure the test suite passes and formatting checks succeed:

- **Linting & Code Style**:
  ```bash
  ruff check .
  ruff format --check .
  ```

- **Type Checking**:
  ```bash
  mypy src/
  ```

- **Running Tests**:
  ```bash
  pytest
  ```

## Pull Request Guidelines

- Keep changes focused and atomic.
- Include unit tests for any new features or bug fixes.
- Ensure all CI checks (Ruff, mypy, pytest) pass.
- Write clear, descriptive commit messages.
