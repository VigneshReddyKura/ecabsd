# Contributing to ECABSD

Thank you for your interest in contributing! This guide explains how to
set up your development environment, run tests, and submit pull requests.

---

## Development Setup

```bash
git clone https://github.com/amanigreeva/ECABSD.git
cd ecabsd

# Create environment
conda env create -f environment.yml
conda activate ecabsd
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_model_ml.py -v

# With coverage
pytest tests/ --cov=models --cov-report=term-missing
```

All PRs must pass the full test suite. Tests run automatically via GitHub
Actions on every push and pull request.

---

## Code Style

- Python 3.10+
- PEP 8 style
- Type hints encouraged for new public functions
- Docstrings for all public classes and functions

---

## Adding a New Feature

1. **Fork** the repository
2. **Create a branch**: `git checkout -b feature/your-feature-name`
3. **Write tests** for your feature in `tests/`
4. **Run tests** to confirm everything passes: `pytest tests/ -v`
5. **Commit** with a descriptive message
6. **Open a Pull Request** against `main`

---

## Reporting Issues

Please use [GitHub Issues](https://github.com/amanigreeva/ECABSD/issues) and
include:
- Python and PyTorch version
- Operating system
- Full error traceback
- Minimal reproducible example (PDB file if applicable)

---

## Scientific Contributions

For dataset contributions, new benchmarks, or architecture improvements:
- Open an issue first to discuss the approach
- Include evaluation metrics on the standard test set
- Reference relevant prior work

---

## License

By contributing, you agree that your contributions will be licensed under
the [MIT License](LICENSE).
