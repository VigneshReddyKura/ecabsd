"""
ECABSD — Main entry point.

Usage:
    python main.py train
    python main.py predict --pdb 1AY7.pdb --chain-a A
    python main.py --help
"""

from cli import app

if __name__ == "__main__":
    app()
