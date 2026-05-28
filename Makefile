# ECABSD Makefile — Common operations

.PHONY: help test train evaluate benchmark leakage kfold clean

help:
	@echo ""
	@echo "  ECABSD — Available Commands"
	@echo "  ─────────────────────────────────────────"
	@echo "  make test          Run full test suite"
	@echo "  make train         Train V3 model"
	@echo "  make evaluate      Evaluate best checkpoint"
	@echo "  make benchmark     Run baseline comparison"
	@echo "  make leakage       Check for data leakage"
	@echo "  make splits        Generate homology-aware splits (needs mmseqs2)"
	@echo "  make kfold         Run 5-fold cross-validation"
	@echo "  make clean         Remove cache files"
	@echo ""

test:
	pytest tests/ -v --tb=short

train:
	python train.py

evaluate:
	python main.py evaluate --checkpoint checkpoints/best_model_v3.pt

benchmark:
	python scripts/benchmark_crossPPI.py --checkpoint checkpoints/best_model_v3.pt

leakage:
	python check_leakage.py --mmseqs

splits:
	python scripts/generate_homology_splits.py \
		--splits data/splits.csv \
		--pdb-dir data/raw/pdbs \
		--output data/splits_homology.csv \
		--identity 0.30

kfold:
	python scripts/train_kfold.py \
		--config config.yaml \
		--splits data/splits_homology.csv \
		--folds 5 \
		--output results/kfold_results.json

predict:
	python predict.py --pdb 1AY7.pdb --chain-a A --chain-b B

web:
	python web/app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache
