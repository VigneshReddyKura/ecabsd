"""
ECABSD CLI — Command-line interface for all pipelines.

Usage:
    python cli.py train
    python cli.py evaluate --checkpoint checkpoints/best_model.pt
    python cli.py predict --pdb 1AY7.pdb --chain-a A --chain-b B
    python cli.py batch-predict --input-dir data/raw/pdbs
    python cli.py export --format csv --results results/predictions.json
    python cli.py web
"""

import typer
from pathlib import Path
from typing import Optional

app = typer.Typer(
    name="ecabsd",
    help="ECABSD — Equivariant Cross-Attention for Binding Site Detection",
    add_completion=False,
)


@app.command()
def train(
    config: Path = typer.Option("config.yaml", help="Path to config YAML file"),
    resume: Optional[Path] = typer.Option(None, help="Resume from checkpoint"),
):
    """Train the ECABSD model."""
    from train import run_training
    run_training(config_path=str(config), resume_from=str(resume) if resume else None)


@app.command()
def evaluate(
    config: Path = typer.Option("config.yaml", help="Path to config YAML file"),
    checkpoint: Path = typer.Option(
        "checkpoints/best_model.pt", help="Model checkpoint path"
    ),
):
    """Evaluate model on the test set."""
    from evaluate import run_evaluation
    run_evaluation(config_path=str(config), checkpoint_path=str(checkpoint))


@app.command()
def predict(
    pdb: Path = typer.Option(..., help="Path to PDB file"),
    chain_a: str = typer.Option(..., help="Chain ID for target protein"),
    chain_b: Optional[str] = typer.Option(None, help="Chain ID for partner protein"),
    checkpoint: Path = typer.Option(
        "checkpoints/best_model.pt", help="Model checkpoint"
    ),
    threshold: float = typer.Option(0.5, help="Binding probability threshold"),
    output: Optional[Path] = typer.Option(None, help="Output file path"),
    config: Path = typer.Option("config.yaml", help="Path to config YAML file"),
):
    """Predict binding sites for a single PDB structure."""
    from predict import run_prediction
    run_prediction(
        pdb_path=str(pdb),
        chain_a=chain_a,
        chain_b=chain_b,
        checkpoint_path=str(checkpoint),
        threshold=threshold,
        output_path=str(output) if output else None,
        config_path=str(config),
    )


@app.command(name="batch-predict")
def batch_predict(
    input_dir: Path = typer.Option(..., help="Directory of PDB files"),
    checkpoint: Path = typer.Option(
        "checkpoints/best_model.pt", help="Model checkpoint"
    ),
    chain_a: str = typer.Option("A", help="Default chain A"),
    chain_b: Optional[str] = typer.Option(None, help="Default chain B"),
    threshold: float = typer.Option(0.5, help="Binding probability threshold"),
    output_dir: Path = typer.Option("results/batch", help="Output directory"),
    config: Path = typer.Option("config.yaml", help="Path to config YAML file"),
):
    """Batch predict binding sites for all PDB files in a directory."""
    from batch_predict import run_batch_prediction
    run_batch_prediction(
        input_dir=str(input_dir),
        checkpoint_path=str(checkpoint),
        chain_a=chain_a,
        chain_b=chain_b,
        threshold=threshold,
        output_dir=str(output_dir),
        config_path=str(config),
    )


@app.command()
def export(
    results: Path = typer.Option(..., help="Path to prediction results JSON"),
    format: str = typer.Option("csv", help="Export format: csv | json | pymol"),
    output: Optional[Path] = typer.Option(None, help="Output file path"),
):
    """Export prediction results to various formats."""
    if format == "csv":
        from exports.csv_export import export_csv
        export_csv(str(results), str(output) if output else None)
    elif format == "json":
        from exports.json_export import export_json
        export_json(str(results), str(output) if output else None)
    elif format == "pymol":
        from exports.pymol_export import export_pymol
        export_pymol(str(results), str(output) if output else None)
    else:
        typer.echo(f"Unknown format: {format}. Use csv, json, or pymol.")
        raise typer.Exit(code=1)
    typer.echo(f"Exported to {format} successfully.")


@app.command()
def web(
    config: Path = typer.Option("config.yaml", help="Path to config YAML file"),
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to listen on"),
):
    """Launch the ECABSD web interface."""
    import uvicorn
    from web.app import create_app

    create_app(config_path=str(config))
    uvicorn.run("web.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
