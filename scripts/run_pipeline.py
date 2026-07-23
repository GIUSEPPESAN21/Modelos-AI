"""
End-to-end ML pipeline CLI for LA public health data.

Orchestrates: data loading → VAE imputation → DML causal inference → Streamlit dashboard.

Usage:
    python scripts/run_pipeline.py [--vae-train] [--dml-train] [--streamlit] [--verbose]

Options:
    --vae-train      Train VAE model (default: use pre-trained if available)
    --dml-train      Train DML model (default: use pre-trained if available)
    --streamlit      Launch Streamlit dashboard after pipeline
    --verbose        Enable verbose logging
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
DATA_DIR = ROOT / "data"
VAE_DIR = ROOT / "vae"
DML_DIR = ROOT / "dml"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_verbose_logging(verbose: bool) -> None:
    """Configure verbose logging if requested."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")


def ensure_directories() -> None:
    """Ensure required directories exist."""
    for d in [DATA_DIR, VAE_DIR / "results" / "trained_models", 
              VAE_DIR / "results" / "metrics", DML_DIR / "results" / "trained_models",
              DML_DIR / "results" / "metrics"]:
        d.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ensured: {d}")


def load_data() -> pd.DataFrame:
    """Load or generate main panel dataset."""
    parquet_path = DATA_DIR / "la_public_health.parquet"
    
    if parquet_path.exists():
        logger.info(f"Loading data from {parquet_path}")
        return pd.read_parquet(parquet_path)
    
    logger.info("Generating demo data...")
    sys.path.insert(0, str(SCRIPTS_DIR))
    from generate_data import generate_dataset
    generate_dataset(DATA_DIR)
    return pd.read_parquet(parquet_path)


def load_config(module: str) -> dict:
    """Load config for a module (vae or dml)."""
    config_path = ROOT / module / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    logger.warning(f"Config not found for {module}: {config_path}")
    return {}


def train_vae(data: pd.DataFrame, force: bool = False) -> dict:
    """Train VAE model if not already trained or if forced."""
    model_path = VAE_DIR / "results" / "trained_models" / "vae_best.pt"
    metrics_path = VAE_DIR / "results" / "metrics" / "eval_metrics.json"
    
    if model_path.exists() and metrics_path.exists() and not force:
        logger.info("✅ VAE model already trained, skipping training")
        with open(metrics_path, "r") as f:
            return json.load(f)
    
    logger.info("🚀 Starting VAE training...")
    try:
        # Import VAE training module
        sys.path.insert(0, str(VAE_DIR / "src"))
        from train import train_vae as train_vae_fn
        
        config = load_config("vae")
        from shared_utils import set_seed
        set_seed(config.get("seed", 42))
        
        # Train and get metrics
        metrics = train_vae_fn(data, config)
        
        # Save metrics
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"✅ VAE training complete. RMSE: {metrics.get('rmse', 'N/A')}")
        return metrics
    except Exception as e:
        logger.error(f"❌ VAE training failed: {e}")
        raise


def train_dml(data: pd.DataFrame, force: bool = False) -> dict:
    """Train DML model if not already trained or if forced."""
    model_path = DML_DIR / "results" / "trained_models" / "dml_result.joblib"
    metrics_path = DML_DIR / "results" / "metrics" / "eval_metrics.json"
    
    if model_path.exists() and metrics_path.exists() and not force:
        logger.info("✅ DML model already trained, skipping training")
        with open(metrics_path, "r") as f:
            return json.load(f)
    
    logger.info("🚀 Starting DML training...")
    try:
        # Import DML training module
        sys.path.insert(0, str(DML_DIR / "src"))
        from train import train_dml as train_dml_fn
        
        config = load_config("dml")
        from shared_utils import set_seed
        set_seed(config.get("seed", 42))
        
        # Train and get metrics
        metrics = train_dml_fn(data, config)
        
        # Save metrics
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"✅ DML training complete. ATE: {metrics.get('train_ate', 'N/A')}")
        return metrics
    except Exception as e:
        logger.error(f"❌ DML training failed: {e}")
        raise


def validate_results(vae_metrics: dict, dml_metrics: dict) -> None:
    """Validate that results are reasonable."""
    logger.info("📋 Validating results...")
    
    # VAE validation
    if "rmse" in vae_metrics:
        rmse = vae_metrics["rmse"]
        if rmse > 10:
            logger.warning(f"⚠️  VAE RMSE seems high: {rmse}")
        else:
            logger.info(f"✅ VAE RMSE within reasonable range: {rmse:.4f}")
    
    # DML validation
    if "train_ate" in dml_metrics:
        ate = dml_metrics["train_ate"]
        ci = dml_metrics.get("train_ci", [0, 0])
        
        # Check if CI includes 0
        if ci[0] < 0 < ci[1]:
            logger.warning(f"⚠️  DML ATE not significant at 95%: ATE={ate:.6f}, CI={ci}")
        else:
            logger.info(f"✅ DML ATE is significant: ATE={ate:.6f}, CI={ci}")
        
        # Placebo test
        if "placebo_ate" in dml_metrics:
            placebo_ci = dml_metrics.get("placebo_ci", [0, 0])
            if placebo_ci[0] < 0 < placebo_ci[1]:
                logger.info("✅ Placebo test passed (includes 0)")
            else:
                logger.warning(f"⚠️  Placebo test failed: {placebo_ci}")


def launch_streamlit() -> None:
    """Launch Streamlit dashboard."""
    logger.info("🎨 Launching Streamlit dashboard...")
    try:
        subprocess.run(
            ["streamlit", "run", str(ROOT / "app.py")],
            cwd=str(ROOT),
            check=True,
        )
    except Exception as e:
        logger.error(f"Failed to launch Streamlit: {e}")


def main() -> None:
    """Main pipeline orchestration."""
    parser = argparse.ArgumentParser(
        description="End-to-end ML pipeline for LA public health",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--vae-train", action="store_true", help="Train VAE (default: use pre-trained)")
    parser.add_argument("--dml-train", action="store_true", help="Train DML (default: use pre-trained)")
    parser.add_argument("--streamlit", action="store_true", help="Launch Streamlit after pipeline")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    setup_verbose_logging(args.verbose)
    ensure_directories()
    
    logger.info("=" * 70)
    logger.info("🏥 LA Public Health ML Pipeline")
    logger.info("=" * 70)
    
    try:
        # Step 1: Load data
        logger.info("\n[1/4] Loading data...")
        data = load_data()
        logger.info(f"✅ Loaded {len(data)} observations with {len(data.columns)} features")
        logger.info(f"   Missing data: {(data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100):.1f}%")
        
        # Step 2: VAE imputation
        logger.info("\n[2/4] VAE Imputation")
        vae_metrics = train_vae(data, force=args.vae_train)
        
        # Step 3: DML causal inference
        logger.info("\n[3/4] DML Causal Inference")
        dml_metrics = train_dml(data, force=args.dml_train)
        
        # Step 4: Validation
        logger.info("\n[4/4] Validation")
        validate_results(vae_metrics, dml_metrics)
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ Pipeline complete!")
        logger.info("=" * 70)
        
        # Results summary
        logger.info("\n📊 Results Summary:")
        logger.info(f"  VAE RMSE:        {vae_metrics.get('rmse', 'N/A')}")
        logger.info(f"  DML ATE:         {dml_metrics.get('train_ate', 'N/A')}")
        logger.info(f"  DML CI:          {dml_metrics.get('train_ci', 'N/A')}")
        logger.info(f"  Placebo ATE:     {dml_metrics.get('placebo_ate', 'N/A')}")
        
        logger.info("\n📁 Results saved to:")
        logger.info(f"  {VAE_DIR}/results/metrics/eval_metrics.json")
        logger.info(f"  {DML_DIR}/results/metrics/eval_metrics.json")
        
        logger.info("\n🎨 View results in Streamlit:")
        logger.info(f"  streamlit run {ROOT / 'app.py'}")
        
        if args.streamlit:
            logger.info("\nLaunching Streamlit dashboard...")
            launch_streamlit()
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
