"""Download Rossmann Store Sales competition files through Kaggle tooling."""
from __future__ import annotations
import argparse
import logging
import shutil
import subprocess
import zipfile
from pathlib import Path

from utils import configure_logging

LOGGER = logging.getLogger(__name__)
COMPETITION = "rossmann-store-sales"
EXPECTED = {"train.csv", "test.csv", "store.csv"}


def _extract_archives(directory: Path) -> None:
    for archive in directory.glob("*.zip"):
        LOGGER.info("Extracting %s", archive.name)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(directory)


def _validate(directory: Path) -> None:
    missing = [name for name in EXPECTED if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError(f"Download completed but files are missing: {missing}")


def download_with_kaggle_cli(output_dir: Path) -> None:
    """Download via the Kaggle CLI; competition rules must be accepted first."""
    executable = shutil.which("kaggle")
    if not executable:
        raise RuntimeError("Kaggle CLI is not installed or not available on PATH")
    command = [executable, "competitions", "download", "-c", COMPETITION, "-p", str(output_dir), "--force"]
    LOGGER.info("Running authenticated Kaggle competition download")
    subprocess.run(command, check=True)
    _extract_archives(output_dir)
    _validate(output_dir)


def download_with_kagglehub(output_dir: Path) -> None:
    """Download via KaggleHub when the environment is authenticated."""
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("kagglehub is not installed") from exc
    LOGGER.info("Downloading competition through KaggleHub")
    source = Path(kagglehub.competition_download(COMPETITION))
    output_dir.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        for item in source.iterdir():
            target = output_dir / item.name
            if item.is_file():
                shutil.copy2(item, target)
    else:
        shutil.copy2(source, output_dir / source.name)
    _extract_archives(output_dir)
    _validate(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--source", choices=["auto", "kagglehub", "cli"], default="auto")
    args = parser.parse_args()
    configure_logging()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    methods = ([download_with_kagglehub, download_with_kaggle_cli] if args.source == "auto" else
               [download_with_kagglehub] if args.source == "kagglehub" else [download_with_kaggle_cli])
    for method in methods:
        try:
            method(output)
            LOGGER.info("Rossmann data is ready in %s", output)
            return
        except Exception as exc:
            errors.append(f"{method.__name__}: {exc}")
            LOGGER.warning("%s failed: %s", method.__name__, exc)
    raise RuntimeError("Unable to download competition data. Accept the Kaggle rules and authenticate. " + " | ".join(errors))


if __name__ == "__main__":
    main()
