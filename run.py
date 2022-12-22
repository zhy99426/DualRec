from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI
from src.model import DualRecModule as Module
from src.datamodule import DualRecDataModule as DataModule


def cli_main():
    pl.seed_everything(42)

    cli = LightningCLI(
        Module, DataModule, subclass_mode_data=True, save_config_overwrite=True
    )


if __name__ == "__main__":
    cli_main()
