import os

import hydra
from omegaconf import OmegaConf

from modulus.launch.logging import LaunchLogger, initialize_mlflow

from diagnostic import data, distribute, loss, models, precip, train


@hydra.main(
    version_base=None, config_path="config", config_name="diagnostic_precip.yaml"
)
def main(cfg):
    train_diagnostic(**OmegaConf.to_container(cfg))


def train_diagnostic(**cfg):
    """Top-level training function: setup everything and train model."""

    # setup model
    model = models.setup_model(**cfg["model"])
    (model, dist_manager) = distribute.distribute_model(model)

    # setup datapipes
    (train_specs, valid_specs) = data.data_source_specs(
        cfg["sources"]["state_params"], cfg["sources"]["diag_params"]
    )
    (train_datapipe, valid_datapipe) = data.setup_datapipes(
        train_specs,
        valid_specs,
        **cfg["datapipe"],
        dist_manager=dist_manager,
    )

    # setup MLFlow logging
    mlflow_cfg = cfg.get("logging", {}).get("mlflow", {})
    if mlflow_cfg.pop("use_mlflow", False):
        initialize_mlflow(**mlflow_cfg)
        LaunchLogger.initialize(use_mlflow=True)

    # setup loss
    loss_func = loss.GeometricL2Loss(
        lat_indices_used=train_datapipe.crop_window[0]
    )  # TODO: this should be configurable
    loss_func.to(device=dist_manager.device)

    # conversion from datapipe format to (input, target) tuples
    batch_conv = data.batch_converter(
        *train_specs, train_datapipe, diag_norm=precip.PrecipNorm()
    )

    # setup training loop
    trainer = train.Trainer(
        model,
        dist_manager=dist_manager,
        loss=loss_func,
        train_datapipe=train_datapipe,
        valid_datapipe=valid_datapipe,
        input_output_from_batch_data=batch_conv,
        **cfg["training"],
    )

    # train model
    trainer.fit()


if __name__ == "__main__":
    main()
