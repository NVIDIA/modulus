from omegaconf import DictConfig
import hydra


@hydra.main(version_base="1.2", config_path="conf", config_name="config_training_template")
def main(cfg: DictConfig) -> None:
    return


if __name__ == "__main__":
    main()
