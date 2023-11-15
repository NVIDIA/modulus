<!-- markdownlint-disable -->
## Model recipes / Training configurations

This folder contains the configurations for training the ML models. This folder is structured as follows:


```
makani
├── ...
├── config                  # configurations
│   ├── afnonet.yaml        # baseline configurations for original FourCastNet paper
│   ├── icml_models.yaml    # contains various dataloaders
│   ├── sfnonet_devel.yaml  # SFNO configurations used for active development
│   ├── sfnonet_legacy.yaml # legacy SFNO configurations
│   ├── sfnonet.yaml        # stable SFNO baselines
│   └── Readme.md           # this file
...

```

For the most recent configurations, check `sfnonet_devel.yaml`. The current baselines are `sfno_linear_73chq_sc3_layers8_edim384_wstgl2` for single GPU and `sfno_linear_73chq_sc2_layers8_edim960_wstgl2` for 4 GPUs.