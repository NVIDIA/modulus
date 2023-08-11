import json
import omegaconf

class OmegaConfEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (omegaconf.ListConfig, omegaconf.DictConfig)):
            return omegaconf.OmegaConf.to_container(obj, resolve=True)
        return super().default(obj)

