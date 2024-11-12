import yaml


class HParams():

    def __init__(self, params_dict):
        self.params_dict = params_dict
        for key, val in params_dict.items():
            self.__setattr__(key, val)

        self.__setattr__ = self.__custom_setattr__

    def __getitem__(self, key):
        return self.params_dict[key]
    
    def __setitem__(self, key, val):
        self.params_dict[key] = val
        self.__setattr__(key, val)


    def __custom_setattr__(self, key, val):
      self.params[key] = val
      super().__setattr__(key, val)

def simple_load_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        params_dict = yaml.safe_load(f)
    return HParams(params_dict)

if __name__ == '__main__':


    params = simple_load_yaml('/pscratch/sd/j/jpathak/hrrr_experiments/baseline_v3/new-stats/hyperparams.yaml')
    
    print(params.batch_size)

    params.new_param = 10

    print(params.new_param)

