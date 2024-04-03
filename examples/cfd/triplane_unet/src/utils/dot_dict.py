import copy


class DotDict(dict):
    """dot.notation access to dictionary attributes."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"


def flatten_dict(d, parent_key="", sep="_", no_sep_keys=["base"]):
    items = []
    for k, v in d.items():
        # Do not expand parent key if it is "base"
        if parent_key in no_sep_keys:
            new_key = k
        else:
            new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
