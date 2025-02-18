import copy
from typing import Any


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, mapping: dict[Any, Any] = None, **kwargs):
        mapping = mapping or {}
        mapping.update(kwargs)
        converted = {k: recursive_to_dotdict(v) for k, v in mapping.items()}
        super().__init__(converted)

    def update(self, other: dict[Any, Any]) -> None:
        """
        Recursively update the DotDict with another dictionary.

        For matching keys that are dictionaries, merge them recursively.
        Otherwise, the value from 'other' overrides.
        """
        merged = recursive_merge(self, other)
        self.clear()
        self.update(merged)


def recursive_to_dotdict(x: Any) -> Any:
    """
    Recursively convert nested dictionaries (and lists) to DotDict instances.

    If x is already a DotDict, it is returned as-is.
    """
    if isinstance(x, DotDict):
        return x
    if isinstance(x, dict):
        return DotDict({k: recursive_to_dotdict(v) for k, v in x.items()})
    elif isinstance(x, list):
        return [recursive_to_dotdict(item) for item in x]
    return x


def recursive_merge(default: dict[Any, Any], user: dict[Any, Any]) -> dict[Any, Any]:
    """
    Recursively merge a user dictionary into a default dictionary.

    For matching keys whose values are dictionaries, the merge is performed recursively.
    Otherwise, the value from 'user' replaces the default.
    """
    result = copy.deepcopy(default)
    for k, v in user.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = recursive_merge(result[k], v)
        else:
            result[k] = v
    return result


if __name__ == "__main__":
    cfg = {
        "figure": {
            "figsize": (10, 5),
            "dpi": 150,
            "style": "seaborn-v0_8-bright",
            "font_family": "sans-serif",
            "font_size": 12,
            "title": "Temporal distribution of SPOT pairs",
        },
        "scatter": {
            "alpha": 0.8,
            "size": 40,
            "edge_color": "white",
            "edge_width": 0.4,
            "cmap": "viridis",
            "use_jitter": True,
            "jitter": 0.2,
        },
        "periods": {
            "start": 9,  # September
            "end": 6,  # June
            "lines": {
                "style": "--",
                "alpha": 0.7,
                "width": 0.8,
                "color": "#ff9999",
            },
            "bands": {
                "accumulation": {"alpha": 0.1, "color": "#b0d8ff"},
                "ablation": {"alpha": 0.1, "color": "#f7a6a6"},
            },
        },
    }

    user_cfg = {
        "figure": {
            "title": "Dummy title",
        },
        "periods": {
            "bands": {
                "dummy": {"alpha": 1, "color": "#b0d8ff"},
                "ablation": {"alpha": 1, "color": "#f7a6a6"},
            },
        },
    }

    dotdict = DotDict(cfg)
    dotdict.update(user_cfg)
    print(dotdict)
