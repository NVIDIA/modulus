# File for common tools in shard patching
from collections.abc import Iterable

class UndeterminedShardingError(Exception):
    pass

class MissingShardPatch(NotImplementedError):
    pass


def promote_to_iterable(input_obj, target_iterable):
    """
    Promotes an input to an iterable of the same type as a target iterable,
    unless the input is already an iterable (excluding strings).

    Args:
        input_obj: The object to promote.
        target_iterable: The target iterable whose type determines the result.

    Returns:
        An iterable of the same type as the target iterable.
    """
    
    # If input_obj is a string or not iterable, wrap it in the target's type.
    if isinstance(input_obj, str) or not isinstance(input_obj, Iterable):
        # Also extend it with copies to the same length:
        ret = type(target_iterable)([input_obj]) * len(target_iterable)
        return ret
    
    # If input_obj is already an iterable, return it as-is.
    assert len(input_obj) == len(target_iterable)
    
    return input_obj