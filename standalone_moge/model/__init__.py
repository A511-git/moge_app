from typing import Type
from .v3 import MoGeModel


def import_model_class_by_version(version: str = 'v3') -> Type[MoGeModel]:
    """
    Returns the MoGeModel class.
    MoGe-3 is the only supported version (supporting 'vitl' and 'vitg' backbones).
    """
    if version in ['v1', 'v2']:
        raise ValueError(
            f"MoGe '{version}' has been removed from this system. "
            f"Please use MoGe-3 variants ('vitl' or 'vitg')."
        )
    return MoGeModel


__all__ = ['MoGeModel', 'import_model_class_by_version']
