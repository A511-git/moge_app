"""
Standalone MoGe-3 Panorama Package.
Completely self-contained with zero external dependencies on the root moge repo.
"""

import sys
from .custom_deps import utils3d_moge

try:
    from .custom_deps import flex_gemm
except ImportError:
    flex_gemm = None

# Ensure standalone_moge.utils3d_moge and standalone_moge.flex_gemm aliases exist in sys.modules
sys.modules.setdefault("standalone_moge.utils3d_moge", utils3d_moge)
sys.modules.setdefault("utils3d_moge", utils3d_moge)
if flex_gemm is not None:
    sys.modules.setdefault("standalone_moge.flex_gemm", flex_gemm)
    sys.modules.setdefault("flex_gemm", flex_gemm)

from .model import MoGeModel, import_model_class_by_version

__all__ = ["MoGeModel", "import_model_class_by_version", "utils3d_moge", "flex_gemm"]

