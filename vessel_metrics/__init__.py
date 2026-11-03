"""Vessel Metrics: image analysis of vascular architecture.

Segmentation, skeletonization, vessel density, branchpoint density, network
length, tortuosity, segment length and full-width-half-max vessel diameter.

Every function of the sub-modules is available directly from the package (and
from vm_UI), e.g. vessel_metrics.segment_image(image).

Created by S.D. McGarry under the direction of S. Childs, University of Calgary.
"""
import sys
import types

from . import (
    _fast,
    analysis,
    diameter,
    fileio,
    gui,
    logs,
    measurements,
    segmentation,
    settings,
    skeleton,
)

MODULES = (
    logs, _fast, segmentation, skeleton, diameter, measurements, fileio,
    settings, analysis, gui,
)


def export_all(namespace):
    """Copy every function, class and constant of the sub-modules into namespace."""
    for module in MODULES:
        for name, value in vars(module).items():
            if not name.startswith("__"):
                namespace[name] = value


class ForwardingModule(types.ModuleType):
    """Module whose UPPER_CASE settings, when assigned (vm.VERBOSE = False),
    are also changed in the sub-modules that actually use them."""

    def __setattr__(self, name, value):
        if name.isupper():
            for module in MODULES:
                if hasattr(module, name):
                    setattr(module, name, value)
        super().__setattr__(name, value)


export_all(globals())
sys.modules[__name__].__class__ = ForwardingModule
