#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vessel Metrics.

Run it to open the graphical interface:

    python vm_UI.py

or import it as a library (the GUI only starts when run as a script):

    import vm_UI as vm
    seg = vm.segment_image(image)

The code lives in the vessel_metrics package next to this file:

    logs.py          progress log printed to the terminal
    segmentation.py  preprocessing, vessel enhancement filters, segmentation
    skeleton.py      skeletonization and skeleton clean-up
    diameter.py      full-width-half-max vessel diameter
    measurements.py  density, branchpoints, length, tortuosity
    fileio.py        reading images and .czi files, saving settings
    settings.py      segmentation settings
    analysis.py      whole analysis runs and ROI analysis
    gui.py           graphical user interface
    _fast.py         shared array helpers

Created by S.D. McGarry under the direction of S. Childs, University of Calgary.
"""
import sys

import vessel_metrics
from vessel_metrics.gui import main

# make every function available as vm_UI.<name>, as in the single-file version
vessel_metrics.export_all(globals())
sys.modules[__name__].__class__ = vessel_metrics.ForwardingModule

if __name__ == "__main__":
    sys.exit(main())
