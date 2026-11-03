# Vessel Metrics

This repository is a fork of the original [Vessel Metrics project](https://github.com/mcgarrysd/vessel_metrics), created by S.D. McGarry under the direction of S. Childs at the University of Calgary. The original project and its scientific work are credited to the original owner. This fork preserves the original analysis functionality while providing updated documentation, packaging, and executable-build instructions.

## Fork changes

This fork includes:

* A documented graphical workflow for selecting input images, segmentation settings, and output parameters.
* Recommended starting settings for clear and accurate vessel segmentation.
* Windows and macOS packaging scripts for building standalone executables.
* A PyInstaller specification that includes CZI-reading support.

## Recommended image settings

The defaults are optimized for confocal microscopy images with vessels approximately 10 pixels in diameter. Use these values as a starting point and validate the resulting segmentation overlay against representative images:

| Setting | Starting value |
| --- | --- |
| Filter | `meijering` |
| Threshold | `40` |
| Sigma 1 | `1 8 1` |
| Sigma 2 | `10 20 5` |
| Hole size | `50` |
| Ditzle size | `500` |
| Preprocess | `yes` |
| Multi scale | `no` |

Adjust the threshold first: lower it when faint vessels are missing and raise it when background noise is included. Use `jerman` for dim or low-contrast images. Enable multi-scale processing when the image contains both microvasculature and substantially larger vessels. The sigma ranges should match the vessel widths in pixels; overly large sigma values can overestimate vessel boundaries.

For quantitative work, optimize settings on a representative image, inspect the segmentation and skeleton overlays, then save the settings and apply the same settings to the rest of the dataset. Prefer lossless input such as CZI, TIFF, or PNG; JPEG compression can introduce artifacts.

## Description
Vessel metrics is a software package written in python designed to automate the analysis of vascular architecture. Default settings are optimized for confocal microscopy imaging with vessels on the order of 10 pixels in diameter, but will support other image types with manual user adjustment.

This project is actively under development. A user guide for the user interface will be available shortly.

![alt text](https://github.com/mcgarrysd/vessel_metrics/blob/main/sample_ims/methods_figure_v1.png "Vessel diameters")


## Features
### Processing raw microscopy data
Data input accepted as either native .czi files or in image format

### Image preprocessing
A variety of commonly used preprocessing techniques are included, including tophat background suppression, smoothing, contrast stretching and histogram equalization, and 4 vessel enhancement filters.

A python implementation of the 2d Jerman vessel enhancement filter is included with this software package.

![alt text](https://github.com/mcgarrysd/vessel_metrics/blob/main/sample_ims/raw_im.png "Raw image")

### Vessel segmentation
Vessel segmentation included is optimized for confocal imaging. The default pipeline performs a top hat filter, background smoothing, a contrast stretch, and a Meijering filter before binarizing the image and performing morphological operations to adjust for common artifacts. Segmentation accuracy can be verified using a jaccard index or vessel specific accuracy metrics (connectivity, area, and length).

![alt text](https://github.com/mcgarrysd/vessel_metrics/blob/main/sample_ims/segmentation.png "Vessel segmentation")

### Vessel architecture
Default skeletonization using skimage inbuilt function is built upon to handle common artefacts in vascular imaging. Small terminal segments are removed and parent segments with multiple child branches are merged to produce a more accurate representation of vessel structure.

![alt text](https://github.com/mcgarrysd/vessel_metrics/blob/main/sample_ims/edge_labels.png "Vessel segments")
![alt text](https://github.com/mcgarrysd/vessel_metrics/blob/main/sample_ims/fix_skeleton.png "Erroneous segments removed from skeleton")

### Fully automatic vessel diameter calculation
Using the vessel skeleton, vessel metrics identifies individual vessel segments and branch points, determines the local slope of a segment, places a cross line of appropriate length, and measures the full width half max diameter across the vessel at each crossline. User defined crossline spacing (default every 10 pixels). 

![alt text](https://github.com/mcgarrysd/vessel_metrics/blob/main/sample_ims/segment_diameter.png "Single segment diameter")
![alt text](https://github.com/mcgarrysd/vessel_metrics/blob/main/sample_ims/diam_removal.png "Erroneous idameter measurements removed")
### Vessel parameter calculation
Vessel metrics calculates branchpoint density, segment length, vessel density, and network length.

![alt text](https://github.com/mcgarrysd/vessel_metrics/blob/main/sample_ims/vessel_density.png "Vessel density map")

### Second channel analysis
Optimized for pericyte detection in confocal microscopy. This feature allows the integration of vessel parameters with information derived from a second image channel. 

![alt text](https://github.com/mcgarrysd/vessel_metrics/blob/main/sample_ims/pericyte_ovl.png "Pericyte overlay")

### Visualization tools
Functions are included to aid in the visualization of the parameter maps, overlaid over the raw image for debugging or figure generation.

## Requirements
Vessel metrics is written for python 3 and tested on python 3.13. `vm_UI.py` is the entry point; the code lives in the `vessel_metrics` package next to it (segmentation, skeleton, diameter, measurements, file reading, settings, analysis runs and the interface each have their own module). Install the packages it uses with

```
pip install -r requirements.txt
```

* numpy, scipy, scikit-image, opencv-python, matplotlib
* tkinter (graphical interface, part of python; on macOS use the python.org installer or `brew install python-tk`)
* aicsimageio[czi] or czifile (only for reading .czi files)

## Running the program
```
python vm_UI.py
```
opens the graphical interface (see User Interface below). It can also be imported as a library, `import vm_UI as vm` (or `import vessel_metrics as vm`), in which case no window is opened.

## Standalone executable
Vessel metrics can be packaged as a single executable that runs without a python installation. The progress log is printed in a terminal window that opens next to the interface.

* Windows: run `build_windows.bat`; the executable is written to `dist\VesselMetrics.exe`
* macOS: run `./build_mac.sh` on a Mac, the program is written to `dist/VesselMetrics` (double-click it, it opens in Terminal). The executable runs on the processor type of the Mac it was built on (Apple silicon or Intel).

Both scripts create a `.venv` virtual environment, install `requirements.txt` and PyInstaller, and build from `vm_UI.spec`. PyInstaller cannot build a macOS program on Windows or vice versa; the GitHub Actions workflow in `.github/workflows/build.yml` builds the Windows, Apple silicon and Intel Mac versions on GitHub's machines (Actions tab, "Run workflow").

The first start of the executable takes a few seconds while it unpacks itself.

### Build the Windows executable

From the repository directory on Windows, run:

```bat
build_windows.bat
```

The script creates or reuses `.venv`, installs the requirements and PyInstaller, and creates the single-file executable at:

```text
dist\VesselMetrics.exe
```

Run it from PowerShell or Command Prompt with:

```bat
dist\VesselMetrics.exe
```

The Windows executable is a console application, so a terminal window is expected to open alongside the graphical interface. The first launch may take several seconds while PyInstaller unpacks the application.

## Usage
### Processing raw microscopy data
``` Python
import vm_UI as vm
import os
import numpy as np

data_path = 'path/to/data/'
file = 'file_name.czi'

# returns the volume scaled 0-255 and the physical pixel size [z, y, x] (None values if the czifile reader is used)
volume, dims = vm.preprocess_czi(data_path,file, channel = 0)
slice_range = len(volume)
slice_thickness = np.round(slice_range/2).astype(np.uint8)
reslice = vm.reslice_image(volume,slice_thickness)
vessel_raw = reslice[0]

```
### Image preprocessing
``` Python
vessel_preproc = vm.preprocess_seg(vessel_raw)

# Alternatively, this can be done manually
image = normalize_contrast(image)

image = subtract_background(image, radius = radius, light_bg = bright_background)
image = cv2.medianBlur(image.astype(np.uint8),median_size)
image = contrast_stretch(image, upper_lim = upper_lim, lower_lim = lower_lim)
```
### Vessel segmentation
``` Python
# filter options meijering, sato, frangi, jerman
# hole size (default 50) controls the size of small holes to be filled in post processing
# ditzle size (default 500) removes small objects post segmetnation
# sigma1 (default range(1,8,1)) controls sigma values input to enhancement filter
# sigma2 is only used if multi_scale = True, this setting applies 2 vessel enhancement filters and sums the images.
# thresh (default 40) is the threshold value to binarize the final enhanced image
# Your sigma numbers should cover the range of vessel diameters you expect in your image. It's ideal to set the max value smaller than the biggest vessel in your image, as the filter tends to overestimate the vessel boundaries otherwise.
vessel_seg = vm.segment_image(image, im_filter = 'meijering', sigma1 = range(1,8,1), sigma2 = range(10,20,5), hole_size = 50, ditzle_size = 500, thresh = 40, preprocess = True, multi_scale = True)
label = cv2.imread('/path/to/label.png',0)
length, area, conn, Q = vm.cal(label.astype(np.uint8), vessel_seg.astype(np.uint8))
jaccard = vm.jaccard(label,vessel_seg)

```
### Vessel architecture
``` Python
# The skeletonize_vm function uses the skimage skeletonize function and then post processes the skeleton to fix common errors in skeletonizing vascular trees.
skel, edges, bp = vm.skeletonize_vm(vessel_seg)
_, edge_labels = cv2.connectedComponents(edges)

_, branchpoints = vm.find_branchpoints(skel)
coords, endpoints = vm.find_endpoints(edges)
```
### Fully automatic vessel diameter calculation
``` Python
# viz is an image of equal size to vessel_preproc containing binary vessel crosslines for visualization purposes
viz, diameters = vm.whole_anatomy_diameter(vessel_preproc, vessel_seg, edge_labels)

# to visualize a single segment use visualize vessel diameter
# diam_list is the diameter measured at each crossline
# mean_diam is the mean of diam_list
# segment_viz is a binary image showing crosslines for that segment
diam_list, mean_diam, segment_viz = vm.visualize_vessel_diameter(edge_labels, segment_number, vessel_seg,vessel_preproc)
    

```
### Vessel parameter calculation
``` Python
# All of the below is accomplished using the parameter_analysis function in conjuction with the user interface.

# network length is the summation of all segment lengths
net_length = vm.network_length(edges)

# vessel density is the number of vessel pixels vs total pixels
# 16,16 denotes how many x and y chunks to break the image into (in this case 16 and 16)
density_image, vessel_density, overlay = vm.vessel_density(vessel_preproc, vessel_seg, 16, 16)

# bp_density is a list of (measurement number, branchpoints per 50x50 pixel window)
bp_density, overlay = vm.branchpoint_density(vessel_seg)

# length is a list containing the segment length for every segment in edge_labels
_, length = vm.vessel_length(edge_labels)

# tort holds one tortuosity value (straight-line distance / segment length) per segment in labels_used
tort, labels_used = vm.tortuosity(edge_labels)

```
### Second channel analysis
``` Python
volume, _ = vm.preprocess_czi(data_path,this_file, channel = 1)
slice_range = len(volume)
slice_thickness = np.round(slice_range/2).astype(np.uint8)
reslice = vm.reslice_image(volume,slice_thickness)
pericyte_raw = reslice[0]

peri_seg = np.zeros_like(pericyte_raw)
high_vals = np.zeros_like(pericyte_raw)
high_vals[pericyte_raw>75] = 1
peri_seg[(high_vals>0) & (vessel_seg>0)]=1

kernel = np.ones((3,3),np.uint8)
peri_seg = cv2.morphologyEx(peri_seg, cv2.MORPH_OPEN, kernel)

num_labels, labels = cv2.connectedComponents(peri_seg.astype(np.uint8))

unique_labels = np.array(np.nonzero(np.unique(labels))).flatten()

reduced_label = np.zeros_like(peri_seg)
for u in unique_labels:
    numel = len(np.argwhere(labels == u))
    if numel>15 and numel<500:
        reduced_label[labels == u] = 1

```
### Visualization tools
``` Python
# alpha controls opacity
# more complex overlays can be made by for example adding the segmentation and skeleton together
vm.overlay_segmentation(vessel_preproc, vessel_seg, alpha = 0.5)
vm.show_im(image)
```

### User Interface
The user interface is opened by running `python vm_UI.py`. It is a single window: choose a single image or a directory, choose the output folder, fill in the segmentation settings (or load saved ones), tick the parameters to measure, and press Run analysis. Progress is shown in the log box at the bottom. All text and fields use fixed colours so the window stays readable in macOS dark mode.

![alt text](https://github.com/mcgarrysd/vessel_metrics/blob/main/sample_ims/UI_fig.png "User Interface Explanation")

The recommended workflow is to operate on a single image from your dataset, optimize the parameters to your satisfaction, and then save those settings to batch process the remainder of your dataset. Settings are saved as settings.data in the selected output directory.

#### Segmentation settings
* Filter

Determines which filter will be used. Default meijering. Accepted responses: 'meijering', 'sato', 'frangi', 'jerman'. Jerman filter recommended for low contrast data. 

* Threshold

Threshold used for final segmentation. Values are 0-255, vessel metrics applies a contrast stretch to data prior to thresholding to ensure consistency across data acquired with varying microscope settings. Default value 40.
* Sigma 1 and Sigma 2

Conceptually the sigma values describe the thickness of vessels the filters enhance. This value is always expressed in pixels. Accepted input is a comma separated series Start value, End Value, Step Size. i.e. 1,8,1. The stop value should be around 80% of the maximal vessel size expected in your dataset, choosing a larger value tends to cause overestimation of vessel boundaries. 

The second sigma parameter is only used for multi scale processing, if your image contains major vessels and microvasculature both the multi scale processing produces better results. Rather than inputting a sigma value of 1,20,5 you will see better results with 1,8,1, and 10,20,5. 
* Hole size and Ditzle size

Accepted inputs are integers. Small holes (area less than the specified value) are closed in large binary objects post segmentation. Ditzle size removes binary objects less than the specified size. The 'ditzles' are usually partially formed vessels or vessels with poor signal along the periphery of the image.
* Preprocess

Whether to preprocess the data. If you've applied your own preprocessing prior to loading vessel metrics select no. Preprocessing picks up at vessel enhancement. 
* Multi scale

Whether to do a multi scale enhancement. Useful if there is a large discrepancy in vessel size within your image. 

#### CZI Processing
Vessel metrics operates on czi files and image files (tif, png, etc). If you are using a microscopy file that isn't a czi you can create a z projection of your desired thickness using Fiji or your microscopes proprietary software and load that into vessel metrics. 

You can select which channel is used for the analysis. These are integers beginning with 0 (your first channel is channel 0). You may also select how many slices are used to create a z projection. If you select a value less than the total slices in your czi multiple slices will be created and analyzed. 
#### UI Output
A directory is created for each sample analyzed. Within each directory. An unprocessed z proejction is saved as img.png. The vessel segmentation is saved as label.png. 

Vessel_labels.png overlays the vessel skeleton and segment numbers on the vessel label. If for example you wanted to know the diameter of a particularly large vessel you would open the vessel labels image and find the associated segment number and then find that entry in the vessel_diameters.txt file. 

The vessel density function breaks the image into 256 equally sized squares and outputs the number of labelled pixels over the total number of pixels in that tile. The procedure for finding vessel density within a region is identical, find the section in question on the vessel_density.png and the related entry in vessel_density.txt.
