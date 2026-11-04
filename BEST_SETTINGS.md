# Best Settings and Results Guide

This guide explains how to choose settings for clear, reproducible, and accurate vessel segmentation in Vessel Metrics.

## Before you start

Use the highest-quality image available:

* Prefer `.czi`, `.tif`, `.tiff`, or `.png`.
* Avoid JPEG images because compression can create artificial edges and noise.
* Use the original pixel resolution; do not resize images before analysis.
* Select a representative image containing the full range of vessel sizes and signal quality in your dataset.
* For CZI files, confirm the correct channel before changing segmentation settings.

The default pipeline is optimized for confocal images with vessels approximately 10 pixels in diameter. Settings are image-dependent, so the values below are starting points rather than universal values.

## Recommended starting preset

Enter these values in the graphical interface:

| Setting | Value |
| --- | --- |
| Filter | `meijering` |
| Threshold | `40` |
| Sigma 1 | `1 8 1` |
| Sigma 2 | `10 20 5` |
| Hole size | `50` |
| Ditzle size | `500` |
| Image type | your choice (required), see below |
| Multi scale | `no` |

This preset is appropriate when vessels have broadly similar widths and reasonable contrast.

## Choosing the image type and photo preprocessing

Choose the image type before anything else, then press **Check image...** to confirm it: the software labels the image and previews each step.

* **Normal photo** (e.g. a yolk-sac photo with dark red vessels): inverted first, then processed.
* **Inverted / fluorescence**: bright vessels on a dark background.
* **Pre-processed**: already background-corrected; the built-in preprocessing is skipped.

Suggested starting point for colour photos of a sample cropped to a dark or white oval:

```text
Colour channel: green
Local contrast boost (CLAHE): on
Remove dark specks, radius: 6
Mask the area outside the sample: on
```

On the provided sample photo these four options raised vessel coverage from 8.7% to about 11% of the image: CLAHE recovered fine branches, speck removal reconnected vessels broken by debris, and the outside mask removed false vessels along the cropped edge. Keep the speck radius below half the gap between neighbouring vessels, or nearby vessels can merge.

## Choosing a filter

* **Meijering**: best first choice for bright, tube-like vessels and the default pipeline.
* **Jerman**: try this for dim or low-contrast vessels.
* **Sato**: useful when the vessel structures are elongated and the signal is relatively consistent.
* **Frangi**: useful as an alternative when Meijering does not represent the vessel network well.

Compare the segmentation overlay, not just the enhanced image. The best filter is the one that preserves real branches without adding background structures.

## Step-by-step tuning

Change one setting at a time and inspect the output after each run.

### 1. Tune threshold first

The threshold controls how much of the enhanced image becomes vessel label:

* Lower the threshold when faint or incomplete vessels are missing.
* Raise the threshold when background noise or uneven illumination is included.
* Try small changes such as `30`, `40`, `50`, and `60`.

Do not compensate for an incorrect vessel scale by changing the threshold alone.

### 2. Match sigma values to vessel width

Sigma values are expressed in pixels and control the vessel sizes enhanced by the filter.

Suggested ranges:

| Image appearance | Sigma 1 |
| --- | --- |
| Mostly tiny vessels | `1 5 1` |
| Typical vessels around 10 px | `1 8 1` |
| Larger vessels | `2 15 2` |
| Broad range of sizes | `1 8 1` with multi-scale enabled |

Keep the largest sigma below the largest vessel width whenever possible. Excessively large values can make vessel boundaries appear wider than they are.

Use `Sigma 2` only when **Multi scale** is enabled. A useful mixed-size starting point is:

```text
Sigma 1: 1 8 1
Sigma 2: 10 30 5
Multi scale: yes
```

### 3. Tune ditzle size

`Ditzle size` removes connected objects smaller than the selected area in pixels:

* Lower it to preserve small real vessels, for example `100` or `250`.
* Raise it to remove isolated speckles, for example `500` or `1000`.

If peripheral or terminal vessels disappear, this setting is often too high.

### 4. Tune hole size

`Hole size` fills small holes inside segmented structures:

* Keep `50` for the general starting point.
* Increase it when vessels contain small internal gaps.
* Decrease it when nearby structures are being joined incorrectly.

## Image-specific presets

### Dim or low-contrast image

```text
Filter: jerman
Threshold: 30-40
Sigma 1: 1 8 1
Hole size: 50
Ditzle size: 250
Image type: as appropriate
Multi scale: no
```

Start at threshold `40`, then lower it only if real vessels remain missing.

### Noisy image

```text
Filter: meijering
Threshold: 50-70
Sigma 1: 1 8 1
Hole size: 50
Ditzle size: 500-1000
Image type: as appropriate
Multi scale: no
```

Raise the threshold before increasing ditzle size, because an aggressive ditzle size can remove real small branches.

### Mixed microvasculature and large vessels

```text
Filter: meijering
Threshold: 40
Sigma 1: 1 8 1
Sigma 2: 10 30 5
Hole size: 50
Ditzle size: 250-500
Image type: as appropriate
Multi scale: yes
```

Use this only when a single sigma range cannot represent both vessel populations.

## CZI projection guidance

For CZI files:

1. Select the correct channel. Channels start at `0`.
2. Use a smaller projection thickness when z-resolution and separation of nearby vessels are important.
3. Use a larger thickness when the vessel signal is faint and a stronger maximum-intensity projection is helpful.
4. Avoid projecting the entire stack when it causes overlapping vessels to merge.

Keep the channel and projection settings consistent across a dataset.

## How to judge the result

For each trial, inspect:

* The vessel label overlay against the original image.
* Whether faint branches are preserved.
* Whether background texture is incorrectly labelled.
* Whether nearby vessels merge.
* Whether the skeleton contains artificial branches or disconnected segments.
* Whether vessel diameters look plausible rather than systematically inflated.

For quantitative studies, validate settings on manually labelled representative images when possible. Use Jaccard or the vessel-specific accuracy metrics available in the package instead of choosing settings based only on visual sharpness.

Once settings are satisfactory:

1. Run the analysis on one representative image.
2. Save the settings to the output directory.
3. Review the generated label and skeleton outputs.
4. Apply the saved settings to the remaining images.
5. Keep the same acquisition channel, projection approach, and image scale across the dataset.

## Common problems

| Problem | First adjustment |
| --- | --- |
| Wrong structures are traced, vessels missed entirely | Check the image type (normal photos must be inverted) |
| Faint vessels are missing | Lower threshold; try `jerman` for dim data, or enable CLAHE |
| Vessels are broken by dust or debris | Set Remove dark specks to 4-6 |
| The sample edge is labelled as a vessel | Enable Mask the area outside the sample |
| Too much background is labelled | Raise threshold |
| Small real branches disappear | Lower ditzle size |
| Many isolated speckles remain | Raise threshold, then increase ditzle size |
| Vessel boundaries are too wide | Reduce the maximum sigma |
| Vessels have small internal gaps | Increase hole size slightly |
| Small and large vessels are not both detected | Enable multi-scale processing |
| Nearby vessels merge | Reduce projection thickness or sigma range |

No single preset guarantees accuracy for every microscope, staining method, or vessel scale. Keep the acquisition and segmentation settings documented with the results so analyses remain reproducible.
