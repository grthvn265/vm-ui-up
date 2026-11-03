"""Whole analysis runs: segment, measure and save every image, ROI analysis."""
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ._fast import _savetxt, split_name, to_uint8
from .diameter import whole_anatomy_diameter
from .fileio import generate_file_list, load_images, save_settings
from .logs import log, stage
from .measurements import (
    branchpoint_density,
    branchpoint_density_roi,
    make_labelled_image,
    network_length,
    tortuosity,
    vessel_density,
    vessel_density_roi,
    vessel_length,
)
from .segmentation import overlay_segmentation
from .settings import (
    _parse_int,
    _sigma_text,
    make_segmentation_settings,
    segment_with_settings,
)
from .skeleton import skeletonize_vm


def parameter_analysis(im, seg, params, output_path, file_name):
    """Measure the selected parameters and save the results next to the segmentation."""
    params = params or []
    base, suffix = split_name(file_name)
    folder = os.path.join(output_path, base)
    os.makedirs(folder, exist_ok=True)

    def out(prefix, ext=".txt"):
        return os.path.join(folder, prefix + suffix + ext)

    seg[seg > 0] = 1
    log("  measuring %s: %s" % (file_name, ", ".join(params) or "no parameters"))
    skel, edges, bp = skeletonize_vm(seg)
    edge_count, edge_labels = cv2.connectedComponents(edges.astype(np.uint8))
    log("    skeleton has %d segments" % (edge_count - 1))
    overlay = np.clip(
        seg.astype(np.int32) * 150 + skel.astype(np.int32) * 200, 0, 255
    )
    cv2.imwrite(out("vessel_centerlines", ".png"), overlay.astype(np.uint8))

    if "vessel density" in params:
        with stage("vessel density"):
            density_image, density_array, density_overlay = vessel_density(
                im, seg, 16, 16
            )
            fig = overlay_segmentation(im, density_image, show=False)
            fig.savefig(out("vessel_density", ".png"), bbox_inches="tight")
            plt.close(fig)
            cv2.imwrite(out("vessel_density_overlay", ".png"), density_overlay)
            _savetxt(
                out("vessel_density"),
                list(zip(range(len(density_array)), density_array)),
                fmt="%.1f",
            )
    if "branchpoint density" in params:
        with stage("branchpoint density"):
            bp_density, bp_overlay = branchpoint_density(seg)
            _savetxt(out("branchpoint_density"), bp_density, fmt="%d")
            cv2.imwrite(out("branchpoint_density_overlay", ".png"), bp_overlay)
    if "network length" in params:
        with stage("network length"):
            _savetxt(
                out("network_length"), [network_length(edges)], fmt="%.1f"
            )
    if "tortuosity" in params:
        with stage("tortuosity"):
            tort, labels_used = tortuosity(edge_labels)
            _savetxt(
                out("tortuosity"),
                list(zip(labels_used, tort)),
                fmt=["%d", "%.3f"],
            )
    if "segment length" in params:
        with stage("segment length"):
            segment_ids, length = vessel_length(edge_labels)
            _savetxt(
                out("vessel_length"),
                list(zip(segment_ids, length)),
                fmt="%d",
            )
    if "diameter" in params:
        with stage("vessel diameter (slowest measurement)"):
            _, diameters = whole_anatomy_diameter(
                im, seg, edge_labels, minimum_length=25, pad_size=50
            )
            _savetxt(out("vessel_diameter"), diameters, fmt=["%d", "%.1f"])
    log("  results for %s saved in %s" % (file_name, folder))


def crop_roi(im, roi):
    im_out = im[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
    return im_out


def generate_roi(im):
    """Let the user drag a rectangle on the image; returns (x, y, w, h)."""
    height, width = im.shape[:2]
    scale = min(1.0, 1024 / max(height, width))
    title = "Select ROI then space bar"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, int(width * scale), int(height * scale))
    roi = cv2.selectROI(title, to_uint8(im))
    cv2.destroyWindow(title)
    return roi


def crop_image(im):
    r = generate_roi(im)
    im_out = crop_roi(im, r)
    return im_out, r


def UI_crop_image(im, output_dir, suffix=""):
    im_out, r = crop_image(im)
    full = cv2.cvtColor(to_uint8(im), cv2.COLOR_GRAY2BGR)
    cv2.rectangle(
        full, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (0, 0, 255), 3
    )
    cv2.imwrite(os.path.join(output_dir, "full_im_roi" + suffix + ".png"), full)
    if im_out.size:
        cv2.imwrite(
            os.path.join(output_dir, "im_roi" + suffix + ".png"),
            to_uint8(im_out),
        )
    return im_out, r


def select_roi(image, output_dir, suffix="", confirm=None):
    """Interactive ROI selection; returns (cropped image, roi) or (None, None).

    Drag a rectangle in the image window and press space or enter. confirm(msg)
    -> bool lets the GUI ask "keep this region?"; without it the first crop is kept.
    """
    os.makedirs(output_dir, exist_ok=True)
    while True:
        cropped_im, r = UI_crop_image(image, output_dir, suffix)
        if cropped_im.size == 0:
            if confirm and confirm("No region was selected. Try again?"):
                continue
            return None, None
        if confirm is None or confirm(
            "Keep this region? Choose No to select it again."
        ):
            return cropped_im, r


def parameter_analysis_roi(im, seg, params, output_path, file_name, r):
    params = params or []
    base, suffix = split_name(file_name)
    folder = os.path.join(output_path, base)
    os.makedirs(folder, exist_ok=True)

    seg[seg > 0] = 1
    log("  ROI analysis for %s, region (x, y, w, h) = %s" % (file_name, tuple(int(v) for v in r)))
    skel, edges, bp = skeletonize_vm(seg)
    edge_count, edge_labels = cv2.connectedComponents(edges.astype(np.uint8))
    im = crop_roi(im, r)
    seg = crop_roi(seg, r)
    skel = crop_roi(skel, r)
    edges = crop_roi(edges, r)
    edge_labels = crop_roi(edge_labels, r)
    bp = crop_roi(bp, r)

    lines = ["ROI_analysis"]
    if "vessel density" in params:
        lines.append("vessel density: %.4f" % vessel_density_roi(im, seg))
    if "branchpoint density" in params:
        num_branchpoints, bp_density = branchpoint_density_roi(skel, edges, bp)
        lines.append("branchpoint density: %.6f" % bp_density)
        lines.append("Num branchpoints: %d" % num_branchpoints)
    if "network length" in params:
        lines.append("network length: %.1f pixels" % network_length(edges))
    if "segment length" in params:
        segment_ids, length = vessel_length(edge_labels)
        _savetxt(
            os.path.join(folder, "ROI_analysis_SL" + suffix + ".txt"),
            list(zip(segment_ids, length)),
            fmt="%d",
        )
    with open(os.path.join(folder, "ROI_analysis" + suffix + ".txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log("  ROI results for %s saved in %s" % (file_name, folder))


def analyze_images(images_to_analyze, file_names, settings, out_dir):
    """Segment every image and save img / label / vessel_labels pngs.

    Each file gets its own sub-folder of out_dir; returns the binary segmentations.
    """
    seg_list = []
    total = len(file_names)
    for n, (im, name) in enumerate(zip(images_to_analyze, file_names), 1):
        t0 = time.time()
        log(
            "[%d/%d] segmenting %s (%d x %d px)"
            % (n, total, name, im.shape[1], im.shape[0])
        )
        base, suffix = split_name(name)
        folder = os.path.join(out_dir, base)
        os.makedirs(folder, exist_ok=True)

        seg = segment_with_settings(im, settings)
        seg_list.append(seg)

        coverage = 100.0 * np.count_nonzero(seg) / seg.size
        log("    segmentation covers %.1f%% of the image" % coverage)
        if coverage == 0:
            log(
                "    WARNING: nothing was segmented. Try a lower threshold "
                "or a smaller ditzle size."
            )

        with stage("saving img and label pngs"):
            cv2.imwrite(os.path.join(folder, "img" + suffix + ".png"), to_uint8(im))
            cv2.imwrite(os.path.join(folder, "label" + suffix + ".png"), seg * 200)
        with stage("building labelled vessel image"):
            labelled = make_labelled_image(im, seg)
            cv2.imwrite(
                os.path.join(folder, "vessel_labels" + suffix + ".png"),
                labelled,
            )
        log(
            "[%d/%d] finished segmenting %s in %.1fs"
            % (n, total, name, time.time() - t0)
        )
    return seg_list


def analyze_single_image(image, file_name, settings, out_dir):
    return analyze_images([image], [file_name], settings, out_dir)[0]


class UserError(Exception):
    """A problem the user can fix; shown as a plain message, not a traceback."""


def run_analysis(cfg, confirm=None):
    """Run a full analysis from the form values; returns the number of images.

    cfg keys: mode ('single' or 'directory'), input, output, fields (the 8
    segmentation entries as text), params, channel, slices, save_settings, roi.
    confirm(message) -> bool is used by the ROI step to ask the user questions.
    """
    path, output_path = cfg["input"].strip(), cfg["output"].strip()
    if not path or not os.path.exists(path):
        raise UserError("Choose the image or folder to analyze.")
    if not output_path:
        raise UserError("Choose an output folder.")
    if cfg["mode"] == "single":
        if not os.path.isfile(path):
            raise UserError("Single image mode needs a file, not a folder.")
        files = [path]
    else:
        if not os.path.isdir(path):
            raise UserError("Directory mode needs a folder, not a file.")
        names, _ = generate_file_list(os.listdir(path))
        files = [os.path.join(path, n) for n in names]
    if not files:
        raise UserError(
            "No supported image files (czi, png, tif, jpg, bmp) were found."
        )
    flags = [f.lower().endswith(".czi") for f in files]
    if any(flags) and not all(flags):
        raise UserError(
            "Multiple file types detected (.czi volumes mixed with images). "
            "Please consolidate your file types."
        )
    is_czi = all(flags)

    run_start = time.time()
    log("=" * 60)
    log("Vessel Metrics run started")
    log("input  : %s" % path)
    log("output : %s" % output_path)
    log("found %d file(s) to analyze" % len(files))

    os.makedirs(output_path, exist_ok=True)
    settings = make_segmentation_settings(cfg["fields"])
    params = list(cfg["params"])
    czi_options = None
    if is_czi:
        czi_options = [
            _parse_int(cfg["channel"], 0, minimum=0),
            _parse_int(cfg["slices"], 20, minimum=1),
        ]
    if cfg.get("save_settings"):
        save_settings(
            [settings, params] + ([czi_options] if is_czi else []),
            os.path.join(output_path, "settings.data"),
        )
        log("settings saved to %s" % os.path.join(output_path, "settings.data"))

    log("settings: %s" % {k: (_sigma_text(v) if isinstance(v, range) else v) for k, v in settings.items()})
    log("measuring: %s" % (", ".join(params) or "nothing"))
    log("loading images ...")
    try:
        images, names = load_images(files, czi_options)
    except ValueError as err:  # e.g. channel does not exist
        raise UserError(str(err))
    if not images:
        raise UserError("None of the selected files could be read.")

    log("%d image(s) loaded, starting segmentation" % len(images))
    segmentations = analyze_images(images, names, settings, output_path)
    for n, (im, seg, name) in enumerate(zip(images, segmentations, names), 1):
        t0 = time.time()
        log("[%d/%d] measuring %s" % (n, len(names), name))
        parameter_analysis(im, seg, params, output_path, name)
        log("[%d/%d] finished measuring %s in %.1fs" % (n, len(names), name, time.time() - t0))

    if cfg.get("roi"):
        for im, seg, name in zip(images, segmentations, names):
            base, suffix = split_name(name)
            log("select a region for %s (an image window will open; drag a box, then press space or enter)" % name)
            _, r = select_roi(
                im, os.path.join(output_path, base), suffix, confirm
            )
            if r is not None:
                parameter_analysis_roi(im, seg, params, output_path, name, r)
    log("DONE: %d image(s) processed in %.1fs" % (len(images), time.time() - run_start))
    log("=" * 60)
    return len(images)
