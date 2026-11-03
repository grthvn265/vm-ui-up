"""Full-width-half-max vessel diameter along crosslines."""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import line

from ._fast import _coords_of, _label_coords
from .logs import log
from .skeleton import _segment_midpoint


def _tangent_slope(coords, point):
    """tangent_slope() for a segment given as row-major pixel coordinates."""
    point = np.asarray(point).flatten()
    r0, c0 = max(point[0] - 5, 0), max(point[1] - 5, 0)
    lo = np.searchsorted(coords[:, 0], r0, "left")
    hi = np.searchsorted(coords[:, 0], point[0] + 5, "left")
    near = coords[lo:hi]
    near = near[(near[:, 1] >= c0) & (near[:, 1] < point[1] + 5)]
    crop_inds = (near - [r0, c0]).astype(np.float32)
    fit = cv2.fitLine(crop_inds, cv2.DIST_L2, 0, 0.1, 0.1)
    return float(fit[0][0]), float(fit[1][0])


def _crossline_pixels(vx, vy, point, length):
    """Row and column indices of the crossline make_crossline() would draw."""
    xlen = vx * length / 2
    ylen = vy * length / 2
    x1 = int(np.round(point[0] - xlen))
    x2 = int(np.round(point[0] + xlen))
    y1 = int(np.round(point[1] - ylen))
    y2 = int(np.round(point[1] + ylen))
    rr, cc = line(x1, y1, x2, y2)
    return (x1, x2, y1, y2), rr, cc


def _segment_diameter(coords, shape, seg, im, use_label=False):
    """Crossline diameters of one segment (the core of visualize_vessel_diameter).

    Returns (diameters, mean diameter, crossline writes); writes is None when
    no crossline length could be found, otherwise a list of (rows, cols, value).
    """
    segment_median = _segment_midpoint(coords, shape)
    vx, vy = _tangent_slope(coords, segment_median)
    bx, by = crossline_slope(vx, vy)
    cross_length = find_crossline_length(bx, by, segment_median, seg)
    if cross_length == 0:
        print("cross length 0")
        return 0, 0, None

    diameter = []
    writes = []
    for i in range(10, len(coords), 10):
        this_point = coords[i]
        vx, vy = _tangent_slope(coords, this_point)
        bx, by = crossline_slope(vx, vy)
        _, rr, cc = _crossline_pixels(bx, by, this_point, cross_length)
        if use_label:
            diam = label_diameter(seg[rr, cc])
        else:
            diam = fwhm_diameter(im[rr, cc])
        writes.append((rr, cc, 5 if diam == 0 else 10))
        diameter.append(diam)
    diameter = [x for x in diameter if x != 0]
    if diameter:
        mean_diameter = np.mean(diameter)
    else:
        mean_diameter = 0
    return diameter, mean_diameter, writes


def _draw_writes(viz, writes):
    for rr, cc, val in writes:
        viz[rr, cc] = val


def _add_writes(total, writes):
    """total = total + (an empty image with writes drawn on it), in place."""
    if not writes:
        return
    h, w = total.shape
    flat = np.concatenate(
        [np.ravel_multi_index((rr, cc), (h, w), mode="wrap") for rr, cc, _ in writes]
    )
    vals = np.concatenate([np.full(len(rr), v) for rr, _, v in writes])
    # later crosslines overwrite earlier ones, so keep the last write per pixel
    flat, first = np.unique(flat[::-1], return_index=True)
    drawn = np.zeros(len(flat), dtype=total.dtype)
    drawn[:] = vals[::-1][first]
    view = total.reshape(-1)
    view[flat] = view[flat] + drawn


def whole_anatomy_diameter(
    im, seg, edge_labels, minimum_length=25, pad_size=50
):
    unique_edges = np.unique(edge_labels)
    unique_edges = np.delete(unique_edges, 0)

    edge_label_pad = np.pad(edge_labels, pad_size)
    seg_pad = np.pad(seg, pad_size)
    im_pad = np.pad(im, pad_size)
    full_viz = np.zeros_like(seg_pad)
    diameters = []
    included_segments = []
    total_edges = len(unique_edges)
    log_every = max(total_edges // 10, 1)
    segments = _label_coords(edge_label_pad)
    for n, i in enumerate(unique_edges, 1):
        if n % log_every == 0 or n == total_edges:
            log("    diameter: segment %d of %d" % (n, total_edges))
        coords = _coords_of(segments, edge_label_pad, i)
        seg_length = len(coords)
        if seg_length > minimum_length:
            included_segments.append(i)
            _, temp_diam, writes = _segment_diameter(
                coords, edge_label_pad.shape, seg_pad, im_pad
            )
            diameters.append(temp_diam)
            _add_writes(full_viz, writes)
    im_shape = edge_label_pad.shape
    full_viz_no_pad = full_viz[
        pad_size : im_shape[0] - pad_size, pad_size : im_shape[1] - pad_size
    ]
    output_diameters = list(zip(included_segments, diameters))
    return full_viz_no_pad, output_diameters


def visualize_vessel_diameter(
    edge_labels, segment_number, seg, im, use_label=False, pad=True
):
    if pad == True:
        pad_size = 25
        edge_labels = np.pad(edge_labels, pad_size)
        seg = np.pad(seg, pad_size)
        im = np.pad(im, pad_size)
    segment_inds = np.argwhere(edge_labels == segment_number)
    diameter, mean_diameter, writes = _segment_diameter(
        segment_inds, edge_labels.shape, seg, im, use_label
    )
    viz = np.zeros_like(seg)
    if writes is None:  # no crossline length found (returned unpadded)
        return diameter, mean_diameter, viz
    _draw_writes(viz, writes)

    if pad == True:
        im_shape = edge_labels.shape
        viz = viz[
            pad_size : im_shape[0] - pad_size,
            pad_size : im_shape[1] - pad_size,
        ]
    return diameter, mean_diameter, viz


def tangent_slope(segment, point):
    point = np.asarray(point).flatten()
    r0, c0 = max(point[0] - 5, 0), max(point[1] - 5, 0)
    crop_im = segment[r0 : point[0] + 5, c0 : point[1] + 5]
    crop_inds = np.transpose(np.where(crop_im)).astype(np.float32)
    fit = cv2.fitLine(crop_inds, cv2.DIST_L2, 0, 0.1, 0.1)
    return float(fit[0][0]), float(fit[1][0])


def crossline_slope(vx, vy):
    bx = -vy
    by = vx
    return bx, by


def make_crossline(vx, vy, point, length):
    coords, rr, cc = _crossline_pixels(vx, vy, point, length)
    cross_index = []
    for r, c in zip(rr, cc):
        cross_index.append([r, c])

    return coords, cross_index


def find_crossline_length(bx, by, point, im):
    d = 5
    diam = 0
    im_size = im.shape[0]
    while diam == 0:
        d += 5
        coords, rr, cc = _crossline_pixels(bx, by, point, d)
        if all(i < im_size for i in coords):
            seg_val = im[rr, cc]
            steps = np.where(np.roll(seg_val, 1) != seg_val)[0]
            if steps.size > 0:
                if steps[0] == 0:
                    steps = steps[1:]
                num_steps = len(steps)
                if num_steps == 2:
                    diam = abs(steps[1] - steps[0])
                if num_steps > 2:
                    new_steps = [
                        x - steps[i - 1] if i else None
                        for i, x in enumerate(steps)
                    ][1:]
                    diam = np.max(new_steps)
            if d > 100:
                break
        else:
            break
    length = diam * 2.5
    return length


def crossline_intensity(cross_index, im, plot=False):
    cross_vals = []
    for i in cross_index:
        cross_vals.append(im[i[0], i[1]])
    if plot == True:
        inds = list(range(len(cross_vals)))
        plt.figure()
        plt.plot(inds, cross_vals)
    return cross_vals


def label_diameter(cross_vals):
    steps = np.where(np.roll(cross_vals, 1) != cross_vals)[0]
    if steps.size > 0:
        if steps[0] == 0:
            steps = steps[1:]
        num_steps = len(steps)
        if num_steps == 2:
            diam = abs(steps[1] - steps[0])
        else:
            diam = 0
    else:
        diam = 0
    return diam


def fwhm_diameter(cross_vals):
    cross_vals = np.asarray(cross_vals, dtype=np.float64)
    peak = np.max(cross_vals)
    half_max = np.round(peak / 2)

    peak_ind = np.where(cross_vals == peak)[0][0]
    before_peak = cross_vals[0:peak_ind]
    after_peak = cross_vals[peak_ind + 1 :]
    try:
        hm_before = np.argmin(np.abs(before_peak - half_max))
        hm_after = np.argmin(np.abs(after_peak - half_max))

        # +2 added because array indexing begins at 0 twice
        diameter = (hm_after + peak_ind) - hm_before + 2
    except ValueError:  # peak at the very edge of the crossline
        diameter = 0
    return diameter
