"""Skeletonization and skeleton clean-up (branchpoints, endpoints, segments)."""
import itertools
from collections import OrderedDict
from math import dist

import cv2
import numpy as np
from scipy.ndimage import find_objects
from skimage.draw import line
from skimage.morphology import skeletonize

from ._fast import (
    EMPTY_COORDS,
    _array_key,
    _box_sums,
    _closest_pairs,
    _coords_of,
    _group_rows,
    _label_coords,
    _window,
    _window_sums,
    flatten,
)
from .logs import log, stage


SKELETON_CACHE_SIZE = 2  # skeletons kept so one image is never skeletonized twice


_skeleton_cache = OrderedDict()


def _segment_endpoints(coords, shape):
    """find_endpoints() coordinates of an image whose only pixels are coords."""
    coords = np.asarray(coords).reshape(-1, 2)
    if len(coords) == 0:
        return EMPTY_COORDS
    if shape[0] < 3 or shape[1] < 3:  # tiny image: use the original code
        mask = np.zeros(shape, np.uint8)
        mask[coords[:, 0], coords[:, 1]] = 1
        return find_endpoints(mask)[0]
    origin = np.maximum(coords.min(axis=0) - 1, 0)
    local = coords - origin
    mask = np.zeros(tuple(local.max(axis=0) + 2), np.uint8)
    mask[local[:, 0], local[:, 1]] = 1
    sums = _box_sums(mask, local)
    sums[(coords[:, 0] == 0) | (coords[:, 1] == 0)] = 0  # empty window
    return coords[sums == 2]


def _segment_midpoint(coords, shape):
    """segment_midpoint() for a segment given as row-major pixel coordinates."""
    endpoint_index = _segment_endpoints(coords, shape)
    if len(endpoint_index) == 0:
        reference = np.mean(coords, axis=0)
    else:
        reference = endpoint_index[0]
    distances = np.max(np.abs(coords - reference), axis=1)  # chebyshev
    order = np.argsort(distances, kind="stable")
    return coords[order[len(order) // 2]].flatten()


def skeletonize_vm(label):
    # The same segmentation is skeletonized for the labelled image, the
    # measurements and the branchpoint density, so the result is cached.
    key = _array_key(label)
    if key in _skeleton_cache:
        _skeleton_cache.move_to_end(key)
        log("  > skeleton already computed for this segmentation, reusing it")
        return tuple(a.copy() for a in _skeleton_cache[key])
    with stage("skeletonize"):
        skel = skeletonize(label)
    with stage("prune short terminal segments"):
        skel = _prune_skeleton(skel)
    with stage("fix skeleton artefacts"):
        edges, bp = fix_skel_artefacts(skel)
    new_skel = edges + bp
    _skeleton_cache[key] = (new_skel.copy(), edges.copy(), bp.copy())
    while len(_skeleton_cache) > SKELETON_CACHE_SIZE:
        _skeleton_cache.popitem(last=False)
    return new_skel, edges, bp


def find_branchpoints(skel):
    skel_binary = np.zeros_like(skel)
    skel_binary[skel > 0] = 1
    skel_index = np.argwhere(skel_binary == True)
    neighborhood_image = np.zeros(skel.shape)
    neighborhood_image[skel_index[:, 0], skel_index[:, 1]] = _window_sums(
        skel_binary, skel_index
    )
    branch_points_messy = np.zeros_like(neighborhood_image)
    branch_points_messy[neighborhood_image > 3] = 1
    branch_points_messy = branch_points_messy.astype(np.uint8)

    branch_points = branch_points_messy
    edges = np.zeros_like(branch_points_messy)
    edges = skel.astype(np.uint8) - branch_points_messy

    return edges, branch_points


def find_endpoints(edges):
    edge_binary = np.zeros_like(edges)
    edge_binary[edges > 0] = 1
    edge_index = np.argwhere(edge_binary == True)
    neighborhood_image = np.zeros(edges.shape)
    neighborhood_image[edge_index[:, 0], edge_index[:, 1]] = _window_sums(
        edge_binary, edge_index
    )
    end_points = np.zeros_like(neighborhood_image)
    end_points[neighborhood_image == 2] = 1
    coords = np.argwhere(end_points == 1)
    return coords, end_points


def find_terminal_segments(skel, edge_labels):
    skel[skel > 0] = 1
    skel_index = np.argwhere(skel == True)
    neighborhood_image = np.zeros(skel.shape)
    neighborhood_image[skel_index[:, 0], skel_index[:, 1]] = _window_sums(
        skel, skel_index
    )
    terminal_points = np.zeros_like(neighborhood_image)
    terminal_points[neighborhood_image == 2] = 1

    terminal_segments = np.zeros_like(terminal_points)
    unique_labels = np.unique(edge_labels)
    unique_labels = unique_labels[1:]

    # a segment is terminal when any of its pixels is a terminal point
    touching = np.unique(edge_labels[terminal_points == 1])
    terminal_labels = np.intersect1d(unique_labels, touching)
    terminal_segments[np.isin(edge_labels, terminal_labels)] = 1
    return terminal_segments


def connect_segments(skel):
    skel = np.pad(skel, 50)
    edges, bp = find_branchpoints(skel)
    _, edge_labels = cv2.connectedComponents(edges)

    edge_labels[edge_labels != 0] += 1
    bp_el = edge_labels + bp

    _, bp_labels = cv2.connectedComponents(bp)
    unique_bp = np.unique(bp_labels)
    unique_bp = unique_bp[1:]

    # Pixels of each branchpoint and endpoints of each segment, looked up per
    # label. Segments are 8-connected components, so the endpoints of the whole
    # edge image, grouped by label, are exactly the endpoints of each segment.
    bp_pixels = _label_coords(bp_labels)
    end_coords, _ = find_endpoints(edges)
    segment_ends = _group_rows(
        end_coords, edge_labels[end_coords[:, 0], end_coords[:, 1]]
    )
    fits = {}

    # the new edges / branchpoints are summed here; dtypes are restored below
    edge_sum = np.zeros(skel.shape, np.int32)
    bp_sum = np.zeros(skel.shape, np.int32)
    any_match = False
    for i in unique_bp:
        this_bp_inds = _coords_of(bp_pixels, bp_labels, i)
        bp_size = len(this_bp_inds)
        if bp_size > 1:
            connected_segs = []
            bp_coords = []
            for x, y in this_bp_inds:
                bp_neighbors = bp_el[x - 1 : x + 2, y - 1 : y + 2]
                if np.any(bp_neighbors > 1):
                    connections = bp_neighbors[bp_neighbors > 1].tolist()
                    connected_segs.append(connections)
                    for c in connections:
                        bp_coords.append((x, y))
            connected_segs = flatten(connected_segs)

            vx = []
            vy = []
            for seg in connected_segs:
                if seg not in fits:
                    endpoints = segment_ends.get(seg, EMPTY_COORDS)
                    fits[seg] = None
                    if np.size(endpoints):
                        fits[seg] = cv2.fitLine(
                            endpoints.astype(np.float32), cv2.DIST_L2, 0, 0.1, 0.1
                        )
                fit = fits[seg]
                if fit is not None:
                    vx.append(float(fit[0][0]))
                    vy.append(float(fit[1][0]))

            vx = np.array(vx).flatten().tolist()
            vy = np.array(vy).flatten().tolist()

            v_r = list(zip(np.round(vx, 3), np.round(vy, 3)))
            slope_tolerance = 0.1

            inds = list(range(len(v_r)))
            pair_inds = list(itertools.combinations(inds, 2))
            count = 0
            match = []
            for x, y in itertools.combinations(v_r, 2):
                if np.abs(x[0] - y[0]) < slope_tolerance:
                    if np.abs(x[1] - y[1]) < slope_tolerance:
                        match = pair_inds[count]
                count += 1

            if match:
                c1 = bp_coords[match[0]]
                c2 = bp_coords[match[1]]
                rr, cc = line(c1[0], c1[1], c2[0], c2[1])
                edge_sum[rr, cc] += 1
                any_match = True
                on_line = set(zip(rr.tolist(), cc.tolist()))
                for x, y in this_bp_inds:
                    if (int(x), int(y)) not in on_line:
                        bp_neighbors = edges[x - 1 : x + 2, y - 1 : y + 2]
                        if np.any(bp_neighbors > 0):
                            bp_sum[x, y] += 1
            else:
                bp_sum[this_bp_inds[:, 0], this_bp_inds[:, 1]] += 1
        else:
            bp_sum[this_bp_inds[:, 0], this_bp_inds[:, 1]] += 1

    # same dtypes the original running sums new_edges / new_bp ended up with
    edges_dtype = np.result_type(skel.dtype, np.int32) if any_match else skel.dtype
    bp_dtype = np.result_type(skel.dtype, np.int32) if len(unique_bp) else skel.dtype
    new_edges = (edge_sum + edges).astype(np.result_type(edges_dtype, edges.dtype))
    new_bp = bp_sum.astype(bp_dtype)
    xdim, ydim = np.shape(skel)

    new_edges = new_edges[50 : xdim - 50, 50 : ydim - 50]
    new_bp = new_bp[50 : xdim - 50, 50 : ydim - 50]
    return new_edges, new_bp


def prune_terminal_segments(skel, seg_thresh=20):
    new_skel = _prune_skeleton(skel, seg_thresh)
    edges, bp = connect_segments(new_skel)
    return edges, bp, new_skel


def _prune_skeleton(skel, seg_thresh=20):
    """The new_skel part of prune_terminal_segments (all skeletonize_vm uses)."""
    edges, bp = connect_segments(skel)
    _, edge_labels = cv2.connectedComponents(edges.astype(np.uint8))
    _, bp_labels = cv2.connectedComponents(bp.astype(np.uint8))

    terminal_segs = find_terminal_segments(skel, edge_labels)
    _, term_labels = cv2.connectedComponents(terminal_segs.astype(np.uint8))
    unique_labels = np.unique(term_labels)[1:]  # omit 0
    seg_lengths = np.bincount(term_labels.ravel())
    short = unique_labels[seg_lengths[unique_labels] < seg_thresh]
    null_points = np.zeros_like(terminal_segs)
    if short.size:
        # terminal segments are 8-connected components, so their endpoints
        # are the endpoints of the whole terminal segment image
        end_coords, _ = find_endpoints(term_labels)
        segment_ends = _group_rows(
            end_coords, term_labels[end_coords[:, 0], end_coords[:, 1]]
        )
        h, w = bp_labels.shape
        for u in short.tolist():
            for i in segment_ends.get(u, EMPTY_COORDS):
                rows, cols = _window(i[0], i[1])
                endpoint_neighborhood = bp_labels[rows, cols]
                if np.any(endpoint_neighborhood > 0):
                    first = np.argwhere(endpoint_neighborhood > 0)[0]
                    null_points[
                        rows.indices(h)[0] + first[0], cols.indices(w)[0] + first[1]
                    ] = 1
        null_points[np.isin(term_labels, short)] = 1
    new_skel = skel - null_points
    new_skel[new_skel < 0] = 0
    new_skel[new_skel > 0] = 1
    return new_skel


def fix_skel_artefacts(skel):
    edges, bp = find_branchpoints(skel)
    edge_count, edge_labels = cv2.connectedComponents(edges.astype(np.uint8))
    bp_count, bp_labels = cv2.connectedComponents(bp.astype(np.uint8))
    new_edge_num = edge_count + 1
    bp_pixels = _label_coords(bp_labels)
    seg_boxes = find_objects(edge_labels)
    line_box = None  # bounding box of the pixels relabelled new_edge_num

    def current_pixels(c):
        """np.argwhere(edge_labels == c) for the edge_labels being edited."""
        if 0 < c < edge_count:
            box = seg_boxes[c - 1]  # lines only remove pixels from label c
        elif c == new_edge_num:
            box = line_box
        else:
            box = None
        if box is None:
            return EMPTY_COORDS
        found = np.argwhere(edge_labels[box] == c)
        return found + [box[0].start, box[1].start]

    # bp_labels[1:] drops the first image row, not label 0, so the background
    # (label 0) is visited too; _background_connections handles it quickly.
    for i in np.unique(bp_labels[1:]):
        if i == 0:
            connected_segs = _background_connections(bp_labels, edge_labels)
        else:
            connected_segs = _bp_connections(
                bp_labels, edge_labels, i, _coords_of(bp_pixels, bp_labels, i)
            )
        if connected_segs is not None and len(connected_segs) == 2:
            coord_list = [current_pixels(c) for c in connected_segs]
            lowest_dist = 500
            # the original compared every pair; only the closest pairs can win
            for x, y in _closest_pairs(coord_list[0], coord_list[1]):
                this_dist = dist(x, y)
                if this_dist < lowest_dist:
                    lowest_dist = this_dist
                    end1, end2 = x, y
            if i != 0:
                pixels = _coords_of(bp_pixels, bp_labels, i)
                bp_labels[pixels[:, 0], pixels[:, 1]] = 0

            rr, cc = line(end1[0], end1[1], end2[0], end2[1])
            edge_labels[rr, cc] = new_edge_num
            r0, r1 = int(rr.min()), int(rr.max()) + 1
            c0, c1 = int(cc.min()), int(cc.max()) + 1
            if line_box is not None:
                r0, r1 = min(r0, line_box[0].start), max(r1, line_box[0].stop)
                c0, c1 = min(c0, line_box[1].start), max(c1, line_box[1].stop)
            line_box = (slice(r0, r1), slice(c0, c1))

    new_edges = np.zeros_like(edge_labels)
    new_bp = np.zeros_like(edge_labels)

    new_edges[edge_labels > 0] = 1
    new_bp[bp_labels > 0] = 1

    edge_count, edge_labels = cv2.connectedComponents(
        new_edges.astype(np.uint8)
    )

    # remove short segments that touch fewer than 2 branchpoints
    bp_num = _branchpoints_per_segment(skel, edge_labels, bp, edge_count)
    seg_length = np.bincount(edge_labels.ravel(), minlength=edge_count)
    remove = (bp_num < 2) & (seg_length < 10)
    remove[0] = False
    edge_labels[remove[edge_labels]] = 0

    new_skel = edge_labels + new_bp
    new_skel[new_skel > 0] = 1
    new_edges, new_bp = find_branchpoints(new_skel)
    return new_edges, new_bp


def branchpoints_per_seg(skel, edge_labels, bp, seg_num):
    bp_count, bp_labels = cv2.connectedComponents(bp.astype(np.uint8))
    bp_labels = bp_labels + 1
    bp_labels[bp_labels < 2] = 0
    temp_seg = np.zeros_like(skel)
    temp_seg[edge_labels == seg_num] = 1
    temp_seg = temp_seg + bp_labels
    seg_inds = np.argwhere(temp_seg == 1)
    seg_lenth = np.shape(seg_inds)[0]
    bp_num = 0
    for i in seg_inds:
        this_tile = temp_seg[i[0] - 1 : i[0] + 2, i[1] - 1 : i[1] + 2]
        unique_bps = np.unique(this_tile)
        unique_bps = np.sum(unique_bps > 1)
        bp_num = bp_num + unique_bps
    return bp_num


def find_connected_segments(bp_labels, edge_labels, bp_num):
    this_bp_inds = np.argwhere(bp_labels == bp_num)
    temp_bp = np.zeros_like(bp_labels)
    for i in this_bp_inds:
        temp_bp[i[0], i[1]] = -1
    bp_el = edge_labels + temp_bp
    connected_segs = []
    bp_coords = []
    for x, y in this_bp_inds:
        bp_neighbors = bp_el[x - 1 : x + 2, y - 1 : y + 2]
        if np.any(bp_neighbors > 0):
            connections = bp_neighbors[bp_neighbors > 0].tolist()
            connected_segs.append(connections)
            for c in connections:
                bp_coords.append((x, y))
    connected_segs = flatten(connected_segs)
    return connected_segs


def _bp_connections(bp_labels, edge_labels, bp_num, this_bp_inds):
    """find_connected_segments() for one branchpoint, without full-size images."""
    connected_segs = []
    for x, y in this_bp_inds:
        rows, cols = _window(x, y)
        # bp_el = edge_labels with -1 added on this branchpoint's pixels
        bp_neighbors = edge_labels[rows, cols] - (bp_labels[rows, cols] == bp_num)
        if np.any(bp_neighbors > 0):
            connected_segs.append(bp_neighbors[bp_neighbors > 0].tolist())
    return flatten(connected_segs)


def _background_connections(bp_labels, edge_labels):
    """find_connected_segments(..., 0), or None when it would not have length 2.

    Label 0 is every non-branchpoint pixel; fix_skel_artefacts only acts when
    exactly 2 connections are found, so they are counted before being listed.
    """
    h, w = bp_labels.shape
    if h < 3 or w < 3:
        return find_connected_segments(bp_labels, edge_labels, 0)
    background = bp_labels == 0
    bp_el = edge_labels - background
    positive = np.pad((bp_el > 0).astype(np.int32), 1)
    counts = sum(
        positive[dr : dr + h, dc : dc + w] for dr in range(3) for dc in range(3)
    )
    counts[0, :] = 0  # windows starting at index -1 are empty slices
    counts[:, 0] = 0
    counts[~background] = 0
    if counts.sum() != 2:
        return None
    connected_segs = []
    for x, y in np.argwhere(counts > 0):
        bp_neighbors = bp_el[_window(x, y)]
        connected_segs.append(bp_neighbors[bp_neighbors > 0].tolist())
    return flatten(connected_segs)


def _branchpoints_per_segment(skel, edge_labels, bp, edge_count):
    """branchpoints_per_seg() for every label 0..edge_count-1 at once."""
    _, bp_labels = cv2.connectedComponents(bp.astype(np.uint8))
    bp_labels = bp_labels + 1
    bp_labels[bp_labels < 2] = 0
    h, w = edge_labels.shape
    # pixels of a segment that are not on a branchpoint (temp_seg == 1)
    coords = np.argwhere((edge_labels > 0) & (bp_labels == 0))
    seg_ids = edge_labels[coords[:, 0], coords[:, 1]]
    # value of every 3x3 neighbour in temp_seg = (edge_labels == seg) + bp_labels,
    # keeping only values > 1 (branchpoints); 0 marks "nothing to count"
    neighbours = np.zeros((len(coords), 9), np.int64)
    k = 0
    for dr in (-1, 0, 1):
        rr = coords[:, 0] + dr
        for dc in (-1, 0, 1):
            cc = coords[:, 1] + dc
            ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            b = np.zeros(len(coords), np.int64)
            b[ok] = bp_labels[rr[ok], cc[ok]]
            same = np.zeros(len(coords), bool)
            same[ok] = edge_labels[rr[ok], cc[ok]] == seg_ids[ok]
            neighbours[:, k] = np.where(b > 1, b + same, 0)
            k += 1
    neighbours.sort(axis=1)
    distinct = (neighbours > 1) & np.c_[
        np.ones(len(coords), bool), neighbours[:, 1:] != neighbours[:, :-1]
    ]
    per_pixel = distinct.sum(axis=1)
    # windows at row / column 0 are evaluated literally (numpy slicing quirk)
    literal = (coords[:, 0] == 0) | (coords[:, 1] == 0)
    if h < 3 or w < 3:
        literal[:] = True
    for k in np.flatnonzero(literal):
        rows, cols = _window(*coords[k])
        tile = (edge_labels[rows, cols] == seg_ids[k]) + bp_labels[rows, cols]
        per_pixel[k] = np.sum(np.unique(tile) > 1)
    return np.bincount(seg_ids, weights=per_pixel, minlength=edge_count)


def segment_midpoint(segment):
    """Pixel of the segment that sits at the median distance from one end."""
    endpoint_index, _ = find_endpoints(segment)
    segment_indexes = np.argwhere(segment)
    if len(endpoint_index) == 0:
        # no endpoints: a closed loop, measure from its centre of mass
        reference = np.mean(segment_indexes, axis=0)
    else:
        reference = endpoint_index[0]
    distances = np.max(np.abs(segment_indexes - reference), axis=1)  # chebyshev
    order = np.argsort(distances, kind="stable")
    return segment_indexes[order[len(order) // 2]].flatten()
