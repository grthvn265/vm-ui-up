"""Vessel parameters: density, branchpoints, length, tortuosity."""
import cv2
import numpy as np
from skimage.draw import line

from ._fast import _coords_of, _label_coords, _text_origin, to_uint8
from .skeleton import _segment_midpoint, find_endpoints, skeletonize_vm


def tortuosity(edge_labels):
    edges = np.zeros_like(edge_labels)
    edges[edge_labels > 0] = 1
    coords, end_points = find_endpoints(edges)
    endpoint_labeled = edge_labels * end_points
    unique_labels, label_sizes = np.unique(edge_labels, return_counts=True)
    unique_labels = unique_labels[1:]
    label_sizes = label_sizes[1:]
    segment_ends = _label_coords(endpoint_labeled)
    tortuosity = []
    labels_used = []
    for u, segment_length in zip(unique_labels, label_sizes):
        end_inds = _coords_of(segment_ends, endpoint_labeled, u)
        if end_inds.size == 4:
            end_point_line = line(
                end_inds[0, 0], end_inds[0, 1], end_inds[1, 0], end_inds[1, 1]
            )
            endpoint_distance = np.max(np.shape(end_point_line))
            tortuosity.append(endpoint_distance / segment_length)
            labels_used.append(u)
    return tortuosity, labels_used


def vessel_length(edge_labels):
    unique_segments, segment_counts = np.unique(
        edge_labels, return_counts=True
    )
    unique_segments = unique_segments[1:]
    segment_counts = segment_counts[1:]
    return unique_segments, segment_counts


def network_length(edges):
    edges[edges > 0] = 1
    net_length = np.sum(edges)
    return net_length


def vessel_density_roi(im, label):
    density = np.zeros_like(im).astype(np.float16)
    density_array = []
    label[label > 0] = 1
    im_size = im.shape
    density = np.sum(label) / (im_size[0] * im_size[1])
    return density


def branchpoint_density_roi(skel, edges, bp):
    _, bp_labels = cv2.connectedComponents(bp.astype(np.uint8), connectivity=8)
    num_branchpoints = int(np.max(bp_labels))
    im_size = skel.shape
    bp_density = num_branchpoints / (im_size[0] * im_size[1])
    return num_branchpoints, bp_density


def make_labelled_image(im, seg):
    skel, edges, bp = skeletonize_vm(seg)
    _, edge_labels = cv2.connectedComponents(edges.astype(np.uint8))
    unique_labels = np.unique(edge_labels)[1:]
    overlay = np.clip(
        seg.astype(np.int32) * 50 + edges.astype(np.int32) * 200, 0, 255
    )
    output = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    segments = _label_coords(edge_labels)
    for i in unique_labels:
        seg_coords = _coords_of(segments, edge_labels, i)
        if len(seg_coords) > 5:
            try:
                midpoint = _segment_midpoint(seg_coords, edge_labels.shape)
                output = cv2.putText(
                    img=output,
                    text=str(i),
                    org=_text_origin(midpoint[1], midpoint[0]),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 255, 255),
                )
            except Exception as err:  # labelling is cosmetic, never fatal
                print("segment %s could not be labelled (%s)" % (i, err))
    return output


def vessel_density(im, label, num_tiles_x, num_tiles_y):
    density = np.zeros(im.shape, dtype=np.float32)
    density_array = []
    label[label > 0] = 1
    step_x = max(int(np.round(im.shape[0] / num_tiles_x)), 1)
    step_y = max(int(np.round(im.shape[1] / num_tiles_y)), 1)
    overlay = cv2.cvtColor(to_uint8(im), cv2.COLOR_GRAY2BGR)
    count = 0
    for x in range(0, im.shape[0], step_x):
        for y in range(0, im.shape[1], step_y):
            tile = label[x : x + step_x - 1, y : y + step_y - 1]
            numel = tile.shape[0] * tile.shape[1]
            tile_density = np.sum(tile) / numel if numel else 0
            tile_val = np.round(tile_density * 1000)
            density[x : x + step_x - 1, y : y + step_y - 1] = tile_val
            density_array.append(tile_val)

            overlay = cv2.putText(
                img=overlay,
                text=str(count),
                org=_text_origin(
                    np.round((y + y + step_y - 1) / 2),
                    np.round((x + x + step_x - 1) / 2),
                ),
                fontScale=1,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                color=(0, 255, 255),
            )
            count += 1
    return density.astype(np.uint16), density_array, overlay


def branchpoint_density(label):
    skel, edges, bp = skeletonize_vm(label)
    _, bp_labels = cv2.connectedComponents(bp.astype(np.uint8), connectivity=8)

    skel_inds = np.argwhere(skel > 0)
    overlay = np.clip(
        label.astype(np.int32) * 50 + edges.astype(np.int32) * 100, 0, 255
    )
    output = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    bp_density = []
    measurement_number = 0
    for i in range(0, len(skel_inds), 50):
        measurement_number += 1
        x, y = skel_inds[i]
        this_tile = bp_labels[max(x - 25, 0) : x + 25, max(y - 25, 0) : y + 25]
        bp_density.append(len(np.unique(this_tile[this_tile > 0])))
        output = cv2.putText(
            img=output,
            text=str(measurement_number),
            org=_text_origin(y, x),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=1,
            color=(0, 255, 255),
        )

    out_list = list(zip(range(1, measurement_number + 1), bp_density))
    return out_list, output
