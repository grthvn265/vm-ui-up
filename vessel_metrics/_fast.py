"""Small array helpers shared by the analysis modules.

The analysis used to loop over every pixel in python and rebuild a full-size
image for every segment and branchpoint. These helpers work on lists of pixel
coordinates instead and reproduce the original results exactly, including
numpy's slicing behaviour at the image border: a 3x3 window
im[i-1:i+2, j-1:j+2] at row or column 0 is an empty slice."""
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np


EMPTY_COORDS = np.empty((0, 2), dtype=np.intp)


FILTER_MEMORY_BUDGET = 1.5e9  # bytes the parallel vessel filters may use


def _window(i, j):
    return slice(i - 1, i + 2), slice(j - 1, j + 2)


def _box_sums(values, coords):
    """3x3 neighbourhood sums at coords, pixels outside the array count as 0."""
    h, w = values.shape
    rows, cols = coords[:, 0], coords[:, 1]
    total = np.zeros(len(coords))
    for dr in (-1, 0, 1):
        rr = rows + dr
        row_ok = (rr >= 0) & (rr < h)
        for dc in (-1, 0, 1):
            cc = cols + dc
            ok = row_ok & (cc >= 0) & (cc < w)
            total[ok] += values[rr[ok], cc[ok]]
    return total


def _window_sums(values, coords):
    """np.sum(values[i-1:i+2, j-1:j+2]) for every (i, j) in coords."""
    values = np.asarray(values)
    coords = np.asarray(coords).reshape(-1, 2)
    if len(coords) == 0:
        return np.zeros(0)
    out = _box_sums(values, coords)
    h, w = values.shape
    literal = (coords[:, 0] == 0) | (coords[:, 1] == 0)
    if h < 3 or w < 3:
        literal[:] = True
    for k in np.flatnonzero(literal):
        out[k] = np.sum(values[_window(*coords[k])])
    return out


def _group_rows(coords, keys):
    """{key: rows of coords with that key}, keeping the original row order."""
    if len(coords) == 0:
        return {}
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    groups = np.split(coords[order], starts[1:])
    return dict(zip(keys[starts].tolist(), groups))


def _label_coords(labels):
    """{label: np.argwhere(labels == label)} for every non-zero label at once."""
    coords = np.argwhere(labels)
    return _group_rows(coords, labels[coords[:, 0], coords[:, 1]])


def _coords_of(groups, labels, value):
    """np.argwhere(labels == value) using a _label_coords() lookup table."""
    if value == 0:
        return np.argwhere(labels == value)
    return groups.get(value.item() if hasattr(value, "item") else value,
                      EMPTY_COORDS)


def _closest_pairs(a, b, max_elements=2_000_000):
    """All (a[k], b[l]) pairs at the minimum distance, in a-major order."""
    if len(a) == 0 or len(b) == 0:
        return []
    chunk = max(1, max_elements // len(b))
    best, hits = None, []
    for s in range(0, len(a), chunk):
        d = a[s : s + chunk, None, :] - b[None, :, :]
        d2 = (d * d).sum(axis=-1)
        m = d2.min()
        if best is None or m < best:
            best, hits = m, []
        if m == best:
            ks, ls = np.nonzero(d2 == m)
            hits.extend(zip((ks + s).tolist(), ls.tolist()))
    return [(a[k], b[l]) for k, l in hits]


def _array_key(a):
    a = np.ascontiguousarray(a)
    return a.shape, a.dtype.str, hashlib.blake2b(a.tobytes(), digest_size=16).digest()


def _parallel_map(func, items, workers):
    items = list(items)
    workers = max(1, min(workers, len(items), os.cpu_count() or 1))
    if workers == 1:
        return [func(x) for x in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(func, items))


def _filter_workers(im):
    """Sigmas processed at once, so huge images do not run out of memory."""
    per_sigma = np.asarray(im).size * 8 * 8  # ~8 float64 images per sigma
    return int(max(1, FILTER_MEMORY_BUDGET // max(per_sigma, 1)))


def flatten(input_list):
    return [item for sublist in input_list for item in sublist]


def to_uint8(im):
    """Clip any numeric image to 0-255 and convert it so cv2.imwrite accepts it."""
    return np.clip(np.round(np.asarray(im, dtype=np.float64)), 0, 255).astype(
        np.uint8
    )


def split_name(file_name):
    """Split 'sample_slice3' into ('sample', '_slice3'); plain names get ''."""
    base, sep, number = file_name.rpartition("_slice")
    if sep and number.isdigit():
        return base, "_slice" + number
    return file_name, ""


def _savetxt(path, rows, fmt="%s"):
    """np.savetxt that also copes with an empty result."""
    if len(rows) == 0:
        open(path, "w").close()
    else:
        np.savetxt(path, rows, fmt=fmt, delimiter=",")


def _text_origin(x, y):
    """cv2.putText needs a tuple of plain ints."""
    return (int(x), int(y))
