"""Preprocessing, vessel enhancement filters, segmentation and its scoring."""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import frangi, meijering, sato
from skimage.morphology import black_tophat, disk, skeletonize, white_tophat
from skimage.util import invert

from ._fast import _filter_workers, _parallel_map
from .logs import stage


# What the user says the input image is. The rest of the pipeline expects
# bright vessels on a dark background, so "normal" photos are inverted first.
IMAGE_TYPE_CHOICES = ["normal", "inverted", "preprocessed"]


IMAGE_TYPE_LABELS = {
    "normal": "Normal photo (dark vessels on a light background)",
    "inverted": "Inverted / fluorescence (bright vessels on a dark background)",
    "preprocessed": "Pre-processed (bright vessels, background already removed)",
}


COLOUR_CHANNEL_CHOICES = ["grayscale", "green", "red", "blue"]


def segment_image(
    im,
    im_filter="meijering",
    sigma1=range(1, 8, 1),
    sigma2=range(10, 20, 5),
    hole_size=50,
    ditzle_size=500,
    thresh=40,
    preprocess=True,
    multi_scale=False,
    clahe=False,
    speck_size=0,
    mask_outside=False,
):
    im, outside = preprocess_for_segmentation(
        im, preprocess, clahe, speck_size, mask_outside
    )

    with stage("vessel enhancement filter: %s, sigma %s" % (im_filter, sigma1)):
        norm = normalize_contrast(_enhance(im, im_filter, sigma1)).astype(
            np.uint16
        )
    if multi_scale:
        with stage("second scale filter: %s, sigma %s" % (im_filter, sigma2)):
            sig2 = normalize_contrast(_enhance(im, im_filter, sigma2))
            norm = norm + sig2.astype(np.uint16)

    with stage("thresholding at %s and morphological opening" % thresh):
        enhanced_label = np.zeros(norm.shape, dtype=np.uint8)
        enhanced_label[norm > thresh] = 1

        kernel = np.ones((6, 6), np.uint8)
        label = cv2.morphologyEx(enhanced_label, cv2.MORPH_OPEN, kernel)
        if outside is not None:
            # the sample edge itself gives a ridge response, so keep a margin
            margin = cv2.dilate(outside.astype(np.uint8), disk(8).astype(np.uint8))
            label[margin > 0] = 0

    with stage("filling holes < %s px and removing objects < %s px" % (hole_size, ditzle_size)):
        label = fill_holes(label, hole_size)
        label = remove_small_objects(label, ditzle_size)
    return label


def preprocess_for_segmentation(
    im, preprocess=True, clahe=False, speck_size=0, mask_outside=False
):
    """Everything segment_image does before the vessel filter.

    Returns (image for the filter, boolean mask of the area outside the
    sample or None when mask_outside is off).
    """
    outside = None
    if mask_outside:
        with stage("masking the area outside the sample"):
            outside = outside_sample_mask(im)
            im = fill_outside(im, outside)
    if preprocess:
        steps = ["background subtraction", "median blur"]
        if speck_size:
            steps.insert(0, "dark speck removal r=%d" % speck_size)
        if clahe:
            steps.insert(len(steps) - 2, "CLAHE")
        with stage("preprocessing (%s)" % ", ".join(steps)):
            im = preprocess_seg(im, clahe=clahe, speck_size=speck_size)
    return im, outside


def invert_image(image):
    """Swap dark and light, keeping the value range (the first preprocessing step
    for normal photos, so vessels become bright like in fluorescence images)."""
    image = np.asarray(image)
    if image.dtype.kind in "ui":
        return np.iinfo(image.dtype).max - image
    return np.max(image) + np.min(image) - image


def prepare_image(raw, image_type=None, colour_channel="grayscale"):
    """Turn a loaded image into the single-channel image that is analyzed.

    Inversion is done first, on the untouched (colour) image, before any other
    change; then the image is reduced to grayscale or one colour channel.
    """
    image = np.asarray(raw)
    if image_type == "normal":
        image = invert_image(image)
    if image.ndim == 3:
        if image.shape[2] == 4:  # drop alpha
            image = image[:, :, :3]
        if colour_channel in ("blue", "green", "red"):
            image = image[:, :, ("blue", "green", "red").index(colour_channel)]
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.ascontiguousarray(image)


def outside_sample_mask(image, min_fraction=0.005):
    """Pixels outside the sample: near-black or near-white regions that touch
    the image border (e.g. the dark surround of a cropped photo)."""
    im = cv2.medianBlur(normalize_contrast(image).astype(np.uint8), 9)
    outside = np.zeros(im.shape, dtype=bool)
    for candidate in (im <= 12, im >= 243):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate.astype(np.uint8)
        )
        touching = np.unique(
            np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
        )
        big = [n for n in touching if n and stats[n, 4] >= min_fraction * im.size]
        outside |= np.isin(labels, big)
    return outside


def fill_outside(image, outside):
    """Replace the outside area with the sample's median so its edge does not
    look like a vessel to the background subtraction and filters."""
    if not outside.any() or outside.all():
        return image
    image = np.array(image, copy=True)
    image[outside] = np.median(image[~outside])
    return image


def remove_dark_specks(image, radius):
    """Grayscale closing: fills dark specks (dust, debris) narrower than
    2 * radius that would otherwise break bright vessels apart."""
    kernel = disk(radius).astype(np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def classify_image_type(image):
    """Guess whether an image is normal, inverted or pre-processed.

    Returns (image type, explanation). Compares bright-ridge and dark-ridge
    filter responses inside the sample; a pre-processed image additionally has
    a near-black background and a contrast stretch that saturates the vessels.
    """
    im = normalize_contrast(image).astype(np.uint8)
    scale = 512 / max(im.shape)
    if scale < 1:
        im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    outside = outside_sample_mask(im)
    inside = cv2.dilate(outside.astype(np.uint8), disk(6).astype(np.uint8)) == 0
    if inside.sum() < 0.05 * im.size:
        inside = np.ones(im.shape, dtype=bool)
    im = fill_outside(im, outside).astype(np.float64) / 255
    bright = np.percentile(sato(im, sigmas=[1, 2, 3], black_ridges=False)[inside], 95)
    dark = np.percentile(sato(im, sigmas=[1, 2, 3], black_ridges=True)[inside], 95)
    ratio = bright / max(dark, 1e-9)
    values = im[inside]
    median, saturated = np.median(values), np.mean(values >= 250 / 255)

    if ratio < 0.95:
        return "normal", (
            "vessels look darker than the background (bright/dark ridge "
            "ratio %.2f)" % ratio
        )
    if ratio > 1.05 and median <= 0.3 and saturated >= 0.03:
        return "preprocessed", (
            "bright vessels on a near-black background with a contrast "
            "stretch (median %.0f, %.0f%% saturated)" % (median * 255, saturated * 100)
        )
    if ratio > 1.05:
        return "inverted", (
            "vessels look brighter than the background (bright/dark ridge "
            "ratio %.2f)" % ratio
        )
    return None, "vessel polarity is unclear (bright/dark ridge ratio %.2f)" % ratio


def preprocess_seg(
    image,
    radius=50,
    median_size=7,
    upper_lim=255,
    lower_lim=0,
    bright_background=False,
    clahe=False,
    speck_size=0,
):
    image = normalize_contrast(image)
    if speck_size or clahe:
        image = image.astype(np.uint8)
    if speck_size:
        image = remove_dark_specks(image, speck_size)
    if clahe:
        image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)

    image = subtract_background(
        image, radius=radius, light_bg=bright_background
    )
    image = cv2.medianBlur(image.astype(np.uint8), median_size)
    image = contrast_stretch(image, upper_lim=upper_lim, lower_lim=lower_lim)
    return image


def overlay_segmentation(
    im,
    label,
    alpha=0.5,
    contrast_adjust=False,
    im_cmap="gray",
    label_cmap="jet",
    show=True,
):
    if contrast_adjust:
        im = contrast_stretch(im)
        im = preprocess_seg(im)
    masked = np.ma.masked_where(label == 0, label)
    fig = plt.figure()
    plt.imshow(im, im_cmap, interpolation="none")
    plt.imshow(masked, label_cmap, interpolation="none", alpha=alpha)
    if show:
        plt.show(block=False)
    return fig


def fill_holes(label_binary, hole_size):
    """Fill background holes smaller than hole_size pixels; returns the new mask."""
    label_inv = np.where(label_binary > 0, 0, 255).astype(np.uint8)
    _, inverted_labels, stats, _ = cv2.connectedComponentsWithStats(label_inv)
    small = np.flatnonzero(stats[:, 4] < hole_size)
    inverted_labels[np.isin(inverted_labels, small)] = 0
    return (inverted_labels == 0).astype(np.uint8)


def normalize_contrast(image):
    image = np.asarray(image)
    peak = np.max(image)
    if peak <= 0:
        return np.zeros(image.shape, dtype=np.float64)
    return np.floor(image / peak * 255)


def remove_small_objects(label, size_thresh):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        (np.asarray(label) > 0).astype(np.uint8)
    )
    keep = stats[:, 4] >= size_thresh
    keep[0] = False  # label 0 is the background
    return keep[labels].astype(np.uint8)


def subtract_background(image, radius=50, light_bg=False):
    str_el = disk(radius)
    fast = _fast_tophat(image, str_el, radius, light_bg)
    if fast is not None:
        return fast
    if light_bg:
        output = black_tophat(image, str_el)
    else:
        output = white_tophat(image, str_el)
    return output


def _fast_tophat(image, str_el, radius, light_bg):
    """OpenCV top hat, ~300x faster than scikit-image and bit-identical for
    8-bit valued images larger than the disk; None when that does not apply."""
    image = np.asarray(image)
    if image.ndim != 2 or min(image.shape) <= radius:
        return None
    if image.dtype == np.uint8:
        im8 = image
    elif image.dtype.kind == "f":
        im8 = image.astype(np.uint8)
        if not np.array_equal(im8, image):  # not whole numbers in 0-255
            return None
    else:
        return None
    kernel = str_el.astype(np.uint8)
    if light_bg:
        closed = cv2.morphologyEx(im8, cv2.MORPH_CLOSE, kernel)
        return closed.astype(image.dtype) - image
    opened = cv2.morphologyEx(im8, cv2.MORPH_OPEN, kernel)
    return image - opened.astype(image.dtype)


def jerman(
    im,
    sigmas=range(1, 10, 2),
    tau=0.75,
    brightondark=True,
    cval=0,
    mode="reflect",
):
    if brightondark == False:
        im = invert(im)
    vesselness = np.zeros_like(im)

    def sigma_response(sigma):
        try:  # keep the behaviour this filter was written against
            h_elems = hessian_matrix(
                im, sigma, mode=mode, cval=cval, use_gaussian_derivatives=False
            )
        except TypeError:  # older scikit-image has no such argument
            h_elems = hessian_matrix(im, sigma, mode=mode, cval=cval)
        lambda1, lambda2 = hessian_matrix_eigvals(h_elems)
        if brightondark == True:
            lambda2 = -lambda2
        lambda3 = lambda2

        lambda_rho = lambda3
        lambda_rho = np.where(
            (lambda3 > 0) & (lambda3 <= tau * np.max(lambda3)),
            tau * np.max(lambda3),
            lambda_rho,
        )

        lambda_rho[lambda3 < 0] = 0

        with np.errstate(divide="ignore", invalid="ignore"):
            response = (
                lambda2
                * lambda2
                * (lambda_rho - lambda2)
                * 27
                / np.power(lambda2 + lambda_rho, 3)
            )

        response = np.where(
            (lambda2 >= lambda_rho / 2) & (lambda_rho > 0), 1, response
        )
        return np.where((lambda2 <= 0) | (lambda_rho <= 0), 0, response)

    # every sigma is independent, so they are filtered in parallel threads
    responses = _parallel_map(sigma_response, sigmas, _filter_workers(im))
    for i, response in enumerate(responses):
        if i == 0:
            vesselness = response
        else:
            vesselness = np.maximum(vesselness, response)
    vesselness = vesselness / np.max(vesselness)
    vesselness[vesselness < 0.001] = 0
    return vesselness


def contrast_stretch(image, upper_lim=255, lower_lim=0):
    c = np.percentile(image, 5)
    d = np.percentile(image, 95)
    if d == c:  # flat image, nothing to stretch
        return np.asarray(image, dtype=np.float64)

    stretch = (image - c) * ((upper_lim - lower_lim) / (d - c)) + lower_lim
    stretch[stretch < lower_lim] = lower_lim
    stretch[stretch > upper_lim] = upper_lim

    return stretch


def jaccard(ground_truth, label):
    ground_truth_binary = np.zeros_like(ground_truth)
    label_binary = np.zeros_like(label)

    ground_truth_binary[ground_truth > 0] = 1
    label_binary[label > 0] = 1

    intersection = np.sum(np.logical_and(ground_truth_binary, label_binary))
    union = np.sum(np.logical_or(ground_truth_binary, label_binary))

    jacc = round(intersection / union, 2)
    return jacc


def cal(ground_truth, label):
    ground_truth_binary = np.zeros_like(ground_truth)
    label_binary = np.zeros_like(label)

    ground_truth_binary[ground_truth > 0] = 1
    label_binary[label > 0] = 1

    num_labels_gt, labels_gt = cv2.connectedComponents(
        ground_truth_binary, connectivity=8
    )
    num_labels_l, labels_l = cv2.connectedComponents(
        label_binary, connectivity=8
    )

    connectivity = round(
        1
        - np.min(
            [
                1,
                np.abs(num_labels_gt - num_labels_l)
                / np.sum(ground_truth_binary),
            ]
        ),
        2,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilation_gt = cv2.dilate(ground_truth_binary, kernel)
    dilation_l = cv2.dilate(label_binary, kernel)

    dilated_label_union = np.logical_and(
        ground_truth_binary == 1, dilation_l == 1
    )
    dilated_gt_union = np.logical_and(label_binary == 1, dilation_gt == 1)

    area_numerator = np.sum(
        np.logical_or(dilated_label_union, dilated_gt_union)
    )
    area_denominator = np.sum(np.logical_or(label_binary, ground_truth_binary))

    area = round(area_numerator / area_denominator, 2)

    gt_skeleton = skeletonize(ground_truth_binary)
    l_skeleton = skeletonize(label_binary)

    label_skel_int = np.logical_and(l_skeleton, dilation_gt)
    gt_skel_int = np.logical_and(gt_skeleton, dilation_l)

    length_numerator = np.sum(np.logical_or(label_skel_int, gt_skel_int))
    length_denominator = np.sum(np.logical_or(l_skeleton, gt_skeleton))

    length = round(length_numerator / length_denominator, 2)

    Q = connectivity * length * area

    return length, area, connectivity, Q


def cohens_kappa_segmentation(seg1, seg2):
    a = np.asarray(seg1) > 0
    b = np.asarray(seg2) > 0
    n = a.size
    A = np.sum(a & b)
    B = np.sum(a & ~b)
    C = np.sum(~a & b)
    D = np.sum(~a & ~b)
    P = (A + D) / n
    Pe = ((A + B) * (A + C) + (C + D) * (B + D)) / n**2
    if Pe == 1:
        return 1.0
    return (P - Pe) / (1 - Pe)


def signal_to_noise(image):
    step_size = [round(z / 8) for z in np.shape(image)]
    mid_point = [round(z / 2) for z in np.shape(image)]

    roi = image[
        mid_point[0] - step_size[0] : mid_point[0] + step_size[0],
        mid_point[1] - step_size[1] : mid_point[1] + step_size[1],
    ]

    px_mean = np.mean(roi)
    px_std = np.std(roi)

    snr = px_mean / px_std

    return snr


def clahe(im, tiles=(16, 16), clip_lim=40):
    cl = cv2.createCLAHE(clipLimit=clip_lim, tileGridSize=tiles)
    im = im.astype(np.uint16)
    im = cl.apply(im)
    im = normalize_contrast(im)
    return im


def _enhance(im, im_filter, sigmas):
    """Apply one of the supported vessel enhancement filters."""
    if im_filter == "jerman":
        return jerman(
            im,
            sigmas=sigmas,
            tau=0.75,
            brightondark=True,
            cval=0,
            mode="reflect",
        )
    filters = {"meijering": meijering, "sato": sato, "frangi": frangi}
    if im_filter not in filters:
        raise ValueError(
            "filter must be one of meijering, sato, frangi or jerman, got %r"
            % (im_filter,)
        )
    sigmas = list(sigmas)
    # frangi derives gamma from the first sigma, so its sigmas depend on each
    # other; meijering and sato take a pixel-wise max of independent sigmas.
    if im_filter == "frangi" or len(sigmas) < 2:
        return filters[im_filter](
            im, sigmas=sigmas, mode="reflect", black_ridges=False
        )

    def one_sigma(sigma):
        return filters[im_filter](
            im, sigmas=[sigma], mode="reflect", black_ridges=False
        )

    per_sigma = _parallel_map(one_sigma, sigmas, _filter_workers(im))
    filtered_max = np.zeros_like(per_sigma[0])
    for vals in per_sigma:
        filtered_max = np.maximum(filtered_max, vals)
    return filtered_max


def show_im(im):
    plt.figure()
    plt.imshow(im, cmap="gray")
