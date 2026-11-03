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
):
    if preprocess:
        with stage("preprocessing (background subtraction, median blur)"):
            im = preprocess_seg(im)

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

    with stage("filling holes < %s px and removing objects < %s px" % (hole_size, ditzle_size)):
        label = fill_holes(label, hole_size)
        label = remove_small_objects(label, ditzle_size)
    return label


def preprocess_seg(
    image,
    radius=50,
    median_size=7,
    upper_lim=255,
    lower_lim=0,
    bright_background=False,
):
    image = normalize_contrast(image)

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
