"""Reading images and .czi volumes, saving settings."""
import os
import pickle

import cv2
import numpy as np

from .logs import log
from .segmentation import normalize_contrast


def reslice_image(image, thickness):
    num_slices = np.shape(image)[0]
    out_slices = np.ceil(num_slices / thickness).astype(np.uint16)
    output = np.zeros([out_slices, np.shape(image)[1], np.shape(image)[2]])
    count = 0
    for i in range(0, num_slices, thickness):
        if i + thickness < num_slices:
            im_chunk = image[i : i + thickness, :, :]
        else:
            im_chunk = image[i:, :, :]

        output[count, :, :] = np.max(im_chunk, axis=0)
        count += 1
    return output


def generate_file_list(file_list):
    accepted_file_types = ["czi", "png", "tif", "tiff", "jpg", "jpeg", "bmp"]
    reduced_file_list = []
    file_types = []
    for file in sorted(file_list):
        if "." in file:
            file_type = file.rsplit(".", 1)[-1].lower()
            if file_type in accepted_file_types:
                reduced_file_list.append(file)
                file_types.append(file_type)
    return reduced_file_list, file_types


def save_settings(settings, path):
    with open(path, "wb") as filehandle:
        pickle.dump(settings, filehandle)
    return


def load_settings(path):
    with open(path, "rb") as filehandle:
        all_settings = pickle.load(filehandle)
    return all_settings


def czi_projection(volume, axis):
    projection = np.max(volume, axis=axis)
    return projection


def sliding_window(volume, thickness):
    num_slices = np.shape(volume)[0]
    out_slices = num_slices - thickness
    output = np.zeros([out_slices, np.shape(volume)[1], np.shape(volume)[2]])
    for i in range(0, out_slices):
        im_chunk = volume[i : i + thickness, :, :]
        output[i, :, :] = np.max(im_chunk, axis=0)
    return output


def _read_czi_array(path):
    """Read a .czi with whichever reader is installed; returns (array, [z, y, x] pixel sizes)."""
    reader_error = None
    try:
        from aicsimageio import AICSImage
        from aicsimageio.readers import CziReader

        img = AICSImage(path, reader=CziReader)
        sizes = img.physical_pixel_sizes
        return np.squeeze(img.data), [sizes[0], sizes[1], sizes[2]]
    except ImportError as err:
        reader_error = err
    try:
        import czifile

        return np.squeeze(czifile.imread(path)), [None, None, None]
    except ImportError:
        raise ImportError(
            "Reading .czi files needs the 'aicsimageio[czi]' or 'czifile' "
            "package (pip install czifile). Original error: %s" % reader_error
        )


def preprocess_czi(input_directory, file_name, channel=0):
    """Load a .czi file, pick one channel and scale the volume to 0-255.

    Returns (volume with shape [z, y, x], [z, y, x] physical pixel sizes).
    """
    path = os.path.join(input_directory, file_name) if file_name else input_directory
    data, out_dims = _read_czi_array(path)
    if data.ndim == 2:
        data = data[np.newaxis]
    if data.ndim == 4:
        if not 0 <= channel < data.shape[0]:
            raise ValueError(
                "channel %d requested but %s only has channels 0-%d"
                % (channel, os.path.basename(path), data.shape[0] - 1)
            )
        data = data[channel]
    elif data.ndim != 3:
        raise ValueError(
            "Unsupported .czi layout %s in %s"
            % (data.shape, os.path.basename(path))
        )
    return normalize_contrast(data), out_dims


def load_images(files, czi_options, progress=log):
    """Read every file into memory; returns (images, names)."""
    images, names = [], []
    for path in files:
        base, ext = os.path.splitext(os.path.basename(path))
        if ext.lower() == ".czi":
            channel, thickness = czi_options
            progress("reading %s" % os.path.basename(path))
            volume, _ = preprocess_czi(path, "", channel=channel)
            thickness = min(thickness, volume.shape[0])
            reslice = reslice_image(volume, thickness)
            for i in range(reslice.shape[0]):
                images.append(reslice[i])
                names.append("%s_slice%d" % (base, i))
        else:
            im = cv2.imread(path, 0)
            if im is None:
                progress("skipping %s (could not be read as an image)" % path)
                continue
            images.append(im)
            names.append(base)
            progress("loaded %s (%d x %d px)" % (base, im.shape[1], im.shape[0]))
    return images, names
