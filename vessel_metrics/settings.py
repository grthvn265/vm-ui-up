"""Segmentation settings: defaults and parsing of the form entries."""

from .segmentation import segment_image


def make_segmentation_settings(settings_list):
    """Turn the 8 values from the settings form into a settings dict.

    Blank or invalid entries fall back to DEFAULT_SEGMENTATION_SETTINGS.
    """
    defaults = DEFAULT_SEGMENTATION_SETTINGS
    raw = list(settings_list or [])
    raw += [""] * (len(SETTINGS_ORDER) - len(raw))
    entered = {}
    for key, value in zip(SETTINGS_ORDER, raw):
        entered[key] = defaults[key] if value is None or value == "" else value

    settings = {}
    chosen = str(entered["filter"]).strip().lower()
    settings["filter"] = (
        chosen
        if chosen in ("meijering", "jerman", "frangi", "sato")
        else defaults["filter"]
    )
    settings["threshold"] = _parse_int(
        entered["threshold"], defaults["threshold"], 0, 255
    )
    settings["sigma1"] = _parse_sigma(entered["sigma1"], defaults["sigma1"])
    settings["sigma2"] = _parse_sigma(entered["sigma2"], defaults["sigma2"])
    settings["hole size"] = _parse_int(
        entered["hole size"], defaults["hole size"], 1
    )
    settings["ditzle size"] = _parse_int(
        entered["ditzle size"], defaults["ditzle size"], 1
    )
    settings["preprocess"] = _parse_yes_no(
        entered["preprocess"], defaults["preprocess"]
    )
    settings["multi scale"] = _parse_yes_no(
        entered["multi scale"], defaults["multi scale"]
    )
    return settings


def segment_with_settings(im, settings):
    return segment_image(
        im,
        im_filter=settings["filter"],
        sigma1=settings["sigma1"],
        sigma2=settings["sigma2"],
        hole_size=settings["hole size"],
        ditzle_size=settings["ditzle size"],
        thresh=settings["threshold"],
        preprocess=settings["preprocess"],
        multi_scale=settings["multi scale"],
    )


DEFAULT_SEGMENTATION_SETTINGS = {
    "filter": "meijering",
    "threshold": 40,
    "sigma1": range(1, 8, 1),
    "sigma2": range(10, 20, 5),
    "hole size": 50,
    "ditzle size": 500,
    "preprocess": True,
    "multi scale": False,
}


SETTINGS_ORDER = [
    "filter",
    "threshold",
    "sigma1",
    "sigma2",
    "hole size",
    "ditzle size",
    "preprocess",
    "multi scale",
]


def _parse_sigma(value, default):
    if isinstance(value, range):
        return value
    try:
        start, stop, step = (
            int(x) for x in str(value).replace(",", " ").split()
        )
        if start < 1 or step < 1 or stop <= start:
            raise ValueError
        return range(start, stop, step)
    except ValueError:
        print(
            "invalid sigma input %r, sigma is entered as start stop step, "
            "i.e. '1 8 1'; using %r" % (value, default)
        )
        return default


def _parse_yes_no(value, default):
    if isinstance(value, bool):
        return value
    answer = str(value).strip().lower()
    if answer in ("yes", "y", "true", "1"):
        return True
    if answer in ("no", "n", "false", "0"):
        return False
    return default


def _parse_int(value, default, minimum=0, maximum=None):
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return default
    if number < minimum or (maximum is not None and number > maximum):
        return default
    return number


def _sigma_text(value):
    if isinstance(value, range):
        return "%d %d %d" % (value.start, value.stop, value.step)
    return str(value)


def settings_to_fields(settings):
    """Inverse of make_segmentation_settings, for filling the form."""
    s = dict(DEFAULT_SEGMENTATION_SETTINGS)
    s.update(settings)
    return [
        str(s["filter"]),
        str(s["threshold"]),
        _sigma_text(s["sigma1"]),
        _sigma_text(s["sigma2"]),
        str(s["hole size"]),
        str(s["ditzle size"]),
        "yes" if s["preprocess"] else "no",
        "yes" if s["multi scale"] else "no",
    ]
