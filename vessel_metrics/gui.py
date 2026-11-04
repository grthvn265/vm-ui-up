"""Graphical user interface (tkinter, ships with python)."""
import base64
import os
import sys
import traceback

import cv2

from .analysis import UserError, check_image, labelled_panels, run_analysis
from .fileio import generate_file_list, load_settings
from .logs import log
from .segmentation import COLOUR_CHANNEL_CHOICES, IMAGE_TYPE_CHOICES, IMAGE_TYPE_LABELS
from .settings import (
    DEFAULT_SEGMENTATION_SETTINGS,
    _parse_int,
    make_segmentation_settings,
    settings_to_fields,
)


APP_TITLE = "Vessel Metrics"


IMAGE_TYPES = ["png", "tif", "tiff", "jpg", "jpeg", "bmp"]


PARAMETER_CHOICES = [
    "vessel density",
    "branchpoint density",
    "network length",
    "tortuosity",
    "segment length",
    "diameter",
]


FILTER_CHOICES = ["meijering", "sato", "frangi", "jerman"]


PARAMETER_HELP = (
    "Image type (required, choose before running): Vessel Metrics finds "
    "bright vessels on a dark background.\n"
    "  Normal photo: dark vessels on a light background. The photo is "
    "inverted first, before any other preprocessing.\n"
    "  Inverted / fluorescence: bright vessels already, used as it is.\n"
    "  Pre-processed: bright vessels with the background already removed; "
    "the Vessel Metrics preprocessing is skipped.\n"
    "Check image... labels the selected image (or the first image of the "
    "folder) and shows every processing step, so you can confirm the choice "
    "and the settings before running. Each result folder also gets a "
    "labelled image_check.png.\n\n"
    "Colour channel: what a colour photo is reduced to. Green usually gives "
    "the best vessel contrast in colour photos (blood absorbs green light).\n\n"
    "Local contrast boost (CLAHE): brings out faint, small vessels in "
    "unevenly lit images.\n\n"
    "Remove dark specks: fills dust or debris specks up to this radius in "
    "pixels so they do not break vessels apart (0 = off, try 4-6 for photos). "
    "Keep it below half the gap between neighbouring vessels.\n\n"
    "Mask the area outside the sample: ignores the black (or white) "
    "surround of a cropped photo so its edge is not detected as a vessel.\n\n"
    "Filter: vessel enhancement filter. Meijering is the default, jerman is "
    "recommended for dim images.\n\n"
    "Threshold: between 0 and 255, default 40. Vessel metrics performs a "
    "contrast stretch prior to thresholding.\n\n"
    "Sigma 1 / Sigma 2: size of vessels the filter enhances, entered as "
    "'start stop step'. Choose an even range up to 80% of your largest vessel, "
    "e.g. '1 8 1' for vessels 1-10 pixels wide. Sigma 2 is only used with "
    "multi scale, for images with both very large and very small vessels.\n\n"
    "Hole size: area in pixels of holes in solid structures that are filled in "
    "post processing.\n\n"
    "Ditzle size: binary objects in the final segmentation smaller than this "
    "(in pixels) are removed.\n\n"
    "Multi scale: also use the sigma 2 range.\n\n"
    "CZI files: channel is an integer starting at 0; slices per projection "
    "sets how thick each z projection is.\n\n"
    "Refer to github.com/mcgarrysd/vessel_metrics for the readme."
)


# Explicit colours everywhere: native widgets inherit the system theme, which
# on a dark-mode Mac gives white text on a white window.
BG, FG, MUTED = "#f2f2f2", "#1a1a1a", "#555555"


FIELD_BG, FIELD_FG = "#ffffff", "#000000"


LOG_BG, LOG_FG = "#1e1e1e", "#e6e6e6"


WARN_FG = "#b00020"


NO_TYPE = "unset"  # image type radio value meaning "not chosen yet"


def launch_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.title(APP_TITLE)
    root.configure(bg=BG)
    font = ("Helvetica", 13)
    bold = ("Helvetica", 13, "bold")

    def frame(parent, **kw):
        return tk.Frame(parent, bg=BG, **kw)

    def section(parent, text):
        return tk.LabelFrame(
            parent, text=" %s " % text, bg=BG, fg=FG, font=bold,
            padx=10, pady=8, highlightthickness=0, bd=1, relief="groove",
        )

    def label(parent, text, muted=False):
        return tk.Label(
            parent, text=text, bg=BG, fg=MUTED if muted else FG, font=font,
            anchor="w", justify="left",
        )

    def entry(parent, var, width=10):
        return tk.Entry(
            parent, textvariable=var, width=width, font=font, bg=FIELD_BG,
            fg=FIELD_FG, insertbackground=FIELD_FG, relief="solid", bd=1,
            highlightthickness=0, disabledbackground="#dddddd",
            disabledforeground="#777777",
        )

    def check(parent, text, var):
        return tk.Checkbutton(
            parent, text=text, variable=var, bg=BG, fg=FG, font=font,
            activebackground=BG, activeforeground=FG, selectcolor=FIELD_BG,
            highlightthickness=0, anchor="w", bd=0,
        )

    def radio(parent, text, var, value, command=None):
        return tk.Radiobutton(
            parent, text=text, variable=var, value=value, command=command,
            bg=BG, fg=FG, font=font, activebackground=BG,
            activeforeground=FG, selectcolor=FIELD_BG, highlightthickness=0,
            anchor="w", bd=0,
        )

    def button(parent, text, command):
        return tk.Button(parent, text=text, command=command, font=font,
                         highlightbackground=BG)

    # ---- form variables ----
    mode = tk.StringVar(value="single")
    in_var, out_var = tk.StringVar(), tk.StringVar()
    defaults = settings_to_fields(DEFAULT_SEGMENTATION_SETTINGS)
    filter_var = tk.StringVar(value=defaults[0])
    field_vars = {k: tk.StringVar(value=v) for k, v in zip(
        ["threshold", "sigma1", "sigma2", "hole", "ditzle"], defaults[1:6])}
    multi_var = tk.BooleanVar(value=defaults[7] == "yes")
    type_var = tk.StringVar(value=NO_TYPE)  # no default, the user must choose
    colour_var = tk.StringVar(value=defaults[9])
    clahe_var = tk.BooleanVar(value=defaults[10] == "yes")
    speck_var = tk.StringVar(value=defaults[11])
    mask_var = tk.BooleanVar(value=defaults[12] == "yes")
    param_vars = {p: tk.BooleanVar(value=True) for p in PARAMETER_CHOICES}
    channel_var, slices_var = tk.StringVar(value="0"), tk.StringVar(value="20")
    save_var, roi_var = tk.BooleanVar(value=True), tk.BooleanVar(value=False)

    # ---- layout ----
    outer = frame(root)
    outer.pack(fill="both", expand=True, padx=14, pady=12)
    left, right = frame(outer), frame(outer)
    left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
    right.grid(row=0, column=1, sticky="nsew")

    src = section(left, "1. Input and output")
    src.pack(fill="x", pady=(0, 8))
    radio(src, "Single image", mode, "single").grid(row=0, column=0, sticky="w")
    radio(src, "Directory (batch)", mode, "directory").grid(row=0, column=1, sticky="w")

    def browse_in():
        if mode.get() == "single":
            p = filedialog.askopenfilename(
                title="Select the image to analyze",
                filetypes=[("Microscopy images",
                            ["*.czi"] + ["*." + t for t in IMAGE_TYPES]),
                           ("All files", "*")])
        else:
            p = filedialog.askdirectory(title="Select the directory to analyze")
        if p:
            in_var.set(p)

    def browse_out():
        p = filedialog.askdirectory(title="Select output directory")
        if p:
            out_var.set(p)

    label(src, "Image or folder:").grid(row=1, column=0, sticky="w", pady=(8, 0))
    entry(src, in_var, 34).grid(row=2, column=0, columnspan=2, sticky="we")
    button(src, "Browse...", browse_in).grid(row=2, column=2, padx=(6, 0))
    label(src, "Output folder:").grid(row=3, column=0, sticky="w", pady=(8, 0))
    entry(src, out_var, 34).grid(row=4, column=0, columnspan=2, sticky="we")
    button(src, "Browse...", browse_out).grid(row=4, column=2, padx=(6, 0))

    prep = section(left, "2. Image preprocessing (choose before running)")
    prep.pack(fill="x", pady=(0, 8))
    label(prep, "Image type (required):").grid(row=0, column=0, columnspan=2, sticky="w")
    for i, kind in enumerate(IMAGE_TYPE_CHOICES, start=1):
        radio(prep, IMAGE_TYPE_LABELS[kind], type_var, kind).grid(
            row=i, column=0, columnspan=2, sticky="w", padx=(10, 0))
    type_hint = tk.Label(prep, bg=BG, fg=WARN_FG, font=font, anchor="w", justify="left")
    type_hint.grid(row=4, column=0, columnspan=2, sticky="w", pady=(2, 0))
    label(prep, "Colour channel:").grid(row=5, column=0, sticky="w", pady=(6, 0))
    crow = frame(prep)
    crow.grid(row=5, column=1, sticky="w", pady=(6, 0))
    for i, name in enumerate(COLOUR_CHANNEL_CHOICES):
        radio(crow, name, colour_var, name).grid(row=0, column=i, sticky="w", padx=(0, 8))
    clahe_box = check(prep, "Local contrast boost (CLAHE)", clahe_var)
    clahe_box.grid(row=6, column=0, columnspan=2, sticky="w", pady=(6, 0))
    label(prep, "Remove dark specks, radius px (0 = off):").grid(row=7, column=0, sticky="w", pady=(5, 0))
    speck_entry = entry(prep, speck_var, 6)
    speck_entry.grid(row=7, column=1, sticky="w", pady=(5, 0))
    check(prep, "Mask the area outside the sample (dark or white border)", mask_var).grid(
        row=8, column=0, columnspan=2, sticky="w", pady=(5, 0))
    check_btn = button(prep, "Check image...", lambda: on_check())
    check_btn.grid(row=9, column=0, sticky="w", pady=(8, 0))
    label(prep, "labels the image and previews each step", muted=True).grid(
        row=9, column=1, sticky="w", pady=(8, 0))

    czi = section(left, "3. CZI files only")
    czi.pack(fill="x", pady=(0, 8))
    label(czi, "Channel (starts at 0):").grid(row=0, column=0, sticky="w")
    entry(czi, channel_var, 6).grid(row=0, column=1, padx=6)
    label(czi, "Slices per projection:").grid(row=1, column=0, sticky="w", pady=(6, 0))
    entry(czi, slices_var, 6).grid(row=1, column=1, padx=6, pady=(6, 0))

    par = section(left, "4. Parameters to measure")
    par.pack(fill="x")
    for i, p in enumerate(PARAMETER_CHOICES):
        check(par, p, param_vars[p]).grid(row=i // 2, column=i % 2, sticky="w", padx=(0, 14))

    seg = section(right, "Segmentation settings")
    seg.pack(fill="x", pady=(0, 8))
    label(seg, "Filter").grid(row=0, column=0, sticky="w")
    frow = frame(seg)
    frow.grid(row=0, column=1, sticky="w", padx=6)
    for i, name in enumerate(FILTER_CHOICES):
        radio(frow, name, filter_var, name).grid(row=i // 2, column=i % 2, sticky="w", padx=(0, 10))
    rows = [("Threshold (0-255)", "threshold"), ("Sigma 1 (start stop step)", "sigma1"),
            ("Sigma 2 (start stop step)", "sigma2"), ("Hole size (px)", "hole"),
            ("Ditzle size (px)", "ditzle")]
    for i, (text, key) in enumerate(rows, start=1):
        label(seg, text).grid(row=i, column=0, sticky="w", pady=(5, 0))
        entry(seg, field_vars[key], 12).grid(row=i, column=1, sticky="w", padx=6, pady=(5, 0))
    check(seg, "Multi scale (uses sigma 2)", multi_var).grid(row=6, column=0, columnspan=2, sticky="w", pady=(6, 0))

    def show_help():
        win = tk.Toplevel(root)
        win.title("Parameter explanation")
        win.configure(bg=BG)
        text = tk.Text(win, width=70, height=24, wrap="word", font=font,
                       bg=FIELD_BG, fg=FIELD_FG, relief="solid", bd=1, padx=8, pady=8)
        text.insert("1.0", PARAMETER_HELP)
        text.config(state="disabled")
        text.pack(padx=10, pady=10)
        button(win, "Close", win.destroy).pack(pady=(0, 10))

    def load_saved():
        p = filedialog.askopenfilename(
            title="Select a saved settings file",
            filetypes=[("Settings", "*.data"), ("All files", "*")])
        if not p:
            return
        try:
            saved = load_settings(p)
            if not (isinstance(saved, (list, tuple)) and len(saved) >= 2
                    and isinstance(saved[0], dict)):
                raise ValueError("unexpected contents")
            f = settings_to_fields(saved[0])
            filter_var.set(f[0] if f[0] in FILTER_CHOICES else FILTER_CHOICES[0])
            for key, value in zip(["threshold", "sigma1", "sigma2", "hole", "ditzle"], f[1:6]):
                field_vars[key].set(value)
            multi_var.set(f[7] == "yes")
            # files saved before the image type existed leave it unchosen
            type_var.set(f[8] if f[8] in IMAGE_TYPE_CHOICES else NO_TYPE)
            colour_var.set(f[9])
            clahe_var.set(f[10] == "yes")
            speck_var.set(f[11])
            mask_var.set(f[12] == "yes")
            for name, var in param_vars.items():
                var.set(name in saved[1])
            if len(saved) > 2 and saved[2]:
                channel_var.set(str(saved[2][0]))
                slices_var.set(str(saved[2][1]))
        except Exception as err:
            messagebox.showwarning(
                APP_TITLE, "That file could not be read as a Vessel Metrics "
                "settings file (%s)." % err)

    tools = frame(right)
    tools.pack(fill="x", pady=(0, 8))
    button(tools, "Load saved settings...", load_saved).pack(side="left")
    button(tools, "Explain settings", show_help).pack(side="left", padx=8)

    opts = section(right, "Options")
    opts.pack(fill="x")
    check(opts, "Save these settings as settings.data in the output folder", save_var).pack(anchor="w")
    check(opts, "Do an ROI analysis afterwards (opens an image window)", roi_var).pack(anchor="w")

    log_frame = frame(root)
    log_frame.pack(fill="both", expand=True, padx=14, pady=(0, 12))
    run_btn = button(log_frame, "Run analysis", lambda: on_run())
    run_btn.pack(anchor="w", pady=(0, 6))
    status_box = tk.Text(log_frame, height=4, wrap="word", font=("Courier", 12),
                         bg=LOG_BG, fg=LOG_FG, insertbackground=LOG_FG, relief="solid", bd=1)
    status_box.pack(fill="both", expand=True)

    def status(text):
        status_box.insert("end", text + "\n")
        status_box.see("end")
        root.update_idletasks()

    status("Ready. Choose an image or folder and an output folder, then press Run analysis.")
    status("Detailed progress is printed in the terminal window. This window may look frozen while it works.")

    def collect():
        kind = type_var.get() if type_var.get() in IMAGE_TYPE_CHOICES else ""
        return {
            "mode": mode.get(), "input": in_var.get(), "output": out_var.get(),
            "fields": [filter_var.get(), field_vars["threshold"].get(),
                       field_vars["sigma1"].get(), field_vars["sigma2"].get(),
                       field_vars["hole"].get(), field_vars["ditzle"].get(),
                       "no" if kind == "preprocessed" else "yes",
                       "yes" if multi_var.get() else "no",
                       kind, colour_var.get(),
                       "yes" if clahe_var.get() else "no", speck_var.get(),
                       "yes" if mask_var.get() else "no"],
            "params": [p for p, v in param_vars.items() if v.get()],
            "channel": channel_var.get(), "slices": slices_var.get(),
            "save_settings": save_var.get(), "roi": roi_var.get(),
        }

    def refresh_type(*_):
        """Run needs an image type; pre-processed images skip CLAHE and specks."""
        chosen = type_var.get() in IMAGE_TYPE_CHOICES
        type_hint.config(text="" if chosen else "Choose an image type to enable Run analysis.")
        run_btn.config(state="normal" if chosen else "disabled")
        state = "disabled" if type_var.get() == "preprocessed" else "normal"
        clahe_box.config(state=state)
        speck_entry.config(state=state)

    def check_target():
        path = in_var.get().strip()
        if mode.get() == "directory" and os.path.isdir(path):
            names, _ = generate_file_list(os.listdir(path))
            if not names:
                raise UserError("No supported image files were found in that folder.")
            return os.path.join(path, names[0])
        if not os.path.isfile(path):
            raise UserError("Choose the image (or folder) to check first.")
        return path

    def on_check():
        """The image checker: label the input and preview every step."""
        try:
            path = check_target()
            status("Checking %s ..." % os.path.basename(path))
            root.config(cursor="watch")
            root.update()
            suggestion, reason, steps = check_image(
                path, make_segmentation_settings(collect()["fields"]),
                [_parse_int(channel_var.get(), 0, minimum=0),
                 _parse_int(slices_var.get(), 20, minimum=1)])
        except UserError as err:
            messagebox.showwarning(APP_TITLE, str(err))
            return
        except Exception:
            log("ERROR - the image check stopped. Full traceback:")
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            messagebox.showerror(APP_TITLE, "The image check hit an error. Details are in the terminal.")
            return
        finally:
            root.config(cursor="")
        chosen = type_var.get() if type_var.get() in IMAGE_TYPE_CHOICES else None
        verdict = ("Vessel Metrics labels this image: %s" % IMAGE_TYPE_LABELS[suggestion]
                   if suggestion else "Vessel Metrics could not label this image.")
        status("%s - %s" % (verdict, reason))
        log("image check of %s: %s (%s)" % (path, suggestion or "unclear", reason))

        win = tk.Toplevel(root)
        win.title("Image check - %s" % os.path.basename(path))
        win.configure(bg=BG)
        tk.Label(win, text=verdict, bg=BG, fg=FG, font=bold, anchor="w").pack(fill="x", padx=10, pady=(10, 0))
        label(win, "Reason: %s" % reason, muted=True).pack(fill="x", padx=10)
        if chosen and suggestion and chosen != suggestion:
            tk.Label(win, text="You chose \"%s\" - check that this is right." % IMAGE_TYPE_LABELS[chosen],
                     bg=BG, fg=WARN_FG, font=font, anchor="w").pack(fill="x", padx=10)
        label(win, "Steps below use %s." % ("your choice" if chosen else "the suggested type"),
              muted=True).pack(fill="x", padx=10)
        width = max(160, min(320, (root.winfo_screenwidth() - 80) // len(steps) - 8))
        ok, png = cv2.imencode(".png", labelled_panels(steps, width))
        photo = tk.PhotoImage(data=base64.b64encode(png.tobytes()))
        preview = tk.Label(win, image=photo, bg=BG)
        preview.image = photo  # keep a reference, tkinter does not
        preview.pack(padx=10, pady=8)
        buttons = frame(win)
        buttons.pack(pady=(0, 10))
        if suggestion:
            def use():
                type_var.set(suggestion)
                win.destroy()
            button(buttons, "Use \"%s\"" % suggestion, use).pack(side="left", padx=4)
        button(buttons, "Close", win.destroy).pack(side="left", padx=4)

    def on_run():
        run_btn.config(state="disabled")
        status_box.delete("1.0", "end")
        status("Running - watch the terminal for progress ...")
        try:
            count = run_analysis(
                collect(), confirm=lambda m: messagebox.askyesno(APP_TITLE, m))
            status("Done: %d image(s) processed. Results are in %s" % (count, out_var.get()))
            messagebox.showinfo(
                APP_TITLE, "Analysis complete: %d image(s) processed.\nResults are in:\n%s"
                % (count, out_var.get()))
        except UserError as err:
            log("Cannot start: %s" % err)
            status("Cannot start: %s" % err)
            messagebox.showwarning(APP_TITLE, str(err))
        except Exception:
            log("ERROR - the run stopped. Full traceback:")
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            status("Error - see the terminal for details.")
            messagebox.showerror(APP_TITLE, "Vessel Metrics hit an error. Details are in the terminal.")
        finally:
            refresh_type()

    type_var.trace_add("write", refresh_type)
    refresh_type()
    root.update_idletasks()
    root.minsize(root.winfo_reqwidth(), root.winfo_reqheight())
    root.lift()
    root.mainloop()


def main():
    try:
        import tkinter  # noqa: F401
    except ImportError:
        print(
            "The graphical interface needs tkinter, which is part of python.\n"
            "Use the installer from python.org, or on macOS with Homebrew: "
            "brew install python-tk",
            file=sys.stderr,
        )
        return 1
    import matplotlib

    matplotlib.use("Agg")  # figures are written to disk, never shown
    log("Vessel Metrics started. Progress for every image is printed in this terminal.")
    launch_gui()
    log("Vessel Metrics closed.")
    return 0
