"""Graphical user interface (tkinter, ships with python)."""
import sys
import traceback

from .analysis import UserError, run_analysis
from .fileio import load_settings
from .logs import log
from .settings import DEFAULT_SEGMENTATION_SETTINGS, settings_to_fields


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
    "Preprocess: apply the vessel metrics preprocessing.\n\n"
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
    pre_var = tk.BooleanVar(value=defaults[6] == "yes")
    multi_var = tk.BooleanVar(value=defaults[7] == "yes")
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

    czi = section(left, "2. CZI files only")
    czi.pack(fill="x", pady=(0, 8))
    label(czi, "Channel (starts at 0):").grid(row=0, column=0, sticky="w")
    entry(czi, channel_var, 6).grid(row=0, column=1, padx=6)
    label(czi, "Slices per projection:").grid(row=1, column=0, sticky="w", pady=(6, 0))
    entry(czi, slices_var, 6).grid(row=1, column=1, padx=6, pady=(6, 0))

    par = section(left, "3. Parameters to measure")
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
    check(seg, "Preprocess", pre_var).grid(row=6, column=0, sticky="w", pady=(6, 0))
    check(seg, "Multi scale (uses sigma 2)", multi_var).grid(row=6, column=1, sticky="w", pady=(6, 0))

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
            pre_var.set(f[6] == "yes")
            multi_var.set(f[7] == "yes")
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
        return {
            "mode": mode.get(), "input": in_var.get(), "output": out_var.get(),
            "fields": [filter_var.get(), field_vars["threshold"].get(),
                       field_vars["sigma1"].get(), field_vars["sigma2"].get(),
                       field_vars["hole"].get(), field_vars["ditzle"].get(),
                       "yes" if pre_var.get() else "no",
                       "yes" if multi_var.get() else "no"],
            "params": [p for p, v in param_vars.items() if v.get()],
            "channel": channel_var.get(), "slices": slices_var.get(),
            "save_settings": save_var.get(), "roi": roi_var.get(),
        }

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
            run_btn.config(state="normal")

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
