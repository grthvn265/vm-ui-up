"""Progress log printed to the terminal."""
import threading
import time
from contextlib import contextmanager
from datetime import datetime


VERBOSE = True  # set to False to silence the progress log


HEARTBEAT_SECONDS = 15  # "still working" message interval for slow stages


def log(msg=""):
    """Print a timestamped line to the terminal right away."""
    if VERBOSE:
        stamp = datetime.now().strftime("%H:%M:%S")
        print("[%s] %s" % (stamp, msg), flush=True)


@contextmanager
def stage(name):
    """Log the start / end / duration of a step, with a heartbeat if it is slow."""
    start = time.time()
    log("  > %s ..." % name)
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(HEARTBEAT_SECONDS):
            log(
                "    ... still working on '%s' (%.0fs elapsed)"
                % (name, time.time() - start)
            )

    beat = threading.Thread(target=heartbeat, daemon=True)
    beat.start()
    try:
        yield
    except BaseException:
        stop.set()
        log("  ! %s FAILED after %.1fs" % (name, time.time() - start))
        raise
    else:
        stop.set()
        log("  < %s done (%.1fs)" % (name, time.time() - start))
