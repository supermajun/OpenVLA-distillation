"""Step-boundary interruption: finish the update, then save a resumable checkpoint."""
import signal


class StopRequest:
    def __init__(self):
        self.requested = False
        self.signal = None
        self.previous = {}

    def receive(self, signum, frame):
        self.requested = True
        self.signal = signum

    def __enter__(self):
        for sig in (signal.SIGTERM, signal.SIGINT, getattr(signal, "SIGUSR1", None)):
            if sig is not None:
                self.previous[sig] = signal.signal(sig, self.receive)
        return self

    def __exit__(self, *args):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)
