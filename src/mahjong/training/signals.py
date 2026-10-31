"""First signal requests a safe stop; the second signal exits immediately."""

from contextlib import contextmanager
import os
import signal

from mahjong.contracts import StopToken


class SignalStopToken:
    def __init__(self) -> None:
        self._signal = None

    def request(self, signal_number: int) -> None:
        if self._signal is None:
            self._signal = int(signal_number)

    def requested(self) -> bool:
        return self._signal is not None

    def signal_number(self) -> int | None:
        return self._signal


@contextmanager
def signal_handlers(stop: StopToken):
    previous = {}

    def handler(number, frame):
        if stop.requested():
            os._exit(128 + number)
        stop.request(number)

    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            previous[number] = signal.getsignal(number)
            signal.signal(number, handler)
        yield stop
    finally:
        for number, old in previous.items():
            signal.signal(number, old)
