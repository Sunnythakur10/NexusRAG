import threading


class LamportClock:
    """
    Thread-safe Lamport logical clock.

    Rules:
    1. Local event: increment clock.
    2. On receiving a message with timestamp T:
       clock = max(local_clock, T) + 1.
    """

    def __init__(self, initial_time: int = 0):
        self._time = initial_time
        self._lock = threading.Lock()

    def tick(self) -> int:
        """Increment the clock for a local event."""
        with self._lock:
            self._time += 1
            return self._time

    def receive(self, received_time: int) -> int:
        """Update clock when receiving a message."""
        with self._lock:
            self._time = max(self._time, received_time) + 1
            return self._time

    def now(self) -> int:
        """Return the current logical time."""
        with self._lock:
            return self._time