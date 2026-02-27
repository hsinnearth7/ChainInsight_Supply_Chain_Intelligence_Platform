"""Watchdog file monitor — watches data/raw/ for new CSV files and triggers pipelines."""

import logging
import threading
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class CSVHandler(FileSystemEventHandler):
    """Handles new CSV file creation with debounce for Windows multi-event behavior."""

    def __init__(self, callback: Callable[[str], None], debounce_seconds: float = 2.0):
        super().__init__()
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def on_created(self, event):
        if event.is_directory:
            return
        if not event.src_path.lower().endswith(".csv"):
            return
        self._debounce(event.src_path)

    def on_modified(self, event):
        if event.is_directory:
            return
        if not event.src_path.lower().endswith(".csv"):
            return
        self._debounce(event.src_path)

    def _debounce(self, path: str):
        """Debounce repeated events for the same file (Windows fires multiple events)."""
        with self._lock:
            if path in self._timers:
                self._timers[path].cancel()
            timer = threading.Timer(self.debounce_seconds, self._fire, args=[path])
            timer.daemon = True
            timer.start()
            self._timers[path] = timer

    def _fire(self, path: str):
        with self._lock:
            self._timers.pop(path, None)
        logger.info("Watchdog detected new CSV: %s", path)
        try:
            self.callback(path)
        except Exception:
            logger.exception("Watchdog callback failed for %s", path)


def start_watcher(
    watch_dir: str,
    callback: Callable[[str], None],
    debounce_seconds: float = 2.0,
) -> Observer:
    """Start watching a directory for new CSV files.

    Args:
        watch_dir: Directory path to watch.
        callback: Function called with the CSV file path when a new file is detected.
        debounce_seconds: Seconds to wait before triggering (handles Windows multi-events).

    Returns:
        The Observer instance (daemon thread). Call observer.stop() to shut down.
    """
    watch_path = Path(watch_dir)
    watch_path.mkdir(parents=True, exist_ok=True)

    handler = CSVHandler(callback=callback, debounce_seconds=debounce_seconds)
    observer = Observer()
    observer.schedule(handler, str(watch_path), recursive=False)
    observer.daemon = True
    observer.start()
    logger.info("Watchdog started — monitoring %s for CSV files", watch_path)
    return observer
