import logging
import os
import time
from typing import List, Tuple


def _snapshot_directory(path: str) -> List[Tuple[str, int, float]]:
    snapshot = []
    for root, _, files in os.walk(path):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                stat_result = os.stat(file_path)
                snapshot.append((file_path, stat_result.st_size, stat_result.st_mtime))
            except FileNotFoundError:
                continue
    snapshot.sort()
    return snapshot


def wait_for_directory_stable(path: str, timeout: float = 60.0, check_interval: float = 0.5,
                              required_matches: int = 3) -> bool:
    """
    Wait until the directory contents (size + mtime) remain stable.

    Returns:
        True if stability was detected, False if timed out.
    """
    if not os.path.exists(path):
        return True

    start_time = time.time()
    last_snapshot = None
    stable_checks = 0

    while time.time() - start_time < timeout:
        snapshot = _snapshot_directory(path)
        if snapshot == last_snapshot:
            stable_checks += 1
            if stable_checks >= required_matches or not snapshot:
                return True
        else:
            stable_checks = 0

        last_snapshot = snapshot

        if not snapshot:
            return True

        time.sleep(check_interval)

    logging.warning(f"Directory {path} did not stabilize within {timeout} seconds, proceeding with cleanup.")
    return False


def reset_directory(path: str, timeout: float = 60.0) -> None:
    """
    Remove and recreate the directory after ensuring it is stable.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        return

    wait_for_directory_stable(path, timeout=timeout)
    import shutil
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
