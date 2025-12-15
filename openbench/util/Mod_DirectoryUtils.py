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


def reset_directory(path: str, timeout: float = 60.0, max_retries: int = 5, retry_delay: float = 2.0) -> None:
    """
    Remove and recreate the directory after ensuring it is stable.

    Args:
        path: Directory path to reset
        timeout: Timeout for waiting directory to be stable (seconds)
        max_retries: Maximum number of retry attempts for removal
        retry_delay: Delay between retry attempts (seconds)
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        return

    # Wait for directory to be stable
    wait_for_directory_stable(path, timeout=timeout)

    import shutil
    import gc

    # Try to remove the directory with retry mechanism
    for attempt in range(max_retries):
        try:
            # Force garbage collection to release any file handles
            gc.collect()

            # Try to remove with ignore_errors first
            shutil.rmtree(path, ignore_errors=False)

            # Successfully removed, recreate and return
            os.makedirs(path, exist_ok=True)
            logging.info(f"Successfully reset directory: {path}")
            return

        except PermissionError as e:
            # File is still in use
            if attempt < max_retries - 1:
                logging.warning(f"Permission denied when resetting {path} (attempt {attempt + 1}/{max_retries}): {e}")
                logging.info(f"Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
                continue
            else:
                # Final attempt failed - try to clean individual files
                logging.error(f"Failed to reset directory after {max_retries} attempts: {path}")
                logging.warning("Attempting to remove files individually...")

                try:
                    removed_count = 0
                    failed_files = []

                    for root, dirs, files in os.walk(path, topdown=False):
                        for name in files:
                            file_path = os.path.join(root, name)
                            try:
                                os.remove(file_path)
                                removed_count += 1
                            except Exception as file_err:
                                failed_files.append(file_path)
                                logging.debug(f"Failed to remove {file_path}: {file_err}")

                        for name in dirs:
                            dir_path = os.path.join(root, name)
                            try:
                                os.rmdir(dir_path)
                            except Exception as dir_err:
                                logging.debug(f"Failed to remove directory {dir_path}: {dir_err}")

                    if failed_files:
                        logging.warning(f"Could not remove {len(failed_files)} files. They may still be in use.")
                        logging.warning(f"Removed {removed_count} files successfully.")
                        logging.info("Continuing execution - scratch directory will be partially cleaned.")
                    else:
                        logging.info(f"Successfully removed all {removed_count} files individually.")
                        # Try to recreate the directory
                        try:
                            if not os.path.exists(path):
                                os.makedirs(path, exist_ok=True)
                        except Exception:
                            pass

                    return

                except Exception as cleanup_err:
                    logging.error(f"Individual file cleanup also failed: {cleanup_err}")
                    logging.warning("Continuing execution - scratch directory cleanup incomplete.")
                    return

        except Exception as e:
            # Other unexpected errors
            logging.error(f"Unexpected error resetting directory {path}: {e}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                continue
            else:
                logging.error("All retry attempts exhausted. Continuing without full cleanup.")
                return
