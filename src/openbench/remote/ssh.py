# -*- coding: utf-8 -*-
"""
SSH Manager for remote server connections.

Handles SSH connections, file transfers, and remote command execution.
"""

import os
import posixpath
import re
import shlex
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Tuple, List, Callable, Generator

import paramiko
from paramiko import SSHClient, SSHException
from paramiko.hostkeys import HostKeys

logger = logging.getLogger(__name__)


class SSHConnectionError(Exception):
    """SSH connection error."""

    pass


class HostKeyVerificationError(Exception):
    """Host key verification failed."""

    pass


class InteractiveHostKeyPolicy(paramiko.MissingHostKeyPolicy):
    """Host key policy that prompts user for unknown hosts.

    This policy:
    1. Accepts hosts that are in the known_hosts file
    2. Calls a callback function for unknown hosts to get user confirmation
    3. Optionally saves confirmed hosts to the known_hosts file
    """

    def __init__(
        self,
        known_hosts_path: Optional[str] = None,
        confirm_callback: Optional[Callable[[str, str, str], bool]] = None,
        auto_add: bool = False,
    ):
        """Initialize the host key policy.

        Args:
            known_hosts_path: Path to known_hosts file. Defaults to ~/.openbench_wizard/known_hosts
            confirm_callback: Callback function that receives (hostname, key_type, fingerprint)
                            and returns True to accept the key, False to reject.
                            If None and host is unknown, raises HostKeyVerificationError.
            auto_add: If True, automatically add unknown hosts (less secure, for testing)
        """
        self._known_hosts_path = known_hosts_path or self._get_default_known_hosts_path()
        self._confirm_callback = confirm_callback
        self._auto_add = auto_add
        self._host_keys = HostKeys()
        self._load_known_hosts()

    @staticmethod
    def _get_default_known_hosts_path() -> str:
        """Get the default known_hosts file path."""
        config_dir = Path.home() / ".openbench_wizard"
        config_dir.mkdir(parents=True, exist_ok=True)
        return str(config_dir / "known_hosts")

    def _load_known_hosts(self) -> None:
        """Load known hosts from file."""
        if os.path.exists(self._known_hosts_path):
            try:
                self._host_keys.load(self._known_hosts_path)
                logger.debug(f"Loaded known hosts from {self._known_hosts_path}")
            except Exception as e:
                logger.warning(f"Failed to load known hosts: {e}")

    def _save_host_key(self, hostname: str, key: paramiko.PKey) -> None:
        """Save a host key to the known_hosts file.

        Args:
            hostname: The hostname
            key: The host key to save
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self._known_hosts_path), exist_ok=True)

            # Add the key
            self._host_keys.add(hostname, key.get_name(), key)

            # Save to file
            self._host_keys.save(self._known_hosts_path)

            # Set restrictive permissions (Unix only)
            try:
                os.chmod(self._known_hosts_path, 0o600)
            except (OSError, AttributeError):
                pass  # Windows doesn't support chmod

            logger.info(f"Saved host key for {hostname}")
        except Exception as e:
            logger.warning(f"Failed to save host key: {e}")

    @staticmethod
    def get_fingerprint(key: paramiko.PKey) -> str:
        """Get the fingerprint of a host key.

        Args:
            key: The host key

        Returns:
            SHA256 fingerprint string
        """
        import hashlib
        import base64

        key_bytes = key.asbytes()
        digest = hashlib.sha256(key_bytes).digest()
        fingerprint = base64.b64encode(digest).decode("ascii").rstrip("=")
        return f"SHA256:{fingerprint}"

    def missing_host_key(self, client: SSHClient, hostname: str, key: paramiko.PKey) -> None:
        """Handle a missing host key.

        Args:
            client: The SSH client
            hostname: The hostname
            key: The host's public key

        Raises:
            HostKeyVerificationError: If the key is rejected
        """
        key_type = key.get_name()
        fingerprint = self.get_fingerprint(key)

        # Check if we already have a key for this host
        existing_key = self._host_keys.lookup(hostname)
        if existing_key is not None:
            # Host exists but key is different - potential MITM attack!
            if key_type in existing_key:
                stored_key = existing_key[key_type]
                if stored_key.asbytes() == key.asbytes():
                    # Same key, all good
                    return
                else:
                    # Different key - this is suspicious!
                    raise HostKeyVerificationError(
                        f"WARNING: Remote host identification has changed!\n"
                        f"Host: {hostname}\n"
                        f"Expected fingerprint: {self.get_fingerprint(stored_key)}\n"
                        f"Received fingerprint: {fingerprint}\n"
                        f"This could indicate a man-in-the-middle attack."
                    )

        # Auto-add mode (for testing/development)
        if self._auto_add:
            logger.warning(f"Auto-adding host key for {hostname} (auto_add=True)")
            self._save_host_key(hostname, key)
            return

        # Ask user via callback
        if self._confirm_callback is not None:
            if self._confirm_callback(hostname, key_type, fingerprint):
                self._save_host_key(hostname, key)
                return
            else:
                raise HostKeyVerificationError(f"Host key verification rejected by user for {hostname}")

        # No callback and not auto-add - reject
        raise HostKeyVerificationError(
            f"Unknown host: {hostname}\n"
            f"Key type: {key_type}\n"
            f"Fingerprint: {fingerprint}\n"
            f"No confirmation callback provided."
        )


class SSHManager:
    """Manage SSH connections, file transfer, and remote command execution."""

    def __init__(
        self,
        timeout: int = 30,
        host_key_callback: Optional[Callable[[str, str, str], bool]] = None,
        auto_add_host_keys: bool = False,
    ):
        """Initialize SSH manager.

        Args:
            timeout: Connection timeout in seconds
            host_key_callback: Callback function for unknown host keys.
                             Receives (hostname, key_type, fingerprint) and returns
                             True to accept, False to reject. If None and auto_add_host_keys
                             is False, unknown hosts will be rejected.
            auto_add_host_keys: If True, automatically accept unknown host keys.
                              WARNING: This is less secure and should only be used
                              for testing or in trusted networks.
        """
        self._client: Optional[SSHClient] = None
        self._sftp: Optional[paramiko.SFTPClient] = None
        self._jump_sftp: Optional[paramiko.SFTPClient] = None
        self._timeout = timeout
        self._host = ""
        self._user = ""
        self._port = 22
        self._host_key_callback = host_key_callback
        self._auto_add_host_keys = auto_add_host_keys
        # Multi-hop connection attributes
        self._jump_client: Optional[SSHClient] = None
        self._jump_channel = None
        self._last_detection_errors: list[str] = []
        # Reentrant lock guarding mutations of self._client / self._sftp /
        # self._jump_* and short reads of the active client. Held only across
        # state changes and channel acquisition, NOT across long-running
        # channel I/O — that would block stop()/kill paths that need to open
        # a second channel on the same SSHManager from another thread.
        self._state_lock = threading.RLock()

    @property
    def last_detection_errors(self) -> tuple[str, ...]:
        """Errors suppressed during the last interpreter/conda discovery call."""
        return tuple(self._last_detection_errors)

    def _record_detection_error(self, message: str, exc: Exception) -> None:
        detail = f"{message}: {exc}"
        self._last_detection_errors.append(detail)
        logger.debug(detail)

    @property
    def is_connected(self) -> bool:
        """Check if connected to server."""
        if self._client is None:
            return False
        try:
            transport = self._client.get_transport()
            return transport is not None and transport.is_active()
        except Exception:
            return False

    def _parse_host_string(self, host_string: str) -> Tuple[Optional[str], str, int]:
        """Parse host string in format [user@]host[:port].

        Args:
            host_string: Host string like "user@192.168.1.100:22"

        Returns:
            Tuple of (user, host, port)
        """
        user = None
        port = 22

        # Extract user if present
        if "@" in host_string:
            user, host_string = host_string.split("@", 1)

        # Extract port if present
        if ":" in host_string:
            host, port_str = host_string.rsplit(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                host = host_string
        else:
            host = host_string

        return user, host, port

    def connect(
        self,
        host_string: str,
        password: Optional[str] = None,
        key_file: Optional[str] = None,
        passphrase: Optional[str] = None,
    ) -> None:
        """Connect to SSH server.

        Args:
            host_string: Host in format [user@]host[:port]
            password: Password for authentication
            key_file: Path to SSH private key file
            passphrase: Passphrase for encrypted key file

        Raises:
            SSHConnectionError: If connection fails
        """
        user, host, port = self._parse_host_string(host_string)

        if user is None:
            raise SSHConnectionError("Username is required (format: user@host)")

        if self._jump_client is not None or self._jump_channel is not None or self._jump_sftp is not None:
            self.disconnect_jump()

        # Close any prior client+sftp before reassigning, otherwise the old
        # socket/thread leaks and a stale self._sftp keeps a dead client alive.
        if self._client is not None or self._sftp is not None:
            self._safe_close_client()

        self._client = paramiko.SSHClient()

        # Use secure host key policy
        policy = InteractiveHostKeyPolicy(confirm_callback=self._host_key_callback, auto_add=self._auto_add_host_keys)
        self._client.set_missing_host_key_policy(policy)

        try:
            self._client.connect(
                hostname=host,
                port=port,
                username=user,
                password=password,
                key_filename=key_file,
                passphrase=passphrase,
                timeout=self._timeout,
                allow_agent=False,
                look_for_keys=False,
            )
            self._host = host
            self._user = user
            self._port = port
        except SSHException as e:
            self._safe_close_client()
            raise SSHConnectionError(f"SSH connection failed: {e}") from e
        except Exception as e:
            self._safe_close_client()
            raise SSHConnectionError(f"Connection failed: {e}") from e

    def _safe_close_client(self) -> None:
        """Best-effort close of self._client (and self._sftp) and reset to None.

        Used by `connect()` failure paths and re-connect to avoid leaking the
        underlying socket when paramiko has instantiated SSHClient but
        `connect()` raised before it was fully wired up, or when callers
        reuse the manager without `disconnect()` first.

        The SFTP channel is owned by the client transport, so close it here
        too — leaving a stale self._sftp pointing at a dead client made
        subsequent operations raise SSHException from inside paramiko.
        """
        with self._state_lock:
            if self._sftp is not None:
                try:
                    self._sftp.close()
                except Exception:
                    pass
                self._sftp = None
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None

    def disconnect(self) -> None:
        """Disconnect from server.

        Cleans up both main and jump connections if present.
        """
        with self._state_lock:
            # First disconnect jump connection if present
            self.disconnect_jump()

            if self._sftp:
                try:
                    self._sftp.close()
                except Exception:
                    pass
                self._sftp = None

            if self._client:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None

    def test_connection(self) -> bool:
        """Test if connection is alive.

        Returns:
            True if connected and responsive
        """
        if not self.is_connected:
            return False
        try:
            stdin, stdout, stderr = self._client.exec_command("echo ok", timeout=5)
            stdout.read()
            stderr.read()
            return stdout.channel.recv_exit_status() == 0
        except Exception:
            return False
        finally:
            for stream_name in ("stdin", "stdout", "stderr"):
                stream = locals().get(stream_name)
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass

    @property
    def is_jump_connected(self) -> bool:
        """Check if connected to jump/compute node."""
        if self._jump_client is None:
            return False
        try:
            transport = self._jump_client.get_transport()
            return transport is not None and transport.is_active()
        except Exception:
            return False

    def connect_with_jump(
        self,
        main_host: str,
        jump_host: Optional[str] = None,
        jump_password: Optional[str] = None,
        jump_key_file: Optional[str] = None,
        main_password: Optional[str] = None,
        main_key_file: Optional[str] = None,
    ) -> None:
        """Connect to main/compute node through jump server.

        This method supports multi-hop SSH connections where:
        - The jump server is already connected (self._client)
        - The main_host is the target compute node to connect to

        Args:
            main_host: Target node name or address (e.g., "node110")
            jump_host: Not used when called after connect() - for API compatibility
            jump_password: Not used - for API compatibility
            jump_key_file: Not used - for API compatibility
            main_password: Password for main/compute node (None for internal trust)
            main_key_file: SSH key file for main/compute node

        Raises:
            SSHConnectionError: If connection fails
        """
        if not self.is_connected:
            raise SSHConnectionError("Must connect to main server first")

        if self._jump_client is not None or self._jump_channel is not None or self._jump_sftp is not None:
            self.disconnect_jump()

        try:
            # Open channel to node through main server
            transport = self._client.get_transport()
            dest_addr = (main_host, 22)
            local_addr = ("127.0.0.1", 0)
            self._jump_channel = transport.open_channel("direct-tcpip", dest_addr, local_addr)

            # Connect through the channel
            self._jump_client = paramiko.SSHClient()

            # Use secure host key policy for jump connection
            policy = InteractiveHostKeyPolicy(
                confirm_callback=self._host_key_callback, auto_add=self._auto_add_host_keys
            )
            self._jump_client.set_missing_host_key_policy(policy)

            if main_password:
                # Password authentication for compute node
                self._jump_client.connect(
                    hostname=main_host,
                    username=self._user,
                    password=main_password,
                    sock=self._jump_channel,
                    timeout=self._timeout,
                    allow_agent=False,
                    look_for_keys=False,
                )
            elif main_key_file:
                # SSH key authentication for compute node
                self._jump_client.connect(
                    hostname=main_host,
                    username=self._user,
                    key_filename=main_key_file,
                    sock=self._jump_channel,
                    timeout=self._timeout,
                    allow_agent=False,
                    look_for_keys=False,
                )
            else:
                # Internal trust - try without explicit auth
                # (relies on SSH agent or authorized_keys on compute node)
                self._jump_client.connect(
                    hostname=main_host,
                    username=self._user,
                    sock=self._jump_channel,
                    timeout=self._timeout,
                    allow_agent=True,
                    look_for_keys=True,
                )
        except Exception as e:
            try:
                if self._jump_client:
                    self._jump_client.close()
            except Exception:
                pass
            try:
                if self._jump_channel:
                    self._jump_channel.close()
            except Exception:
                pass
            self._jump_client = None
            self._jump_channel = None
            raise SSHConnectionError(f"Jump connection failed: {e}") from e

    def disconnect_jump(self) -> None:
        """Disconnect from compute node (jump connection)."""
        if self._jump_sftp:
            try:
                self._jump_sftp.close()
            except Exception:
                pass
            self._jump_sftp = None

        if self._jump_client:
            try:
                self._jump_client.close()
            except Exception:
                pass
            self._jump_client = None

        if self._jump_channel:
            try:
                self._jump_channel.close()
            except Exception:
                pass
            self._jump_channel = None

    def get_active_client(self) -> Optional[SSHClient]:
        """Get the active SSH client (jump or main).

        Returns:
            Active SSH client for command execution
        """
        if self.is_jump_connected:
            return self._jump_client
        return self._client

    def execute(self, command: str, timeout: Optional[int] = None) -> Tuple[str, str, int]:
        """Execute command on remote server.

        Uses the active client (jump client if connected, otherwise main client).

        Args:
            command: Command to execute
            timeout: Command timeout in seconds

        Returns:
            Tuple of (stdout, stderr, exit_code)

        Raises:
            SSHConnectionError: If not connected
        """
        with self._state_lock:
            client = self.get_active_client()
            if client is None:
                raise SSHConnectionError("Not connected to server")

        # Treat the user-visible `timeout` as a wall-clock deadline for the
        # whole command. paramiko's `timeout=` on exec_command is only a
        # per-recv idle timeout, so a remote process that keeps dribbling
        # output (or stays stuck inside the read loop) would never abort.
        total_timeout = timeout or self._timeout
        start = time.monotonic()
        try:
            stdin, stdout, stderr = client.exec_command(command, timeout=total_timeout)
            channel = stdout.channel
            out_chunks: list[bytes] = []
            err_chunks: list[bytes] = []
            while not channel.exit_status_ready() or channel.recv_ready() or channel.recv_stderr_ready():
                if (time.monotonic() - start) > total_timeout:
                    try:
                        channel.close()
                    except Exception:
                        pass
                    raise SSHConnectionError(f"execute exceeded total_timeout={total_timeout}s; aborting")
                if channel.recv_ready():
                    out_chunks.append(channel.recv(65536))
                if channel.recv_stderr_ready():
                    err_chunks.append(channel.recv_stderr(65536))
                if not channel.recv_ready() and not channel.recv_stderr_ready():
                    time.sleep(0.01)
            while channel.recv_ready():
                out_chunks.append(channel.recv(65536))
            while channel.recv_stderr_ready():
                err_chunks.append(channel.recv_stderr(65536))
            exit_code = channel.recv_exit_status()
            return (
                b"".join(out_chunks).decode("utf-8", errors="replace"),
                b"".join(err_chunks).decode("utf-8", errors="replace"),
                exit_code,
            )
        except SSHException as e:
            raise SSHConnectionError(f"Command execution failed: {e}")
        finally:
            for stream_name in ("stdin", "stdout", "stderr"):
                stream = locals().get(stream_name)
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass

    def execute_stream(
        self,
        command: str,
        callback: Optional[Callable[[str], None]] = None,
        total_timeout: Optional[float] = None,
    ) -> Generator[str, None, int]:
        """Execute command and stream output (both stdout and stderr).

        Uses the active client (jump client if connected, otherwise main client).

        Args:
            command: Command to execute
            callback: Optional callback for each line of output
            total_timeout: Wall-clock deadline (seconds) for the entire
                streaming session. Default None preserves prior unbounded
                behavior; pass a number to abort if the remote process hangs.

        Yields:
            Lines of output (from both stdout and stderr)

        Returns:
            Exit code
        """
        import select

        with self._state_lock:
            client = self.get_active_client()
            if client is None:
                raise SSHConnectionError("Not connected to server")
            transport = client.get_transport()
            channel = transport.open_session()
        # Channel I/O runs outside the state lock so concurrent operations
        # on the same SSHManager (e.g. a stop-button kill from another
        # thread) can still open their own session.

        start = time.monotonic()
        try:
            channel.exec_command(command)

            # Make channel non-blocking for reading
            channel.setblocking(0)

            # Read output in real-time from both stdout and stderr
            while not channel.exit_status_ready() or channel.recv_ready() or channel.recv_stderr_ready():
                if total_timeout is not None and (time.monotonic() - start) > total_timeout:
                    try:
                        channel.close()
                    except Exception:
                        pass
                    raise SSHConnectionError(f"execute_stream exceeded total_timeout={total_timeout}s; aborting")
                # Wait up to 0.1s for either stdout or stderr to become
                # readable. We don't use the return value — the subsequent
                # recv_ready() / recv_stderr_ready() checks are the source
                # of truth — but the select call still serves as an
                # event-driven sleep that yields the thread until data
                # arrives or the timeout elapses, instead of busy-looping.
                select.select([channel], [], [], 0.1)

                if channel.recv_ready():
                    data = channel.recv(4096).decode("utf-8", errors="replace")
                    for line in data.splitlines(keepends=True):
                        if callback:
                            callback(line)
                        yield line

                if channel.recv_stderr_ready():
                    data = channel.recv_stderr(4096).decode("utf-8", errors="replace")
                    for line in data.splitlines(keepends=True):
                        if callback:
                            callback(line)
                        yield line

            # Read any remaining data after exit
            while channel.recv_ready():
                data = channel.recv(4096).decode("utf-8", errors="replace")
                for line in data.splitlines(keepends=True):
                    if callback:
                        callback(line)
                    yield line

            while channel.recv_stderr_ready():
                data = channel.recv_stderr(4096).decode("utf-8", errors="replace")
                for line in data.splitlines(keepends=True):
                    if callback:
                        callback(line)
                    yield line

            return channel.recv_exit_status()
        finally:
            try:
                channel.close()
            except Exception:
                pass

    def _get_sftp(self) -> paramiko.SFTPClient:
        """Get or create SFTP client.

        Returns:
            SFTP client

        Raises:
            SSHConnectionError: If not connected
        """
        with self._state_lock:
            active_client = self.get_active_client()
            if active_client is None:
                raise SSHConnectionError("Not connected to server")

            if active_client is self._jump_client:
                if self._jump_sftp is None:
                    self._jump_sftp = self._jump_client.open_sftp()
                return self._jump_sftp

            if self._sftp is None:
                self._sftp = self._client.open_sftp()
        return self._sftp

    def open_sftp(self) -> paramiko.SFTPClient:
        """Public alias for the cached SFTP client.

        The returned client is owned by this SSHManager and reused across
        callers; callers must NOT call .close() on it (closing is handled
        by SSHManager.disconnect()).
        """
        return self._get_sftp()

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a file to remote server.

        Args:
            local_path: Local file path
            remote_path: Remote destination path (POSIX)
        """
        sftp = self._get_sftp()
        # Use posixpath for remote paths so a Windows client doesn't truncate
        # at a backslash that's actually part of a remote (POSIX) filename.
        remote_dir = posixpath.dirname(remote_path)
        if remote_dir:
            self._ensure_remote_dir(remote_dir)
        sftp.put(local_path, remote_path)

    def download_file(self, remote_path: str, local_path: str) -> None:
        """Download a file from remote server.

        Args:
            remote_path: Remote file path
            local_path: Local destination path
        """
        sftp = self._get_sftp()
        # Ensure local directory exists
        local_dir = os.path.dirname(local_path)
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)
        sftp.get(remote_path, local_path)

    def upload_directory(self, local_dir: str, remote_dir: str) -> None:
        """Upload a directory recursively.

        Args:
            local_dir: Local directory path
            remote_dir: Remote destination path (POSIX)
        """
        sftp = self._get_sftp()
        self._ensure_remote_dir(remote_dir)

        for root, dirs, files in os.walk(local_dir):
            rel_path = os.path.relpath(root, local_dir)
            if rel_path == ".":
                remote_root = remote_dir
            else:
                # On Windows, os.path.relpath returns backslash-separated
                # paths; convert to POSIX before joining onto a POSIX remote
                # path, otherwise the leading-component replace() leaves
                # mid-path backslashes in place.
                rel_path_posix = rel_path.replace(os.sep, "/")
                remote_root = posixpath.join(remote_dir, rel_path_posix)
                self._ensure_remote_dir(remote_root)

            for file in files:
                local_file = os.path.join(root, file)
                remote_file = posixpath.join(remote_root, file)
                sftp.put(local_file, remote_file)

    def _ensure_remote_dir(self, remote_dir: str) -> None:
        """Ensure remote directory exists.

        Args:
            remote_dir: Remote directory path
        """
        sftp = self._get_sftp()
        normalized = remote_dir.replace("\\", "/").rstrip("/")
        dirs = [part for part in normalized.split("/") if part]
        is_absolute = normalized.startswith("/")
        path = ""
        for d in dirs:
            if is_absolute:
                path = f"{path.rstrip('/')}/{d}" if path else f"/{d}"
            else:
                path = f"{path}/{d}" if path else d
            try:
                sftp.stat(path)
            except FileNotFoundError:
                sftp.mkdir(path)

    # Allowed characters in a remote $HOME path before we trust it for
    # interpolation into shell commands. Anything outside this set (spaces,
    # `;`, backticks, `$`, quotes, ...) could break command construction or
    # enable injection via callers that interpolate the path directly into
    # shell strings (detect_python_interpreters / detect_conda_envs do this
    # to support `ls -d {home}/miniconda*/bin/python` style globs, which
    # cannot be quoted as a single token without disabling the glob).
    _SAFE_HOME_RE = re.compile(r"^[A-Za-z0-9_./\-]+$")

    def _get_home_dir(self) -> str:
        """Get remote home directory.

        Returns:
            Home directory path. Falls back to ``/home/<user>`` if the
            remote echo fails or returns a value with characters that are
            unsafe for unquoted shell interpolation (so that downstream
            ``f"ls -d {home}/..."`` commands cannot be hijacked).
        """
        stdout, _, _ = self.execute("echo $HOME", timeout=5)
        candidate = stdout.strip()
        fallback = f"/home/{self._user}"
        if not candidate:
            return fallback
        if not self._SAFE_HOME_RE.match(candidate):
            logger.warning(
                "Remote $HOME contains unsafe characters (%r); falling back to %s",
                candidate,
                fallback,
            )
            return fallback
        return candidate

    def detect_python_interpreters(self) -> List[str]:
        """Detect available Python interpreters on remote server.

        Searches for conda/miniconda installations in user's home directory.
        Excludes system Python paths.

        Returns:
            List of Python interpreter paths
        """
        self._last_detection_errors = []
        pythons = []
        home = self._get_home_dir()

        # System paths to exclude (don't want system or /opt installs)
        system_prefixes = ["/usr/bin", "/bin", "/opt/", "/usr/local/bin"]

        def is_system_path(path: str) -> bool:
            return any(path.startswith(prefix) for prefix in system_prefixes)

        # Method 1: Find conda/miniconda directories in home (handles miniconda3-3.12 etc.)
        try:
            # Find all conda-like directories
            cmd = f"ls -d {home}/miniconda*/bin/python {home}/miniforge*/bin/python {home}/anaconda*/bin/python {home}/mambaforge*/bin/python 2>/dev/null"
            stdout, _, exit_code = self.execute(cmd, timeout=10)
            if exit_code == 0 and stdout.strip():
                for path in stdout.strip().split("\n"):
                    path = path.strip()
                    if path and path not in pythons:
                        pythons.append(path)
        except Exception as exc:
            self._record_detection_error("Python discovery command failed", exc)

        # Method 2: Use interactive login shell (bash -i -l) to get same env as user
        login_cmds = [
            "bash -i -l -c 'which python3' 2>/dev/null",
            "bash -i -l -c 'which python' 2>/dev/null",
        ]

        for cmd in login_cmds:
            try:
                stdout, _, exit_code = self.execute(cmd, timeout=15)
                if exit_code == 0 and stdout.strip():
                    # `bash -i -l` runs the user's .bashrc/.profile, which
                    # commonly print welcome banners, prompt fragments, or
                    # `cd` notices before our `which python` output. Pick
                    # the last line that actually looks like an absolute
                    # path instead of blindly taking `[0]`.
                    path = ""
                    for line in stdout.splitlines():
                        line = line.strip()
                        if line.startswith("/"):
                            path = line
                    if path and path not in pythons and not is_system_path(path):
                        pythons.append(path)
            except Exception as exc:
                self._record_detection_error("Python discovery command failed", exc)
                continue

        # Method 3: Check .local/bin (pip user install)
        try:
            local_python = f"{home}/.local/bin/python3"
            quoted_python = shlex.quote(local_python)
            stdout, _, exit_code = self.execute(f"test -x {quoted_python} && echo {quoted_python}", timeout=5)
            if exit_code == 0 and stdout.strip():
                result = stdout.strip()
                if result and result not in pythons:
                    pythons.append(result)
        except Exception as exc:
            self._record_detection_error("Python discovery command failed", exc)

        return pythons

    def detect_conda_envs(self) -> List[Tuple[str, str]]:
        """Detect conda environments on remote server.

        Returns:
            List of (env_name, env_path) tuples
        """
        self._last_detection_errors = []
        envs = []
        home = self._get_home_dir()

        conda_exe = None

        # Method 1: Use wildcard to find conda in versioned directories (e.g., miniconda3-3.12)
        try:
            cmd = f"ls -d {home}/miniconda*/bin/conda {home}/miniforge*/bin/conda {home}/anaconda*/bin/conda {home}/mambaforge*/bin/conda 2>/dev/null | head -1"
            stdout, _, exit_code = self.execute(cmd, timeout=10)
            if exit_code == 0 and stdout.strip():
                conda_exe = stdout.strip().split("\n")[0]
        except Exception as exc:
            self._record_detection_error("Conda discovery command failed", exc)

        # Method 2: Try interactive login shell to get conda from user's environment
        if not conda_exe:
            try:
                stdout, _, exit_code = self.execute("bash -i -l -c 'which conda' 2>/dev/null", timeout=15)
                if exit_code == 0 and stdout.strip():
                    conda_exe = stdout.strip().split("\n")[0]
            except Exception as exc:
                self._record_detection_error("Conda discovery command failed", exc)

        if not conda_exe:
            return envs

        # Get environment list
        try:
            quoted_conda = shlex.quote(conda_exe)
            stdout, _, exit_code = self.execute(f"{quoted_conda} env list", timeout=10)
            if exit_code == 0:
                for line in stdout.strip().split("\n"):
                    line = line.strip()
                    if line and not line.startswith("#"):
                        parts = line.split()
                        if len(parts) >= 1:
                            name = parts[0].replace("*", "").strip()
                            path = parts[-1] if len(parts) > 1 else ""
                            if name and name != "base":
                                envs.append((name, path))
                            elif name == "base":
                                envs.insert(0, (name, path))
        except Exception as exc:
            self._record_detection_error("Conda discovery command failed", exc)

        return envs

    def check_openbench_installed(self, path: str, python_path: str = "python3") -> bool:
        """Check if OpenBench v3 is installed at the given path.

        v3 is a `pip install colm-openbench` package — the legacy v2 marker
        ``openbench/openbench.py`` no longer exists. We accept either:
          * an editable / source checkout: ``src/openbench/cli/main.py``
            present under ``path``, OR
          * an importable installed module: ``python -m openbench --help``
            exits 0 from the given ``path``.

        Args:
            path: Remote directory to check.
            python_path: Remote Python interpreter to use for the
                module-import probe (defaults to "python3").
        """
        # Editable / repo-checkout marker first (cheap, no Python startup).
        quoted_path = shlex.quote(path)
        cli_marker = shlex.quote(f"{path}/src/openbench/cli/main.py")
        stdout, _, exit_code = self.execute(f"test -f {cli_marker} && echo exists", timeout=5)
        if exit_code == 0 and "exists" in stdout:
            return True
        # Fall back to actually importing the module from that cwd.
        quoted_python = shlex.quote(python_path)
        _, _, exit_code = self.execute(
            f"cd {quoted_path} && {quoted_python} -m openbench --help >/dev/null 2>&1",
            timeout=15,
        )
        return exit_code == 0
