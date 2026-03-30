# -*- coding: utf-8 -*-
"""
SSH Manager for remote server connections.

Handles SSH connections, file transfers, and remote command execution.
"""

import os
import re
import shlex
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Callable, Generator

import paramiko
from paramiko import SSHClient, RSAKey, SSHException
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
        auto_add: bool = False
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
        fingerprint = base64.b64encode(digest).decode('ascii').rstrip('=')
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
                raise HostKeyVerificationError(
                    f"Host key verification rejected by user for {hostname}"
                )

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
        auto_add_host_keys: bool = False
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
        self._timeout = timeout
        self._host = ""
        self._user = ""
        self._port = 22
        self._host_key_callback = host_key_callback
        self._auto_add_host_keys = auto_add_host_keys
        # Multi-hop connection attributes
        self._jump_client: Optional[SSHClient] = None
        self._jump_channel = None

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
        passphrase: Optional[str] = None
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

        self._client = paramiko.SSHClient()

        # Use secure host key policy
        policy = InteractiveHostKeyPolicy(
            confirm_callback=self._host_key_callback,
            auto_add=self._auto_add_host_keys
        )
        self._client.set_missing_host_key_policy(policy)

        try:
            self._client.connect(
                hostname=host,
                port=port,
                username=user,
                password=password,
                key_filename=key_file,
                timeout=self._timeout,
                allow_agent=False,
                look_for_keys=False
            )
            self._host = host
            self._user = user
            self._port = port
        except SSHException as e:
            self._client = None
            raise SSHConnectionError(f"SSH connection failed: {e}")
        except Exception as e:
            self._client = None
            raise SSHConnectionError(f"Connection failed: {e}")

    def disconnect(self) -> None:
        """Disconnect from server.

        Cleans up both main and jump connections if present.
        """
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
            self._client.exec_command("echo ok", timeout=5)
            return True
        except Exception:
            return False

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
        main_key_file: Optional[str] = None
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

        try:
            # Open channel to node through main server
            transport = self._client.get_transport()
            dest_addr = (main_host, 22)
            local_addr = ('127.0.0.1', 0)
            self._jump_channel = transport.open_channel(
                "direct-tcpip", dest_addr, local_addr
            )

            # Connect through the channel
            self._jump_client = paramiko.SSHClient()

            # Use secure host key policy for jump connection
            policy = InteractiveHostKeyPolicy(
                confirm_callback=self._host_key_callback,
                auto_add=self._auto_add_host_keys
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
                    look_for_keys=False
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
                    look_for_keys=False
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
                    look_for_keys=True
                )
        except Exception as e:
            self._jump_client = None
            self._jump_channel = None
            raise SSHConnectionError(f"Jump connection failed: {e}")

    def disconnect_jump(self) -> None:
        """Disconnect from compute node (jump connection)."""
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
        client = self.get_active_client()
        if client is None:
            raise SSHConnectionError("Not connected to server")

        try:
            stdin, stdout, stderr = client.exec_command(
                command,
                timeout=timeout or self._timeout
            )
            exit_code = stdout.channel.recv_exit_status()
            return (
                stdout.read().decode('utf-8', errors='replace'),
                stderr.read().decode('utf-8', errors='replace'),
                exit_code
            )
        except SSHException as e:
            raise SSHConnectionError(f"Command execution failed: {e}")

    def execute_stream(
        self,
        command: str,
        callback: Optional[Callable[[str], None]] = None
    ) -> Generator[str, None, int]:
        """Execute command and stream output (both stdout and stderr).

        Uses the active client (jump client if connected, otherwise main client).

        Args:
            command: Command to execute
            callback: Optional callback for each line of output

        Yields:
            Lines of output (from both stdout and stderr)

        Returns:
            Exit code
        """
        import select

        client = self.get_active_client()
        if client is None:
            raise SSHConnectionError("Not connected to server")

        transport = client.get_transport()
        channel = transport.open_session()
        channel.exec_command(command)

        # Make channel non-blocking for reading
        channel.setblocking(0)

        # Read output in real-time from both stdout and stderr
        while not channel.exit_status_ready() or channel.recv_ready() or channel.recv_stderr_ready():
            # Use select to wait for data with timeout
            readable, _, _ = select.select([channel], [], [], 0.1)

            if channel.recv_ready():
                data = channel.recv(4096).decode('utf-8', errors='replace')
                for line in data.splitlines(keepends=True):
                    if callback:
                        callback(line)
                    yield line

            if channel.recv_stderr_ready():
                data = channel.recv_stderr(4096).decode('utf-8', errors='replace')
                for line in data.splitlines(keepends=True):
                    if callback:
                        callback(line)
                    yield line

        # Read any remaining data after exit
        while channel.recv_ready():
            data = channel.recv(4096).decode('utf-8', errors='replace')
            for line in data.splitlines(keepends=True):
                if callback:
                    callback(line)
                yield line

        while channel.recv_stderr_ready():
            data = channel.recv_stderr(4096).decode('utf-8', errors='replace')
            for line in data.splitlines(keepends=True):
                if callback:
                    callback(line)
                yield line

        return channel.recv_exit_status()

    def _get_sftp(self) -> paramiko.SFTPClient:
        """Get or create SFTP client.

        Returns:
            SFTP client

        Raises:
            SSHConnectionError: If not connected
        """
        if not self.is_connected:
            raise SSHConnectionError("Not connected to server")

        if self._sftp is None:
            self._sftp = self._client.open_sftp()
        return self._sftp

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a file to remote server.

        Args:
            local_path: Local file path
            remote_path: Remote destination path
        """
        sftp = self._get_sftp()
        # Ensure remote directory exists
        remote_dir = os.path.dirname(remote_path)
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
            remote_dir: Remote destination path
        """
        sftp = self._get_sftp()
        self._ensure_remote_dir(remote_dir)

        for root, dirs, files in os.walk(local_dir):
            rel_path = os.path.relpath(root, local_dir)
            if rel_path == ".":
                remote_root = remote_dir
            else:
                remote_root = os.path.join(remote_dir, rel_path).replace("\\", "/")
                self._ensure_remote_dir(remote_root)

            for file in files:
                local_file = os.path.join(root, file)
                remote_file = os.path.join(remote_root, file).replace("\\", "/")
                sftp.put(local_file, remote_file)

    def _ensure_remote_dir(self, remote_dir: str) -> None:
        """Ensure remote directory exists.

        Args:
            remote_dir: Remote directory path
        """
        sftp = self._get_sftp()
        dirs = remote_dir.replace("\\", "/").split("/")
        path = ""
        for d in dirs:
            if not d:
                continue
            path = f"{path}/{d}"
            try:
                sftp.stat(path)
            except FileNotFoundError:
                sftp.mkdir(path)

    def _get_home_dir(self) -> str:
        """Get remote home directory.

        Returns:
            Home directory path
        """
        stdout, _, _ = self.execute("echo $HOME", timeout=5)
        return stdout.strip() or f"/home/{self._user}"

    def detect_python_interpreters(self) -> List[str]:
        """Detect available Python interpreters on remote server.

        Searches for conda/miniconda installations in user's home directory.
        Excludes system Python paths.

        Returns:
            List of Python interpreter paths
        """
        pythons = []
        home = self._get_home_dir()

        # System paths to exclude (don't want system or /opt installs)
        system_prefixes = ['/usr/bin', '/bin', '/opt/', '/usr/local/bin']

        def is_system_path(path: str) -> bool:
            return any(path.startswith(prefix) for prefix in system_prefixes)

        # Method 1: Find conda/miniconda directories in home (handles miniconda3-3.12 etc.)
        try:
            # Find all conda-like directories
            cmd = f"ls -d {home}/miniconda*/bin/python {home}/miniforge*/bin/python {home}/anaconda*/bin/python {home}/mambaforge*/bin/python 2>/dev/null"
            stdout, _, exit_code = self.execute(cmd, timeout=10)
            if exit_code == 0 and stdout.strip():
                for path in stdout.strip().split('\n'):
                    path = path.strip()
                    if path and path not in pythons:
                        pythons.append(path)
        except Exception:
            pass

        # Method 2: Use interactive login shell (bash -i -l) to get same env as user
        login_cmds = [
            "bash -i -l -c 'which python3' 2>/dev/null",
            "bash -i -l -c 'which python' 2>/dev/null",
        ]

        for cmd in login_cmds:
            try:
                stdout, _, exit_code = self.execute(cmd, timeout=15)
                if exit_code == 0 and stdout.strip():
                    path = stdout.strip().split('\n')[0]
                    if path and path not in pythons and not is_system_path(path):
                        pythons.append(path)
            except Exception:
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
        except Exception:
            pass

        return pythons

    def detect_conda_envs(self) -> List[Tuple[str, str]]:
        """Detect conda environments on remote server.

        Returns:
            List of (env_name, env_path) tuples
        """
        envs = []
        home = self._get_home_dir()

        conda_exe = None

        # Method 1: Use wildcard to find conda in versioned directories (e.g., miniconda3-3.12)
        try:
            cmd = f"ls -d {home}/miniconda*/bin/conda {home}/miniforge*/bin/conda {home}/anaconda*/bin/conda {home}/mambaforge*/bin/conda 2>/dev/null | head -1"
            stdout, _, exit_code = self.execute(cmd, timeout=10)
            if exit_code == 0 and stdout.strip():
                conda_exe = stdout.strip().split('\n')[0]
        except Exception:
            pass

        # Method 2: Try interactive login shell to get conda from user's environment
        if not conda_exe:
            try:
                stdout, _, exit_code = self.execute("bash -i -l -c 'which conda' 2>/dev/null", timeout=15)
                if exit_code == 0 and stdout.strip():
                    conda_exe = stdout.strip().split('\n')[0]
            except Exception:
                pass

        if not conda_exe:
            return envs

        # Get environment list
        try:
            quoted_conda = shlex.quote(conda_exe)
            stdout, _, exit_code = self.execute(f"{quoted_conda} env list", timeout=10)
            if exit_code == 0:
                for line in stdout.strip().split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 1:
                            name = parts[0].replace('*', '').strip()
                            path = parts[-1] if len(parts) > 1 else ""
                            if name and name != "base":
                                envs.append((name, path))
                            elif name == "base":
                                envs.insert(0, (name, path))
        except Exception:
            pass

        return envs

    def check_openbench_installed(self, path: str) -> bool:
        """Check if OpenBench is installed at given path.

        Args:
            path: Path to check

        Returns:
            True if OpenBench is installed
        """
        check_file = f"{path}/openbench/openbench.py"
        quoted_file = shlex.quote(check_file)
        stdout, _, exit_code = self.execute(f"test -f {quoted_file} && echo exists", timeout=5)
        return exit_code == 0 and "exists" in stdout
