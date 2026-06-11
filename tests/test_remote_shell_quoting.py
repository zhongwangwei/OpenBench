import base64
import json
import re
import shlex
from pathlib import Path

from openbench.gui.data_validator import FilePathGenerator, RemoteNetCDFValidator


class CapturingSSH:
    is_connected = True

    def __init__(self, response=("", "", 0)):
        self.commands = []
        self.response = response

    def execute(self, command, timeout=30):
        self.commands.append(command)
        return self.response


def test_remote_glob_quotes_directory_and_find_pattern():
    ssh = CapturingSSH(response=("/remote/proj/a.nc\n", "", 0))
    root = "/remote/proj/bad'; touch /tmp/openbench_pwn; echo '"
    prefix = "tas'; touch /tmp/prefix_pwn; echo '"
    suffix = ""
    gen = FilePathGenerator(
        root_dir=root,
        sub_dir="",
        prefix=prefix,
        suffix=suffix,
        data_groupby="Year",
        syear=2000,
        eyear=2001,
        is_remote=True,
        ssh_manager=ssh,
    )

    assert gen.get_sample_paths() == ["/remote/proj/a.nc"]

    pattern = f"{prefix}*{suffix}.nc"
    assert ssh.commands == [
        f"find {shlex.quote(root)} -maxdepth 1 -name {shlex.quote(pattern)} -type f 2>/dev/null | sort"
    ]


def test_remote_validator_quotes_file_checks():
    ssh = CapturingSSH(response=("", "", 0))
    path = "/data/a'; touch /tmp/openbench_pwn; echo '.nc"

    result = RemoteNetCDFValidator(ssh).check_file_exists(path)

    assert result.passed is True
    assert ssh.commands == [f"test -f {shlex.quote(path)}"]


def test_remote_inspect_script_embeds_path_as_json_and_quotes_environment():
    ssh = CapturingSSH(response=(json.dumps({"success": True, "variables": ["tas"]}), "", 0))
    path = '/data/has "quotes" and \\ backslash.nc'
    python_path = "/opt/py'thon/bin/python"
    conda_env = "env'; touch /tmp/openbench_pwn; echo '"

    result = RemoteNetCDFValidator(ssh, python_path=python_path, conda_env=conda_env).check_variable(path, "tas")

    assert result.passed is True
    command = ssh.commands[-1]
    # No derivable conda base from this python_path -> login-shell fallback,
    # with the entire inner command shlex-quoted once more for bash -l -c.
    assert command.startswith("bash -l -c ")
    inner = shlex.split(command.split("bash -l -c ", 1)[1])[0]
    assert f"conda activate {shlex.quote(conda_env)}" in inner
    assert f"| base64 -d | {shlex.quote(python_path)}" in inner

    encoded_arg = inner.split("printf %s ", 1)[1].split(" | base64 -d", 1)[0]
    encoded = shlex.split(encoded_arg)[0]
    script = base64.b64decode(encoded).decode("utf-8")
    assert f"safe_open({json.dumps(path)})" in script
    assert f'safe_open("{path}")' not in script


def test_gui_remote_shell_commands_do_not_single_quote_fstring_paths():
    """Avoid cmd f"cat '{path}'" patterns that break on apostrophes."""
    risky = re.compile(r"(?:execute\(f|cmd\s*=\s*f)[\"'][^\n]*(?:cat|test|mkdir|find|ls)[^\n]*'\{")
    offenders = []
    for source in Path("src/openbench/gui").rglob("*.py"):
        text = source.read_text(encoding="utf-8")
        for match in risky.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{source}:{line}: {match.group(0)}")

    assert offenders == []


def test_remote_glob_exposes_ssh_failures_instead_of_looking_empty():
    class FailingSSH:
        def execute(self, command, timeout=30):
            raise RuntimeError("network down")

    gen = FilePathGenerator(
        root_dir="/remote/data",
        sub_dir="",
        prefix="tas",
        suffix="",
        data_groupby="Year",
        syear=2000,
        eyear=2001,
        is_remote=True,
        ssh_manager=FailingSSH(),
    )

    assert gen.get_sample_paths() == []
    assert gen.last_error == "Remote glob failed for /remote/data/tas*.nc: network down"


def test_remote_validation_reports_listing_failure_not_no_files():
    class FailingSSH:
        def execute(self, command, timeout=30):
            raise RuntimeError("network down")

    from openbench.gui.data_validator import DataValidator

    validator = DataValidator(is_remote=True, ssh_manager=FailingSSH())
    result = validator.validate_source(
        "Tas",
        "RemoteSource",
        {
            "general": {"root_dir": "/remote/data", "data_groupby": "Year", "data_type": "grid"},
            "varname": "tas",
            "prefix": "tas",
            "suffix": "",
        },
        {"syear": 2000, "eyear": 2001},
    )

    assert result.checks[0].name == "file_exists"
    assert result.checks[0].passed is False
    assert result.checks[0].message == "Remote glob failed for /remote/data/tas*.nc: network down"
