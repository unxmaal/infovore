"""IPv4 bind-address detection for `sift serve` (issue #131).

No new runtime dependency (`netifaces`/`psutil`) is added just to enumerate
a host's interfaces; instead this reuses whatever the OS already ships:
`ip -4 addr show` (Linux) or, failing that, `ifconfig` (macOS/BSD, and old
Linux net-tools installs). `subprocess` is normally banned repo-wide
(`pyproject.toml`'s `flake8-tidy-imports` banned-api, reserved for
`infovore/llm/claude_cli.py`) — this module is the one other place it is
used, with a matching `per-file-ignores` entry, since a pure-stdlib
alternative (`ctypes` + `getifaddrs(3)`) would need separate, untestable
struct layouts per platform, whereas the text this module parses is fully
exercised by fixtures for both formats regardless of which OS runs the
tests.
"""

import ipaddress
import re
import subprocess
from collections.abc import Callable

LOOPBACK = "127.0.0.1"

CommandRunner = Callable[[list[str]], "str | None"]

_IP_ADDR_INET_RE = re.compile(r"^\s*inet\s+(\d{1,3}(?:\.\d{1,3}){3})/\d+", re.MULTILINE)
_IFCONFIG_INET_RE = re.compile(r"^\s*inet\s+(\d{1,3}(?:\.\d{1,3}){3})\s", re.MULTILINE)

_TAILSCALE_NET = ipaddress.ip_network("100.64.0.0/10")
_RFC1918_NETS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
)


def parse_ip_addr_show(text: str) -> list[str]:
    """Every IPv4 address in `ip -4 addr show` output (Linux), e.g. the
    `192.168.0.47` out of `    inet 192.168.0.47/24 brd ... scope global
    dynamic eth0`."""
    return _IP_ADDR_INET_RE.findall(text)


def parse_ifconfig(text: str) -> list[str]:
    """Every IPv4 address in `ifconfig` output (macOS/BSD's `inet 1.2.3.4
    netmask ...`, and a point-to-point interface's `inet 1.2.3.4 --> 1.2.3.4
    netmask ...`, e.g. Tailscale's `utun9` on the maintainer's Mac)."""
    return _IFCONFIG_INET_RE.findall(text)


def _run_command(argv: list[str], timeout: float = 5.0) -> str | None:
    """Run one address-listing command, returning its stdout, or `None` if
    the binary isn't installed, it times out, or it exits non-zero — any
    of which just means `detect_ipv4_addresses` should try the next
    command (or give up and bind only loopback)."""
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def detect_ipv4_addresses(run: CommandRunner = _run_command) -> list[str]:
    """Every IPv4 address any interface on this host currently holds:
    `ip -4 addr show` (Linux) first, falling back to `ifconfig` (macOS/BSD,
    and old Linux net-tools) when `ip` isn't available or fails, and an
    empty list if neither is. Unfiltered — `default_hosts` is what keeps
    a public address off the default bind set."""
    output = run(["ip", "-4", "addr", "show"])
    if output is not None:
        return parse_ip_addr_show(output)
    output = run(["ifconfig"])
    if output is not None:
        return parse_ifconfig(output)
    return []


def is_bindable_private(ip: str) -> bool:
    """True only for loopback, RFC1918 (`10/8`, `172.16/12`, `192.168/16`)
    and Tailscale/CGNAT (`100.64.0.0/10`) IPv4 addresses — issue #131's
    hard rule that the default `sift serve` bind set never includes a
    public address, since the page has no auth and shows real usernames
    and message text."""
    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if not isinstance(address, ipaddress.IPv4Address):
        return False
    if address.is_loopback:
        return True
    if address in _TAILSCALE_NET:
        return True
    return any(address in net for net in _RFC1918_NETS)


def default_hosts(run: CommandRunner = _run_command) -> list[str]:
    """The default `sift serve` bind set (issue #131): loopback plus every
    private LAN and Tailscale IPv4 address this host has, deduplicated and
    sorted numerically. `--host` (`infovore.sift.command`) overrides this
    entirely rather than filtering into it, since an explicit `--host` is
    the operator's own, informed choice."""
    detected = detect_ipv4_addresses(run)
    hosts = {LOOPBACK} | {ip for ip in detected if is_bindable_private(ip)}
    return sorted(hosts, key=lambda ip: tuple(int(part) for part in ip.split(".")))
