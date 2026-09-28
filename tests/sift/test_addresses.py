from infovore.sift.addresses import (
    LOOPBACK,
    default_hosts,
    detect_ipv4_addresses,
    is_bindable_private,
    parse_ifconfig,
    parse_ip_addr_show,
)
from infovore.sift.addresses import (
    _run_command as run_command,
)

LINUX_IP_ADDR_SHOW = """\
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
    inet 192.168.0.47/24 brd 192.168.0.255 scope global dynamic eth0
       valid_lft 86000sec preferred_lft 86000sec
3: eth1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
    inet 203.0.113.5/24 brd 203.0.113.255 scope global dynamic eth1
       valid_lft 86000sec preferred_lft 86000sec
"""

MACOS_IFCONFIG = """\
lo0: flags=8049<UP,LOOPBACK,RUNNING,MULTICAST> mtu 16384
\tinet 127.0.0.1 netmask 0xff000000
en1: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tinet 192.168.0.47 netmask 0xffffff00 broadcast 192.168.0.255
en5: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tinet 203.0.113.5 netmask 0xffffff00 broadcast 203.0.113.255
utun9: flags=8051<UP,POINTOPOINT,RUNNING,MULTICAST> mtu 1280
\tinet 100.127.62.78 --> 100.127.62.78 netmask 0xffffffff
"""


def test_parse_ip_addr_show_extracts_every_ipv4_address() -> None:
    assert parse_ip_addr_show(LINUX_IP_ADDR_SHOW) == [
        "127.0.0.1",
        "192.168.0.47",
        "203.0.113.5",
    ]


def test_parse_ip_addr_show_empty_text() -> None:
    assert parse_ip_addr_show("") == []


def test_parse_ifconfig_extracts_every_ipv4_address() -> None:
    assert parse_ifconfig(MACOS_IFCONFIG) == [
        "127.0.0.1",
        "192.168.0.47",
        "203.0.113.5",
        "100.127.62.78",
    ]


def test_parse_ifconfig_empty_text() -> None:
    assert parse_ifconfig("") == []


def test_is_bindable_private_loopback() -> None:
    assert is_bindable_private("127.0.0.1") is True


def test_is_bindable_private_rfc1918_ranges() -> None:
    assert is_bindable_private("10.1.2.3") is True
    assert is_bindable_private("172.16.0.5") is True
    assert is_bindable_private("172.31.255.255") is True
    assert is_bindable_private("192.168.0.47") is True


def test_is_bindable_private_rejects_just_outside_172_16_slash_12() -> None:
    assert is_bindable_private("172.32.0.1") is False
    assert is_bindable_private("172.15.255.255") is False


def test_is_bindable_private_tailscale_cgnat_range() -> None:
    assert is_bindable_private("100.64.0.1") is True
    assert is_bindable_private("100.127.62.78") is True
    assert is_bindable_private("100.63.255.255") is False
    assert is_bindable_private("100.128.0.1") is False


def test_is_bindable_private_rejects_public_addresses() -> None:
    assert is_bindable_private("203.0.113.5") is False
    assert is_bindable_private("8.8.8.8") is False


def test_is_bindable_private_rejects_garbage_and_ipv6() -> None:
    assert is_bindable_private("not-an-ip") is False
    assert is_bindable_private("::1") is False


def test_detect_ipv4_addresses_prefers_ip_addr_show() -> None:
    calls = []

    def fake_run(argv: list[str]) -> str | None:
        calls.append(argv)
        if argv[0] == "ip":
            return LINUX_IP_ADDR_SHOW
        raise AssertionError("ifconfig should not be tried when `ip` succeeds")

    result = detect_ipv4_addresses(fake_run)
    assert result == ["127.0.0.1", "192.168.0.47", "203.0.113.5"]
    assert calls[0][0] == "ip"


def test_detect_ipv4_addresses_falls_back_to_ifconfig() -> None:
    def fake_run(argv: list[str]) -> str | None:
        if argv[0] == "ip":
            return None
        if argv[0] == "ifconfig":
            return MACOS_IFCONFIG
        raise AssertionError(argv)

    result = detect_ipv4_addresses(fake_run)
    assert "192.168.0.47" in result
    assert "100.127.62.78" in result


def test_detect_ipv4_addresses_returns_empty_when_nothing_available() -> None:
    assert detect_ipv4_addresses(lambda argv: None) == []


def test_default_hosts_includes_loopback_private_and_tailscale_never_public() -> None:
    def fake_run(argv: list[str]) -> str | None:
        if argv[0] == "ip":
            return None
        return MACOS_IFCONFIG

    hosts = default_hosts(fake_run)
    assert LOOPBACK in hosts
    assert "192.168.0.47" in hosts
    assert "100.127.62.78" in hosts
    assert "203.0.113.5" not in hosts


def test_default_hosts_dedupes_and_sorts_numerically() -> None:
    def fake_run(argv: list[str]) -> str | None:
        if argv[0] == "ip":
            return LINUX_IP_ADDR_SHOW
        raise AssertionError("should not fall back")

    hosts = default_hosts(fake_run)
    assert hosts.count("127.0.0.1") == 1
    assert hosts == sorted(hosts, key=lambda ip: tuple(int(part) for part in ip.split(".")))


def test_run_command_returns_stdout_on_success() -> None:
    assert run_command(["echo", "inet 127.0.0.1"]) == "inet 127.0.0.1\n"


def test_run_command_returns_none_when_binary_missing() -> None:
    assert run_command(["definitely-not-a-real-command-xyz"]) is None


def test_run_command_returns_none_on_nonzero_exit() -> None:
    assert run_command(["false"]) is None


def test_run_command_returns_none_on_timeout() -> None:
    assert run_command(["sleep", "2"], timeout=0.05) is None
