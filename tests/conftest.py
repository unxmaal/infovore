from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _no_real_gateway_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # keeps every test away from the real Keychain and the real key file
    monkeypatch.delenv("INFOVORE_GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("SOHOT_GATEWAY_KEY", raising=False)
    monkeypatch.setenv("SOHOT_GATEWAY_STORE", "file")
    monkeypatch.setenv("LOCALHARNESS_HOME", str(tmp_path / "no-localharness"))
