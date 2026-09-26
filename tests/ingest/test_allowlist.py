from infovore.ingest.allowlist import is_channel_allowed


def test_empty_allowlist_allows_any_channel() -> None:
    assert is_channel_allowed(1, None, ()) is True
    assert is_channel_allowed(999, 1, ()) is True


def test_listed_channel_id_is_allowed() -> None:
    assert is_channel_allowed(1, None, (1, 2)) is True


def test_thread_with_listed_parent_is_allowed() -> None:
    assert is_channel_allowed(5, 1, (1, 2)) is True


def test_unrelated_channel_is_not_allowed() -> None:
    assert is_channel_allowed(9, None, (1, 2)) is False


def test_thread_with_unlisted_parent_is_not_allowed() -> None:
    assert is_channel_allowed(5, 9, (1, 2)) is False
