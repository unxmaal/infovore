from collections.abc import Sequence


def is_channel_allowed(channel_id: int, parent_id: int | None, channel_ids: Sequence[int]) -> bool:
    if not channel_ids:
        return True
    allowlist = set(channel_ids)
    return channel_id in allowlist or (parent_id is not None and parent_id in allowlist)
