import sqlite3
from dataclasses import dataclass, field
from datetime import timedelta

from infovore.chunk.rules import channel_gap, group_messages
from infovore.db.raw import light_channel_messages

BUCKET_LABELS = ("1", "2", "3-5", "6-15", "16-49", "cap")
_BOUNDS = (1, 2, 5, 15, 49)

Distribution = tuple[int, int, int, int, int, int]
ZERO_DISTRIBUTION: Distribution = (0, 0, 0, 0, 0, 0)


@dataclass(frozen=True)
class MeasureRule:
    label: str
    gap: timedelta | None = None
    percentile: float | None = None
    floor: timedelta = timedelta(minutes=30)
    ceiling: timedelta = timedelta(hours=6)
    fold_factor: float = 0
    fold_size: int = 1


@dataclass
class RuleResult:
    rule: MeasureRule
    overall: Distribution = ZERO_DISTRIBUTION
    by_channel: dict[str, Distribution] = field(default_factory=dict)
    gaps: dict[str, timedelta] = field(default_factory=dict)
    volume: dict[str, int] = field(default_factory=dict)


def bucket_index(size: int, max_messages: int) -> int:
    if size >= max_messages:
        return len(_BOUNDS)
    return next(index for index, bound in enumerate(_BOUNDS) if size <= bound)


def _add(distribution: Distribution, index: int) -> Distribution:
    counts = list(distribution)
    counts[index] += 1
    return (counts[0], counts[1], counts[2], counts[3], counts[4], counts[5])


def _sum(left: Distribution, right: Distribution) -> Distribution:
    return (
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
        left[3] + right[3],
        left[4] + right[4],
        left[5] + right[5],
    )


def measure(
    conn: sqlite3.Connection,
    rules: list[MeasureRule],
    max_messages: int,
    include_bots: bool,
) -> list[RuleResult]:
    results = [RuleResult(rule) for rule in rules]
    channels = conn.execute(
        "SELECT m.channel_id AS id, COALESCE(c.name, CAST(m.channel_id AS TEXT)) AS name"
        " FROM (SELECT DISTINCT channel_id FROM messages) m"
        " LEFT JOIN channels c ON c.id = m.channel_id ORDER BY m.channel_id"
    ).fetchall()
    for channel in channels:
        messages = light_channel_messages(conn, channel["id"])
        for result in results:
            rule = result.rule
            if rule.percentile is not None:
                gap = channel_gap(messages, rule.percentile, rule.floor, rule.ceiling, include_bots)
            else:
                assert rule.gap is not None
                gap = rule.gap
            fold_gap = gap * rule.fold_factor if rule.fold_factor else None
            groups = group_messages(
                messages, gap, max_messages, include_bots, fold_gap, rule.fold_size
            )
            if not groups:
                continue
            distribution = ZERO_DISTRIBUTION
            for group in groups:
                distribution = _add(distribution, bucket_index(len(group.messages), max_messages))
            name = channel["name"]
            result.by_channel[name] = _sum(
                result.by_channel.get(name, ZERO_DISTRIBUTION), distribution
            )
            result.gaps[name] = gap
            result.volume[name] = result.volume.get(name, 0) + sum(len(g.messages) for g in groups)
            result.overall = _sum(result.overall, distribution)
    return results


def _share(distribution: Distribution) -> str:
    total = sum(distribution)
    cells = " ".join(f"{count:>8}" for count in distribution)
    small = (distribution[0] + distribution[1]) / total * 100 if total else 0.0
    return f"{cells} {total:>8} {small:>9.1f}%"


def render(results: list[RuleResult], top_channels: int) -> str:
    header = (
        f"{'rule':<24}"
        + " ".join(f"{label:>8}" for label in BUCKET_LABELS)
        + f" {'total':>8} {'1-2 msgs':>10}"
    )
    lines = ["overall", header]
    lines += [f"{result.rule.label:<24}{_share(result.overall)}" for result in results]
    sizes = results[0].volume
    for name in sorted(sizes, key=lambda n: (-sizes[n], n))[:top_channels]:
        lines += ["", f"{name}", header]
        for result in results:
            distribution = result.by_channel.get(name, ZERO_DISTRIBUTION)
            gap = result.gaps.get(name)
            note = f"  gap={int(gap.total_seconds() // 60)}m" if gap is not None else ""
            lines.append(f"{result.rule.label:<24}{_share(distribution)}{note}")
    return "\n".join(lines) + "\n"
