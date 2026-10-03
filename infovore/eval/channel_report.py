import sqlite3
from dataclasses import dataclass

from infovore.rows import Label
from infovore.triage.human import training_labels


@dataclass(frozen=True)
class ChannelRow:
    name: str
    relevant: int
    irrelevant: int
    exchanges: int
    excluded: bool

    @property
    def labelled(self) -> int:
        return self.relevant + self.irrelevant

    @property
    def pct_irrelevant(self) -> float:
        return self.irrelevant / self.labelled


def channel_report(
    conn: sqlite3.Connection, exclude_channels: frozenset[str], min_labels: int = 1
) -> list[ChannelRow]:
    labels, _ = training_labels(conn)
    counts: dict[int, list[int]] = {}
    info: dict[int, tuple[str, bool]] = {}
    rows = conn.execute(
        "SELECT c.name AS name, c.id AS cid, LOWER(c.name) AS lname, LOWER(p.name) AS pname,"
        " e.id AS eid FROM exchanges e JOIN channels c ON c.id = e.channel_id"
        " LEFT JOIN channels p ON p.id = c.parent_id"
    )
    for row in rows:
        cid = row["cid"]
        info[cid] = (row["name"], bool({row["lname"], row["pname"]} & exclude_channels))
        tally = counts.setdefault(cid, [0, 0, 0])
        tally[2] += 1
        label = labels.get(row["eid"])
        if label is not None:
            tally[0 if label is Label.LORE else 1] += 1
    result = [
        ChannelRow(info[cid][0], rel, irr, total, info[cid][1])
        for cid, (rel, irr, total) in counts.items()
        if rel + irr >= max(min_labels, 1)
    ]
    return sorted(result, key=lambda r: (-r.pct_irrelevant, -r.labelled, r.name))


def format_channel_report(rows: list[ChannelRow]) -> str:
    lines = ["channel labelled relevant irrelevant %irrelevant exchanges excluded"]
    lines += [
        f"{r.name} {r.labelled} {r.relevant} {r.irrelevant} {r.pct_irrelevant:.1%}"
        f" {r.exchanges} {'yes' if r.excluded else 'no'}"
        for r in rows
    ]
    return "\n".join(lines) + "\n"
