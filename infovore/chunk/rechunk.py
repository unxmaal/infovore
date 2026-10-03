import sqlite3
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from infovore.chunk.gaps import derive_gap, fold_gap, record_gap
from infovore.chunk.grouper import content_hash
from infovore.chunk.measure import BUCKET_LABELS, ZERO_DISTRIBUTION, Distribution, bucket_index
from infovore.chunk.recipe import ChunkRecipe
from infovore.chunk.rules import group_messages
from infovore.config import ConfigError
from infovore.db.annotations import Annotation, record_annotation
from infovore.db.chunk_recipes import recipe_for_version
from infovore.db.codec import from_db_time, to_db_time
from infovore.db.connection import transaction
from infovore.db.raw import light_channel_messages
from infovore.rows import GroupingRule
from infovore.triage.human import HUMAN_SCORER

RELEVANT = "relevant"
IRRELEVANT = "irrelevant"
LABEL_KINDS = ("one_to_one", "merged", "split", "ambiguous", "conflict")


class UnknownRecipeError(ConfigError):
    pass


@dataclass(frozen=True)
class Target:
    channel_id: int
    thread_id: int | None
    rule: GroupingRule
    message_ids: tuple[int, ...]
    started_at: datetime
    ended_at: datetime
    reuse_id: int | None


@dataclass(frozen=True)
class Remap:
    old_id: int
    target: int | None
    kind: str
    shared: int
    old_messages: int


@dataclass(frozen=True)
class Ref:
    target: int | None = None
    exchange: int | None = None


@dataclass(frozen=True)
class LabelCopy:
    old_id: int
    target: int


@dataclass
class RechunkPlan:
    version: int
    targets: list[Target] = field(default_factory=list)
    remaps: list[Remap] = field(default_factory=list)
    superseded: list[int] = field(default_factory=list)
    gaps: dict[int, timedelta] = field(default_factory=dict)
    done_superseded: int = 0
    copies: list[LabelCopy] = field(default_factory=list)
    label_kinds: Counter[str] = field(default_factory=Counter)
    labels_before: tuple[int, int] = (0, 0)
    labels_after: tuple[int, int] = (0, 0)
    slices: dict[str, list[Ref]] = field(default_factory=dict)
    slice_before: dict[str, int] = field(default_factory=dict)
    sizes_before: Distribution = ZERO_DISTRIBUTION
    sizes_after: Distribution = ZERO_DISTRIBUTION

    @property
    def reused(self) -> int:
        return sum(1 for target in self.targets if target.reuse_id is not None)


def _old_exchanges(conn: sqlite3.Connection, version: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, channel_id, content_hash, message_count, extraction_status FROM exchanges"
        " WHERE superseded_by_recipe IS NULL AND COALESCE(chunk_recipe, 0) != ? ORDER BY id",
        (version,),
    ).fetchall()


def _members(conn: sqlite3.Connection, version: int, channel_id: int) -> dict[int, list[int]]:
    members: dict[int, list[int]] = {}
    for row in conn.execute(
        "SELECT em.exchange_id AS exchange_id, em.message_id AS message_id"
        " FROM exchange_messages em JOIN exchanges e ON e.id = em.exchange_id"
        " WHERE e.channel_id = ? AND e.superseded_by_recipe IS NULL"
        " AND COALESCE(e.chunk_recipe, 0) != ? ORDER BY em.exchange_id, em.position",
        (channel_id, version),
    ):
        members.setdefault(row["exchange_id"], []).append(row["message_id"])
    return members


def _kind(target: Target | None, shared: int, old_messages: int) -> str:
    if target is None:
        return "ambiguous"
    if shared == old_messages:
        return "one_to_one" if len(target.message_ids) == old_messages else "merged"
    return "split"


def _plan_channel(
    conn: sqlite3.Connection,
    plan: RechunkPlan,
    recipe: ChunkRecipe,
    channel_id: int,
    old: list[sqlite3.Row],
    include_bots: bool,
) -> None:
    members = _members(conn, plan.version, channel_id)
    wanted = {message_id for ids in members.values() for message_id in ids}
    everything = light_channel_messages(conn, channel_id)
    gap = derive_gap(conn, recipe, channel_id, include_bots, everything)
    plan.gaps[channel_id] = gap
    pool = [message for message in everything if message.id in wanted]
    groups = group_messages(
        pool,
        gap,
        recipe.max_messages,
        include_bots,
        fold_gap(recipe, gap),
        recipe.fold_size or 1,
    )
    by_hash = {row["content_hash"]: row["id"] for row in old}
    owner: dict[int, int] = {}
    reused: set[int] = set()
    for group in groups:
        ids = tuple(message.id for message in group.messages)
        reuse_id = by_hash.get(content_hash(ids))
        slot = len(plan.targets)
        plan.targets.append(
            Target(
                channel_id=channel_id,
                thread_id=group.messages[0].thread_id,
                rule=group.rule,
                message_ids=ids,
                started_at=group.messages[0].created_at,
                ended_at=group.messages[-1].created_at,
                reuse_id=reuse_id,
            )
        )
        plan.sizes_after = _bump(plan.sizes_after, len(ids), recipe.max_messages)
        for message_id in ids:
            owner[message_id] = slot
        if reuse_id is not None:
            reused.add(reuse_id)
    for row in old:
        old_ids = members.get(row["id"], [])
        plan.sizes_before = _bump(plan.sizes_before, len(old_ids), recipe.max_messages)
        counts = Counter(owner[m] for m in old_ids if m in owner)
        best = counts.most_common(1)[0] if counts else None
        shared = best[1] if best is not None else 0
        index = best[0] if best is not None and 2 * shared > len(old_ids) else None
        target = None if index is None else plan.targets[index]
        kind = _kind(target, shared, len(old_ids))
        plan.remaps.append(Remap(row["id"], index, kind, shared, len(old_ids)))
        if row["id"] not in reused:
            plan.superseded.append(row["id"])
            plan.done_superseded += row["extraction_status"] == "done"


def _bump(distribution: Distribution, size: int, cap: int) -> Distribution:
    counts = list(distribution)
    counts[bucket_index(size, cap)] += 1
    return (counts[0], counts[1], counts[2], counts[3], counts[4], counts[5])


def _effective_labels(conn: sqlite3.Connection) -> dict[int, str]:
    labels: dict[int, str] = {}
    for row in conn.execute(
        "SELECT subject_id, label FROM annotations WHERE scorer = ? AND subject_kind = 'exchange'"
        " AND reproducibility = 'recorded' ORDER BY id",
        (HUMAN_SCORER,),
    ):
        if row["label"] in (RELEVANT, IRRELEVANT):
            labels[row["subject_id"]] = row["label"]
        else:
            labels.pop(row["subject_id"], None)
    return labels


def _tally(labels: Iterable[str]) -> tuple[int, int]:
    counts = Counter(labels)
    return counts[RELEVANT], counts[IRRELEVANT]


def _reused_ids(plan: RechunkPlan) -> set[int]:
    return {target.reuse_id for target in plan.targets if target.reuse_id is not None}


def _plan_labels(conn: sqlite3.Connection, plan: RechunkPlan) -> None:
    superseded = {
        row[0] for row in conn.execute("SELECT id FROM exchanges WHERE superseded_by_recipe")
    }
    live = {eid: label for eid, label in _effective_labels(conn).items() if eid not in superseded}
    remap = {r.old_id: r for r in plan.remaps}
    reused = _reused_ids(plan)
    plan.labels_before = _tally(live.values())
    kept = [label for eid, label in live.items() if eid not in remap or eid in reused]
    labelled = {
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT subject_id FROM annotations WHERE scorer = ?"
            " AND subject_kind = 'exchange' AND reproducibility = 'recorded'",
            (HUMAN_SCORER,),
        )
    }
    by_target: dict[int, list[int]] = {}
    for old_id in sorted(labelled & remap.keys()):
        entry = remap[old_id]
        if old_id in reused:
            plan.label_kinds["one_to_one"] += 1
        elif entry.target is None:
            plan.label_kinds["ambiguous"] += 1
        else:
            by_target.setdefault(entry.target, []).append(old_id)
    verdicts: list[str] = []
    for target, olds in by_target.items():
        labels = {live[old_id] for old_id in olds if old_id in live}
        if len(labels) > 1:
            plan.label_kinds["conflict"] += len(olds)
            continue
        verdicts.extend(labels)
        for old_id in olds:
            plan.label_kinds[remap[old_id].kind] += 1
            plan.copies.append(LabelCopy(old_id, target))
    plan.copies.sort(key=lambda copy: copy.old_id)
    plan.labels_after = _tally([*kept, *verdicts])


def _plan_slices(conn: sqlite3.Connection, plan: RechunkPlan) -> None:
    remap = {r.old_id: r for r in plan.remaps}
    reused = _reused_ids(plan)
    stored = {row["name"] for row in conn.execute("SELECT DISTINCT name FROM eval_slices")}
    rows = conn.execute(
        "SELECT name, exchange_id FROM current_eval_slices ORDER BY name, position"
    ).fetchall()
    for row in rows:
        name = f"{row['name']}@{plan.version}"
        if name in stored:
            continue
        plan.slice_before[name] = plan.slice_before.get(name, 0) + 1
        members = plan.slices.setdefault(name, [])
        entry = remap.get(row["exchange_id"])
        if entry is None or row["exchange_id"] in reused:
            ref: Ref | None = Ref(exchange=row["exchange_id"])
        else:
            ref = None if entry.target is None else Ref(target=entry.target)
        if ref is not None and ref not in members:
            members.append(ref)


def plan_rechunk(conn: sqlite3.Connection, version: int, include_bots: bool) -> RechunkPlan:
    recipe = recipe_for_version(conn, version)
    if recipe is None:
        raise UnknownRecipeError(f"no chunk recipe version {version}")
    plan = RechunkPlan(version)
    by_channel: dict[int, list[sqlite3.Row]] = {}
    for row in _old_exchanges(conn, version):
        by_channel.setdefault(row["channel_id"], []).append(row)
    for channel_id, old in sorted(by_channel.items()):
        _plan_channel(conn, plan, recipe, channel_id, old, include_bots)
    _plan_labels(conn, plan)
    _plan_slices(conn, plan)
    return plan


def apply_rechunk(conn: sqlite3.Connection, plan: RechunkPlan, at: datetime) -> None:
    version = plan.version
    with transaction(conn):
        for old_id in plan.superseded:
            conn.execute(
                "INSERT INTO superseded_exchange_messages (exchange_id, message_id, position)"
                " SELECT exchange_id, message_id, position FROM exchange_messages"
                " WHERE exchange_id = ?",
                (old_id,),
            )
            conn.execute("DELETE FROM exchange_messages WHERE exchange_id = ?", (old_id,))
            conn.execute(
                "UPDATE exchanges SET superseded_by_recipe = ?, extraction_status ="
                " CASE WHEN extraction_status = 'done' THEN 'done' ELSE 'skipped' END"
                " WHERE id = ?",
                (version, old_id),
            )
        ids: list[int] = []
        for target in plan.targets:
            if target.reuse_id is not None:
                conn.execute(
                    "UPDATE exchanges SET chunk_recipe = ? WHERE id = ?", (version, target.reuse_id)
                )
                ids.append(target.reuse_id)
                continue
            cursor = conn.execute(
                "INSERT INTO exchanges (channel_id, thread_id, first_message_id, last_message_id,"
                " started_at, ended_at, message_count, grouping_rule, content_hash,"
                " extraction_status, retry_count, chunk_recipe)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?)",
                (
                    target.channel_id,
                    target.thread_id,
                    target.message_ids[0],
                    target.message_ids[-1],
                    to_db_time(target.started_at),
                    to_db_time(target.ended_at),
                    len(target.message_ids),
                    target.rule.value,
                    content_hash(target.message_ids),
                    version,
                ),
            )
            new_id = int(cursor.lastrowid or 0)
            conn.executemany(
                "INSERT INTO exchange_messages (exchange_id, message_id, position)"
                " VALUES (?, ?, ?)",
                [
                    (new_id, message_id, position)
                    for position, message_id in enumerate(target.message_ids)
                ],
            )
            ids.append(new_id)
        for channel_id, gap in plan.gaps.items():
            record_gap(conn, version, channel_id, gap)
        _write_remaps(conn, plan, ids)
        _write_label_copies(conn, plan, ids)
        _write_slices(conn, plan, ids, at)


def _write_remaps(conn: sqlite3.Connection, plan: RechunkPlan, ids: list[int]) -> None:
    conn.executemany(
        "INSERT INTO exchange_remap (recipe, old_exchange_id, new_exchange_id, kind,"
        " shared_messages, old_messages) VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                plan.version,
                r.old_id,
                None if r.target is None else ids[r.target],
                r.kind,
                r.shared,
                r.old_messages,
            )
            for r in plan.remaps
        ],
    )


def _write_label_copies(conn: sqlite3.Connection, plan: RechunkPlan, ids: list[int]) -> None:
    for copy in plan.copies:
        rows = conn.execute(
            "SELECT * FROM annotations WHERE scorer = ? AND subject_kind = 'exchange'"
            " AND subject_id = ? AND reproducibility = 'recorded' ORDER BY id",
            (HUMAN_SCORER, copy.old_id),
        ).fetchall()
        for row in rows:
            origin = f"rechunk:{plan.version}:{copy.old_id}"
            record_annotation(
                conn,
                Annotation(
                    subject_kind="exchange",
                    subject_id=ids[copy.target],
                    scorer=HUMAN_SCORER,
                    scorer_version=row["scorer_version"],
                    reproducibility="recorded",
                    score=row["score"],
                    label=row["label"],
                    source_ref=f"{origin}|{row['source_ref']}" if row["source_ref"] else origin,
                ),
                from_db_time(row["created_at"]),
            )


def _write_slices(
    conn: sqlite3.Connection, plan: RechunkPlan, ids: list[int], at: datetime
) -> None:
    for name, members in plan.slices.items():
        base = name.split("@")[0]
        meta = conn.execute(
            "SELECT population, seed FROM current_eval_slices WHERE name = ? LIMIT 1", (base,)
        ).fetchone()
        conn.executemany(
            "INSERT INTO eval_slices (name, exchange_id, position, population, seed, frozen_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    name,
                    ids[member.target] if member.target is not None else member.exchange,
                    position,
                    meta["population"],
                    meta["seed"],
                    to_db_time(at),
                )
                for position, member in enumerate(members, start=1)
            ],
        )


def render_plan(plan: RechunkPlan, dry_run: bool) -> str:
    kinds = Counter(r.kind for r in plan.remaps)
    lines = [
        f"rechunk to recipe {plan.version} ({'dry run' if dry_run else 'applied'})",
        f"old exchanges: {len(plan.remaps)}  superseded: {len(plan.superseded)}"
        f"  reused unchanged: {plan.reused}  new: {len(plan.targets) - plan.reused}",
        f"superseded with extraction done (claims stay on the old rows): {plan.done_superseded}",
        "size        " + " ".join(f"{label:>8}" for label in BUCKET_LABELS),
        "before      " + " ".join(f"{n:>8}" for n in plan.sizes_before),
        "after       " + " ".join(f"{n:>8}" for n in plan.sizes_after),
        "exchange remap: " + ", ".join(f"{k}={kinds.get(k, 0)}" for k in LABEL_KINDS[:4]),
        f"human labels before: relevant={plan.labels_before[0]} irrelevant={plan.labels_before[1]}",
        f"human labels after:  relevant={plan.labels_after[0]} irrelevant={plan.labels_after[1]}",
        "labelled exchanges: "
        + ", ".join(f"{k}={plan.label_kinds.get(k, 0)}" for k in LABEL_KINDS),
    ]
    for name, members in plan.slices.items():
        lines.append(f"slice {name}: {plan.slice_before[name]} -> {len(members)}")
    return "\n".join(lines) + "\n"
