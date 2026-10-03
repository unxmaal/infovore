import io
from pathlib import Path

from infovore.cli import ExitCode, main
from infovore.db.connection import open_database


def run(argv: list[str], tmp_path: Path, exclude: str = "") -> tuple[int, str]:
    env = {
        "INFOVORE_DISCORD_TOKEN": "t",
        "INFOVORE_GUILD_ID": "1",
        "INFOVORE_DB_PATH": str(tmp_path / "infovore.db"),
        "INFOVORE_JUDGE_BACKEND": "fake",
        "INFOVORE_EXCLUDE_CHANNELS": exclude,
    }
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, environ=env, dotenv_path=None, stdout=out, stderr=err)
    return code, out.getvalue() + err.getvalue()


def _seed(tmp_path: Path) -> None:
    run(["status"], tmp_path)
    conn = open_database(tmp_path / "infovore.db")
    chans = [(1, None, "alpha", "text"), (2, None, "beta", "text"), (3, 2, "thr", "thread")]
    for cid, parent, name, kind in chans:
        conn.execute(
            "INSERT INTO channels (id, guild_id, parent_id, name, kind) VALUES (?, 1, ?, ?, ?)",
            (cid, parent, name, kind),
        )
    layout = {1: [1, 2, 3, 4], 2: [5, 6], 3: [7]}
    for cid, ids in layout.items():
        for eid in ids:
            conn.execute(
                "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id,"
                " started_at, ended_at, message_count, grouping_rule, content_hash)"
                " VALUES (?, ?, 1, 1, '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00',"
                " 1, 'quiet_gap', ?)",
                (eid, cid, f"h{eid}"),
            )
    labels = [
        (1, "relevant"),
        (2, "irrelevant"),
        (2, "relevant"),
        (3, "irrelevant"),
        (4, "bad_grouping"),
        (5, "irrelevant"),
        (6, "irrelevant"),
        (7, "relevant"),
    ]
    for eid, label in labels:
        conn.execute(
            "INSERT INTO annotations (subject_kind, subject_id, scorer, scorer_version,"
            " reproducibility, label, created_at) VALUES ('exchange', ?, 'human_exchange', 1,"
            " 'recorded', ?, '2026-01-01T00:00:00+00:00')",
            (eid, label),
        )
    conn.commit()
    conn.close()


def test_by_channel_sorted_with_latest_label_and_excluded_marked(tmp_path: Path) -> None:
    _seed(tmp_path)

    code, out = run(["judge", "report", "--by-channel"], tmp_path, exclude="beta")

    assert code == ExitCode.OK
    lines = out.strip().splitlines()
    assert lines[0].split() == [
        "channel",
        "labelled",
        "relevant",
        "irrelevant",
        "%irrelevant",
        "exchanges",
        "excluded",
    ]
    rows = [line.split() for line in lines[1:]]
    assert rows == [
        ["beta", "2", "0", "2", "100.0%", "2", "yes"],
        ["alpha", "3", "2", "1", "33.3%", "4", "no"],
        ["thr", "1", "1", "0", "0.0%", "1", "yes"],
    ]


def test_min_labels_filters_and_empty_says_so(tmp_path: Path) -> None:
    _seed(tmp_path)

    _, out = run(["judge", "report", "--by-channel", "--min-labels", "2"], tmp_path)
    names = [line.split()[0] for line in out.strip().splitlines()[1:]]
    assert names == ["beta", "alpha"]

    _, none = run(["judge", "report", "--by-channel", "--min-labels", "9"], tmp_path)
    assert none.strip() == "no channels with at least 9 labels"
