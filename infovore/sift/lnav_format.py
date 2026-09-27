"""The lnav (https://lnav.org) JSON log format for a sift batch's
`batch.log` (issue #128), verified against real lnav 0.14.0 on the
maintainer's Mac (`eric@styx`): loads cleanly under `lnav -i` / `lnav -I`,
parses `batch.log`'s one-line-per-message format into the `infovore_sift`
sqlite table, and `channel`/`author` are marked as identifiers so
`:filter-expr :channel = 'food'` (the workflow example in the README) works.

`msg`/`ex`/`p` are left queryable as `infovore_sift.msg`/`.ex`/`.p` for
`:filter-expr`, but the "body" capture is the format's `body-field` (the
portion lnav treats as the log line's message text, shown and searched in
the log view) rather than another named SQL column."""

LNAV_FORMAT_JSON = """{
  "$schema": "https://lnav.org/schemas/format-v1.schema.json",
  "infovore_sift": {
    "title": "infovore sift batch",
    "description": "Sift batch log for infovore message-level trash filtering (issue #128)",
    "url": "https://github.com/unxmaal/infovore",
    "file-pattern": "(?:^|/)(?:batch|infovore-sift)\\\\.log$",
    "regex": {
      "std": {
        "pattern": "^(?<timestamp>\\\\S+) #(?<channel>\\\\S+) (?<author>.*?) \\\\[msg:(?<msg>\\\\d+) ex:(?<ex>\\\\d+) p:(?<p>[-0-9.]+)\\\\] (?<body>.*)$"
      }
    },
    "sample": [
      {
        "line": "2026-01-01T12:00:00+00:00 #general alice [msg:101 ex:11 p:0.82] lol that is so true, happens every time honestly"
      },
      {
        "line": "2026-01-01T12:00:05+00:00 #general bob dodd [msg:102 ex:11 p:-] anyone remember the old server rules"
      }
    ],
    "level-field": "",
    "timestamp-field": "timestamp",
    "body-field": "body",
    "value": {
      "channel": { "kind": "string", "identifier": true },
      "author": { "kind": "string", "identifier": true },
      "msg": { "kind": "integer" },
      "ex": { "kind": "integer" },
      "p": { "kind": "string" },
      "body": { "kind": "string" }
    }
  }
}
"""
