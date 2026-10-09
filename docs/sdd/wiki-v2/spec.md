# Wiki v2: open topics and written articles

## Goal

Turn the run 13/14 claims into a wiki people would read: a page for every
thing the community actually talks about, written as short cited prose rather
than a flat list. Quality filtering rests on the extraction prompt, the claim
gate and human reviews, not on the claim check.

## Requirements

- REQ-1: A claim's check verdict no longer decides whether it is published.
  Claims reviewed wrong, made_up or not_useful stay excluded.
- REQ-2: `wiki tag --runs R --endpoint E --model M [--concurrency N] [--limit N] --write [--resume]`
  asks the model for the things each claim is about and stores the tags per
  claim, recorded against a tag run with model id and prompt hash. `--resume`
  skips claims already tagged by the same model id and prompt hash.
- REQ-3: Tags are canonicalised deterministically so surface variants
  ("Indigo 2", "SGI Indigo2", "indigo2") become one topic, shown under its
  most frequent surface form; topics.toml aliases map onto their names.
- REQ-4: `wiki build --tag-run T` assigns claims to topics from tag run T
  instead of topics.toml; a topic with at least `--min-claims` claims gets a page.
- REQ-5: Within a topic, near-duplicate claims (token Jaccard >= 0.5) are
  grouped; each group is shown once with all its sources.
- REQ-6: `wiki write --tag-run T --endpoint E --model M [--min-claims N] [--concurrency N] [--limit N] --write [--resume]`
  writes one article section per (topic, co-topic group) from the grouped
  claims and stores it with model id and prompt hash; every sentence must end
  with citations of the claims it rests on.
- REQ-7: A written sentence is dropped when it cites nothing, cites an unknown
  claim, or fewer than 60% of its content tokens appear in the claims it
  cites. Dropped sentences are counted per section.
- REQ-8: `wiki build --tag-run T --article-run A` renders stored sections as
  prose with numbered citations linking to a source list (speaker, date,
  exchange, check verdict as a label); topics without an article fall back
  to the grouped claim list. Pages cross link to co-occurring topics that
  have pages.

## Non-goals

- Changing claim extraction, the claim check itself, or review pages.
- Web references and validity scores (phase 3, #267, #268).
- Second-opinion verification with OpenJev (#269); REQ-7 is the
  deterministic floor it will sit on.
- Hosting or publishing the wiki.

## Constraints

- Python 3.12+, uv, ruff (line 100), mypy strict, 100% coverage,
  red/green TDD verified by `scripts/red_green.sh` (wrap in `timeout`, #290).
- DB writes are append-only tables with triggers, added by numbered
  migrations (next is 0031). Never migrate the live DB from tests.
- LLM calls go through the gateway (:4000) with `infovore.llm.gateway_key.headers()`,
  `json_schema` strict response format (only eval-* aliases enforce it),
  temperature 0, under `lockf -k ~/localharness/queue/generation.lock`.
- Reuse `infovore.claims.extract` patterns: `Transport`, `http_post`,
  `fetch_model_id`, `extract_concurrently`-style pool, stop on signals.
- Reuse `infovore.claims.value.tokens` for token sets.
- Bulk GPU runs must end before SoHoT discovery at 21:00 EDT or be cleared
  with the projects-a1 session first.
