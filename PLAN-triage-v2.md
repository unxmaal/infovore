# Triage v2: embeddings, a local judge, and a real benchmark

This is an addendum to PLAN.md, drafted by Fable and revised on 2026-09-28 after review against measured results (see "Evidence" and "Review changes"). Operating rules are unchanged: red/green TDD, 100% line and branch coverage, no docs in code, README.md is the only documentation, no network in tests, Opus lead with Sonnet subagents.

## The problem

Triage exists to keep about 1.38M messages away from paid models. It isn't doing that well enough, and the reason is structural rather than a matter of tuning:

1. **The labels aren't ground truth.**
   - `lore`/`noise` on an exchange is derived from whether the extractor (mostly Sonnet) produced a claim that the Sonnet probe plus Haiku judge didn't call `known` (`triage/label.py:41-88`). Zero claims counts as noise.
   - So the classifier learns "what makes the extraction prompt fire". Every reported precision and recall number measures agreement with that pipeline, not with reality.
   - The report card's combined score also reuses the stored `p_lore` (`triage/tuning.py`), so its AUC is partly in-sample.
2. **A bag of words over an exchange can't represent "a resolved technical fact."** Presence-only unigrams across up to 50 messages, combined with Fisher's method over up to 150 correlated clues, saturate (thresholds like 0.9994) and reward length rather than content.
3. **The units don't match.** Lore is the size of one message. Triage scores whole exchanges. Sift scores messages but isn't connected to the gate, and nothing aggregates between them.
4. **Nothing measures what matters.** There is no yield figure (claims per exchange sent) and no recall check against a randomly sampled, fully labeled slice, so the gate can't be shown to help or hurt.

The fix is a funnel: deterministic rules, then embeddings with a small classifier, then a local LLM judge on the uncertain band. It is evaluated against a human-labeled random sample, and paid models run only at the end.

## Evidence (measured on the real DB, 2026-09-27/28)

- **Human labels beat proxy labels decisively.** With 833 human sift labels, a human-only message model scored out-of-fold AUC 0.89 against human labels. A model trained on about 160k citation labels scored 0.75 (#135).
- **Word counting has plateaued at about 0.85 AUC** at the message level. On identical folds (1,184 human labels, #144/#145):
  - `plain` 0.854;
  - `structural` (plus position, question-follow, reply and neighbour fact-shape signals) 0.849;
  - `context` (plus neighbours' words) 0.805, which is worse.
  Neighbour words swamp naive Bayes, the same length bias as problem 2.
- **Sift "keep" is not "has lore".** Of the human-kept messages in extracted exchanges, only about 40% are cited by any claim. The maintainer's keep means "not junk / would be lost from the conversation", not "states a checkable fact".
- **Maintainer rulings that any rubric must encode:**
  - market history (prices, sales, listings, sellers, resellers) **is lore**;
  - references and links to manuals, mirrors and software **are lore**, and they are the least-known claim type (1.6% `known`);
  - members' personal anecdotes and possessions are **trivia, not lore**;
  - the server covers far more than SGI (Sun, DEC, Amiga, IBM, Mac, PC, networking, general computing), so **never frame, filter or exclude by SGI vocabulary**;
  - `#food`, `#motor-vehicles` and `#music-geeks` are pure trash (already denylisted, #140).
- **Local compute (M2 Pro mini, localharness gateway):** Qwen3-30B-A3B ran at about 11 output tokens a second end to end, 36–250 s per exchange for full extraction. An M5 Ultra Mac Studio (96 GB) arrives in about a month.

## What changes

- **New stage 0: `prefilter`.** A hard deterministic pre-filter built from the existing pieces: the channel denylist (`INFOVORE_EXCLUDE_CHANNELS`), trivial-trash patterns, and saved sift trash rules, applied after audit. It runs before anything else.
- **New stage `embed`:** every surviving message gets a vector from a local embedding model.
- **New stage `classify`:** small classifiers on those vectors score messages, and exchanges aggregate the scores.
- **New stage `judge`:** a local LLM answers yes/no on the uncertain band, through the existing `openai_compat` backend.
- **New command `bench`:** recall, precision, yield and a cost projection against a human-labeled random sample. This is the only evaluation that counts.
- **The gate** reads the funnel's final decision instead of `p_lore`/`triage_score`.
- **Bayes** stays as a report-only baseline in `bench` until the new funnel beats it, then it is removed.

## Stack additions

- **Embeddings go through `openai_compat` by default** (localharness / llama.cpp / mlx serve an embeddings endpoint), behind an `Embedder` protocol, with the model name in config. `sentence-transformers` is an **optional extra** only: it pulls in PyTorch (gigabytes), which would bloat the multi-arch Docker image and CI and hurt host neutrality.
- **`numpy` and `scikit-learn`** are for logistic regression and calibration. This is a deliberate reversal of #95's "no heavy deps", justified by the embedding workload; document it in README. Vectors are stored as `float32` blobs in SQLite, with no vector DB.
- **The local judge** is any chat model through `openai_compat`, with an 8B-class model as the target. The model name is in config.

## Data model additions

- **message_embeddings:** message_id PK FK, model, dim, vector BLOB, embedded_at. UNIQUE(message_id, model) if multiple models are kept.
- **message_scores:** message_id FK, classifier_version FK, target (`informative` | `lore`), p REAL, PK(message_id, classifier_version, target).
- **classifier_model:** version PK autoincrement, target, trained_at, embedding_model, labels_by_source_json, params_json (coefficients, combiner, calibration), oof_auc_vs_human.
- **exchange_decisions:** exchange_id PK FK, funnel_version, prefiltered BOOL, p_message_max REAL, p_message_top2_mean REAL, band (`reject` | `uncertain` | `accept`), judge_verdict (`lore` | `noise` | NULL), judge_model NULL, judge_run_id NULL, final (`send` | `drop`), decided_at.
- **judge_runs:** id PK, exchange_id FK, model, prompt_version, started_at, finished_at, input_tokens, output_tokens, verdict, raw_output, error NULL.
- **bench_samples:** exchange_id PK FK, drawn_at, seed, label (`lore` | `noise` | NULL until labeled), labeled_by, labeled_at. Drawn uniformly at random from all exchanges, prefiltered ones included, so the benchmark also measures the pre-filter's lore loss. **Never used for training.**

`exchange_labels` and `message_labels` training queries must exclude any exchange (and its messages) present in `bench_samples`. This is enforced in the repository layer, not by callers.

## Phases

### Phase T0: Benchmark sample and labeling (the lead defines it, one subagent builds it)

1. **`bench draw --n N --seed S`** puts a uniform random sample of exchanges into `bench_samples`. It is idempotent for the same seed and refuses to redraw once any sample is labeled. **Rolling bench:** `bench draw --add N` appends new random samples with a new seed, so the bench can start small (about 150) and grow.
2. **`bench label`** is an exchange-labeling mode of the sift page (`sift/serve.py`), built for speed, because a 50-message exchange is far heavier than a single message and about 200 labels is one sitting's limit:
   - the whole exchange is shown compactly;
   - fact-shaped lines (versions, part numbers, paths, URLs, prices) are highlighted;
   - short chatter is collapsed;
   - one key per verdict, undo, resume, and saving as you go.
   It records `labeled_by=human`.
3. **The rubric**, shown on the page and in README, is written from the maintainer's rulings above.
   - **LORE:** the exchange states, corrects or demonstrates a specific technical fact, procedure, part identity or configuration a later reader could act on. It also includes **market history** (prices, sales, listings, which seller carried what) and **references** (where to get software, manuals, drivers, even as a bare link).
   - **NOISE:** questions never answered, opinions, banter, members' personal anecdotes and possessions that establish no reusable fact, and logistics that aren't market history.
   - The rubric applies to every channel, not just SGI ones.
4. **Repository guard:** training queries exclude `bench_samples`. A test proves that a labeled bench exchange never appears in any training set.

Acceptance: a fresh DB can draw, add, label and report the count labeled, and training with bench rows present provably excludes them.

**Owner task (not Claude Code): label the bench.** Start with about 150 and grow it.

Statistics: at about 25–35% lore, 150 exchanges give about 45 positives, so recall is known to roughly ±7 points. 400 exchanges give about 100–140 positives, roughly ±4 points. `bench report` prints confidence intervals, and the definition of done accounts for them.

### Phase T1: Embeddings

1. **`embed/protocol.py`:** `Embedder.embed(texts: list[str]) -> list[list[float]]`, plus `dim` and `model`. `embed/fake.py` provides deterministic hashed embeddings, fully tested.
2. **`embed/openai_compat.py`** is the default real backend. `embed/sentence_transformers.py` is behind an optional extra. Tests cover request shaping with a fake transport only.
3. **The `embed` command** embeds every non-prefiltered message without a row in `message_embeddings` for the configured model. It works in batches, is resumable, and redacts opted-out users first.
   - Text for embedding is `content` plus attachment filenames.
   - Reply-to content is prepended, truncated, when present. This is the one context feature to carry. At the message level, neighbour words measurably hurt; re-test embedding-level context once bench labels made with context exist.
   - Message ids are strings anywhere they cross JSON (Discord snowflakes exceed 2^53, #92/#134).
4. **`embed status`** reports embedded vs total, model, dim and **measured throughput** (messages per second on this host).

Acceptance: a fake-backed end-to-end run embeds every message once, a second run embeds nothing, and opt-out redaction is applied before the embedder sees text. **Budget: measure the chosen model on styx and record messages/s in README. A full pass over about 1.38M messages must fit comfortably in hours, not days.**

### Phase T2: Per-message classifiers and exchange aggregation

1. **Two targets, trained separately.** Pooling them teaches the wrong concept.
   - **`informative`**: human sift keep/trash labels ("would anything be lost if this vanished from the conversation?").
   - **`lore`**: citation labels (cited by a live claim = lore) plus exchange-level LLM `noise` propagated to that exchange's messages. An exchange labeled `lore` does **not** make all its messages lore.
2. **Label sources aren't pooled into one fit** (lesson of #135). For each target:
   - fit a proxy-label model (citation / LLM) and a human-label model on the embeddings;
   - combine them with a small logistic combiner fitted on **out-of-fold** human labels;
   - fall back to the proxy model only when human labels are too few.
   - Report AUC per source and for the combined model, always evaluated against human labels.
3. **Model details:** logistic regression with balanced class weights; 5-fold CV by `sha256(id)`; Platt or isotonic calibration on out-of-fold scores. Bench rows are excluded by the repository guard.
4. **The `classify` command** scores every embedded message for the current classifier version into `message_scores`.
5. **`classify/aggregate.py`**, per exchange:
   - computes `p_message_max` and `p_message_top2_mean` over the `lore` target, optionally gated by `informative`;
   - assigns bands from config: `reject` below `INFOVORE_BAND_LOW` (default 0.10), `accept` at or above `INFOVORE_BAND_HIGH` (default 0.80), `uncertain` between;
   - writes `exchange_decisions` with `final=drop` for reject, `final=send` for accept, and `final=NULL` for uncertain.
6. **Band budget:** print the uncertain band's size and the projected judge time at the measured judge throughput (T3). Tune the band edges so the judge workload fits the available local compute (M2 Pro now, Studio later).
7. **Property tests:** every exchange with all messages scored gets exactly one decision; aggregation is invariant to message order; changing the classifier version re-decides.

Acceptance: an end-to-end run with a fake embedder and seeded labels produces the expected bands, and per-source and combined AUC against human labels is reported and stored.

### Phase T3: Local judge

1. **`judge/prompt.py`:** a versioned constant, mirrored in README under "Judge prompt".
   - Framing: the model reads one exchange from **a hobbyist retro-computing community centred on SGI/IRIX that also discusses other vintage and general computing platforms**, and answers only `LORE` or `NOISE`.
   - The definitions are **the T0 rubric verbatim**: market history and references are LORE; personal anecdotes are NOISE.
   - Reaction counts are shown as a weak agreement signal.
   - Authors are shown as pseudonyms (`member-A`…) as in extraction (#122).
   - Output is a single token; anything else is a failed run.
2. **The `judge` command** pulls `uncertain` decisions and calls the configured `openai_compat` model with concurrency from config. It parses the verdict, writes `judge_runs`, and updates `exchange_decisions.final`. Transient failures are retried; a malformed verdict is recorded and the exchange stays `uncertain`.
3. **Throughput:** measure and record seconds per exchange for the chosen model on this host. Prompt processing, not the one-token answer, dominates.
4. **Tests:** a fake LLM backend; verdict parsing including whitespace, lowercase and refusals; the concurrency bound is respected; an idempotent re-run judges nothing.

Acceptance: after `judge`, no exchange is `uncertain` except those with a recorded failure.

### Phase T4: Gate and extract integration

1. **`triage/gate.py`** reads `exchange_decisions.final == 'send'`. The old `p_lore`/`triage_score` path is kept behind `INFOVORE_GATE=legacy` for one release so `bench` can compare, then deleted. The denylist remains a hard exclusion in both modes.
2. **`run` orchestration order** becomes: backfill → chunk → prefilter → embed → classify → judge → extract → probe.
3. **`status`** reports prefilter, band and final counts.
4. **Cut-over of a live unattended run** (e.g. `run-unattended.sh` on styx):
   - exchanges already `done` stay done;
   - stop the run, deploy, run `prefilter`/`embed`/`classify`/`judge` to completion, set `INFOVORE_GATE=funnel`, and restart;
   - `extract --order best` orders by the funnel's aggregate score.

Acceptance: `extract` in live mode touches only `send` exchanges, and the legacy gate is selectable and reproduces its old behaviour on the fixture.

### Phase T5: Bench

`bench report` runs the funnel over `bench_samples` (all stages on those exchanges only, if not already scored). It prints the following for the legacy gate, the pre-filter alone, the embedding classifier alone at each band boundary, and the full funnel with the judge:

- **recall of human-labeled `lore`**, with a confidence interval (the number that must not drop);
- **precision on sent**;
- **share of all exchanges that would be sent** (extrapolated from the sample);
- **projected paid-model calls and tokens** for the full corpus;
- **local compute time** for embed, classify and judge at measured throughput;
- **yield:** claims per sent exchange, from any bench exchanges that have live extraction runs.

It also prints `bench sweep` over band thresholds, reporting the cheapest configuration that holds recall ≥ `--min-recall` (default 0.95) **at the lower confidence bound** once the bench is large enough, and flagging when it isn't.

The README gets a "Benchmark" section recording each run's numbers with date, funnel version and bench size. These numbers are the project's definition of progress. `p_lore` holdout metrics are removed from the README once the funnel wins.

Acceptance: report and sweep are deterministic on the fixture, and the extrapolation and confidence-interval arithmetic is tested against hand-computed values.

### Phase T6: Cleanup

- Remove Bayes training, `triage_model`, `triage_tokens` and `p_lore` once `bench` shows the funnel at equal or better recall with fewer sends on two consecutive reports. A migration drops the tables.
- Sift's `p_trash` and message NB models are superseded by `message_scores` (`informative`). Keep the labeling UI and the human labels; remove the sift NB models.
- Rewrite README's "Triage" section around the funnel.

## Definition of done

- `bench report` on a human-labeled random bench of at least 400 exchanges shows the funnel holding recall ≥ 0.95 (lower confidence bound reported) while sending fewer exchanges than the legacy gate.
- No paid-model call happens before an exchange has an `exchange_decisions.final == 'send'` row.
- Training code cannot see bench rows, enforced by a test.
- Coverage is 100%, and README is the only documentation.

## Deferred

- **Message stripping before extraction:** use `informative` scores to drop trash messages from extraction prompts, keeping the question/reply context around kept messages. This was sift's original payoff, worth about half of input tokens. Do it after the funnel is benched, with its own bench-backed recall check (claims lost vs full-exchange extraction).
- **Near-duplicate clustering over embeddings** to skip restated lore. This is a cheap follow-on once vectors exist, it feeds #142 (topics and corroboration), and should come after T5 shows the funnel works.
- **An active-learning loop** that feeds judge verdicts back as classifier labels (it must never touch bench rows).
- **Replacing the judge** with a fine-tuned classifier once enough judge verdicts exist.
- **Re-testing conversation context** (embedding neighbours) once enough human labels were made with the sift context view on (#137). The current labels were mostly made one message at a time.

## Review changes (2026-09-28, from Fable's draft)

1. Added a local-compute throughput budget (embed, judge) and a band-size budget, because the M2 Pro is slow and the Studio is coming.
2. Embeddings go through `openai_compat` by default; `sentence-transformers` became an optional extra (no PyTorch in the base image).
3. The judge prompt and bench rubric now encode the maintainer's rulings: market history and references are lore, anecdotes are trivia, and the framing is not SGI-only.
4. Split the `informative` (sift keep) and `lore` targets instead of pooling them.
5. Per-source models plus an out-of-fold combiner instead of one pooled fit (#135 evidence).
6. A rolling bench, a fast exchange-labeling view, confidence intervals, and owner-effort realism.
7. Added stage 0 `prefilter` (denylist, trivial-trash patterns and audited sift rules), live-run cut-over steps, and message stripping (deferred), and corrected the numbers (1.38M messages; mostly Sonnet extraction).
