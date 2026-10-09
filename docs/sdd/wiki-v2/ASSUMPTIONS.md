# Assumptions

- A-01: Tagging uses eval-7b, 20 claims per request | spike: correct tags in 8.7 s per 20 on run 13 claims, cheap enough for 107k claims (~5.4k requests) | if tags are poor, swap the model; prompt hash keeps runs apart.
- A-02: Article writing uses eval-12b | spike: eval-12b merged claims into prose with citations, eval-7b mostly copied claims one per sentence | if too slow for the 21:00 window, fall back to eval-7b or cap topics with --limit.
- A-03: Canonical form = casefold, drop a leading vendor word (sgi, silicon graphics, sun, ibm, dec, hp, apple), drop spaces, hyphens, slashes and dots except inside version numbers | covers "Indigo 2"/"SGI Indigo2"/"indigo2" | if it merges distinct things (e.g. "Sun 3" vs "Sun3" is fine, "IRIX 6.5" vs "IRIX 65" is not), add an exception list.
- A-04: Wiki builds from runs 13 and 14 only | new prompt, sample 90% good | older runs can be added with --runs.
- A-05: --min-claims default for pages stays 10 | current default | lower it to grow the wiki.
- A-06: A section covers one topic plus its strongest co-topic (as today's "With X" groups) and at most 40 claim groups, largest first | fits eval-12b context with room for output; very large topics lose their long tail in prose but keep it in the source list | raise the cap or split sections if pages read thin.
- A-07: REQ-7 support threshold 0.6 of sentence content tokens found in cited claims | permissive enough for paraphrase, catches invented specifics | tune on the first run's drop rate.
- A-08: Check verdict shown as a plain label on each source line, never used to filter | Eric's reviews: the check flags 38% of good claims | revisit if OpenJev (#269) gives a better signal.
- A-09: Tag runs and article runs get their own tables (tag_runs, claim_tags, article_runs, article_sections), append-only like claims_v2 | matches the repo's pattern | none expected.
