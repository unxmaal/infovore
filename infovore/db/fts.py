from collections.abc import Iterable

TOKENCHARS = "-./_"
TOKENIZE = f"unicode61 tokenchars '{TOKENCHARS}'"


def match_terms(raws: Iterable[str]) -> list[str]:
    """Normalise raw words into FTS5 terms, in order, without duplicates.

    A term ending in a tokenchar produces a token that matches nothing, and
    FTS5 ANDs terms, so a single trailing full stop zeroes the whole result
    set (issue #179). Stripping is deliberately right-hand only: the indexed
    token keeps whatever the message had, so `/usr/people/eric` must keep its
    leading slash to match, while a trailing one is punctuation the writer
    added. Text indexed WITH trailing punctuation stays reachable only by a
    query carrying the same punctuation; fixing that needs a tokenizer FTS5
    will not take from Python."""
    seen: set[str] = set()
    terms: list[str] = []
    for raw in raws:
        term = raw.rstrip(TOKENCHARS)
        if not term or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def quote(term: str) -> str:
    """FTS5 treats `.`, `-` and `/` as QUERY syntax even though they are
    indexing tokenchars, so a bare `6.5.22m` raises "syntax error near .".
    Quoting makes punctuation literal; doubling embedded quotes stops a query
    escaping its own quoting."""
    doubled = term.replace('"', '""')
    return f'"{doubled}"'
