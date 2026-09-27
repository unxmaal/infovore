import sqlite3

from infovore.db.claims import related_claims
from infovore.db.exchanges import exchange_message_ids
from infovore.db.raw import (
    attachments_for_messages,
    get_channel,
    messages_by_ids,
    reactions_for_messages,
)
from infovore.extract.protocol import ExtractionRequest
from infovore.privacy.optout import opted_out_user_ids
from infovore.rows import ExchangeRow, MessageRow


def _context_messages(
    conn: sqlite3.Connection, exchange: ExchangeRow, context_size: int
) -> tuple[MessageRow, ...]:
    if exchange.parent_exchange_id is None:
        return ()
    parent_ids = exchange_message_ids(conn, exchange.parent_exchange_id)
    context_ids = parent_ids[-context_size:] if context_size > 0 else []
    return tuple(messages_by_ids(conn, context_ids))


def build_request(
    conn: sqlite3.Connection,
    exchange: ExchangeRow,
    related_limit: int = 10,
    context_size: int = 3,
) -> ExtractionRequest:
    assert exchange.id is not None
    channel = get_channel(conn, exchange.channel_id)
    channel_name = channel.name if channel is not None else str(exchange.channel_id)

    message_ids = exchange_message_ids(conn, exchange.id)
    messages = tuple(messages_by_ids(conn, message_ids))
    opted_out = opted_out_user_ids(conn)

    query_text = " ".join(
        message.content for message in messages if message.author_id not in opted_out
    )

    return ExtractionRequest(
        exchange=exchange,
        channel_name=channel_name,
        messages=messages,
        context_messages=_context_messages(conn, exchange, context_size),
        attachments=tuple(attachments_for_messages(conn, message_ids)),
        reactions=tuple(reactions_for_messages(conn, message_ids)),
        related_claims=tuple(
            related_claims(conn, query_text, related_limit, exclude_exchange_id=exchange.id)
        ),
        opted_out_user_ids=opted_out,
    )
