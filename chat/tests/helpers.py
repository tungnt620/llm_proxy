from pydantic import BaseModel
from typing import Optional


class ParsedSSEMessage(BaseModel):
    id: Optional[str]
    event: Optional[str]
    data: str
    retry: Optional[int]


def parse_sse_message(sse_message: str) -> ParsedSSEMessage:
    """
    Parse an SSE message string into a dictionary with fields `id`, `event`, `data`, `retry`.
    """
    parsed_message = {
        'id': None,
        'event': None,
        'data': '',
        'retry': None,
    }

    # Split the message by newlines
    lines = sse_message.strip().splitlines()

    # Process each line
    for line in [raw_line.strip() for raw_line in lines]:
        if line.startswith('id:'):
            parsed_message['id'] = line[len('id:'):].strip()
        elif line.startswith('event:'):
            parsed_message['event'] = line[len('event:'):].strip()
        elif line.startswith('data:'):
            if parsed_message['data']:
                parsed_message['data'] += '\n'
            parsed_message['data'] += line[len('data:'):].strip()
        elif line.startswith('retry:'):
            parsed_message['retry'] = int(line[len('retry:'):].strip())

    return ParsedSSEMessage(**parsed_message)
