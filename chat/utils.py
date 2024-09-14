import logging
import os

import openai
import tiktoken
from django.contrib.auth.models import User
from django.forms.models import model_to_dict
from pydantic import BaseModel

from provider.models import ApiKey
from stats.models import TokenUsage
from .config import MODELS
from .models import Message, EmbeddingDocument, Setting
from typing import List, Optional, Dict, Any, Union

logger = logging.getLogger(__name__)


def create_message(user: User,
                   conversation_id: int,
                   message: str,
                   is_bot: bool = False,
                   message_type: int = 0,
                   embedding_doc_id: Optional[int] = None,
                   messages: List[str] = [],
                   tokens: int = 0,
                   api_key: Optional[ApiKey] = None) -> Message:
    message_obj = Message(
        conversation_id=conversation_id,
        user=user,
        message=message,
        is_bot=is_bot,
        message_type=message_type,
        embedding_message_doc=EmbeddingDocument.objects.get(pk=embedding_doc_id) if embedding_doc_id else None,
        messages=messages,
        tokens=tokens,
    )
    if message_type != Message.temp_message_type:
        message_obj.save()

    increase_token_usage(user, tokens, api_key)

    return message_obj


def increase_token_usage(user: User, tokens: int, api_key: Optional[ApiKey] = None) -> None:
    token_usage, created = TokenUsage.objects.get_or_create(user=user)
    token_usage.tokens += tokens
    token_usage.save()

    if api_key:
        api_key.token_used += tokens
        api_key.save()


class BuildMessagesOutput(BaseModel):
    renew: bool
    messages: List[str]
    tokens: int


def build_messages(model: Dict[str, Union[str, int]],
                   conversation_id: Optional[int],
                   new_messages: List[str],
                   system_content: str,
                   frugal_mode: bool = False
                   ) -> BuildMessagesOutput:
    if conversation_id:
        ordered_messages = Message.objects.filter(conversation_id=conversation_id).order_by('created_at')
        ordered_messages_list = list(ordered_messages)
    else:
        ordered_messages_list = []

    ordered_messages_list += [{
        'is_bot': False,
        'message': msg,
    } for msg in new_messages]

    if frugal_mode:
        ordered_messages_list = ordered_messages_list[-1:]

    system_messages = [{"role": "system", "content": system_content}]

    current_token_count = num_tokens_from_messages(system_messages, model['name'])

    max_token_count = model['max_prompt_tokens']

    messages = []

    result: BuildMessagesOutput = BuildMessagesOutput(
        renew=True,
        messages=messages,
        tokens=0,
    )

    logger.debug('new message is: %s', new_messages)
    logger.debug('messages are: %s', ordered_messages_list)

    while current_token_count < max_token_count and len(ordered_messages_list) > 0:
        message = ordered_messages_list.pop()
        if isinstance(message, Message):
            message = model_to_dict(message)
        role = "assistant" if message['is_bot'] else "user"
        message_content = message['message']

        new_message = {"role": role, "content": message_content}
        new_token_count = num_tokens_from_messages(system_messages + messages + [new_message], model['name'])
        if new_token_count > max_token_count:
            if len(messages) > 0:
                break
            raise ValueError(
                f"Prompt is too long. Max token count is {max_token_count}, but prompt is {new_token_count} tokens long.")
        messages.insert(0, new_message)
        current_token_count = new_token_count

    result.messages = system_messages + messages
    result.tokens = current_token_count

    return result


def get_current_model(model_name: Optional[str], request_max_response_tokens: Optional[int]) -> (
        Dict)[str, Union[str, int]]:
    if model_name is None:
        model_name = "gpt-4o"
    model = MODELS[model_name]
    if request_max_response_tokens is not None:
        model['max_response_tokens'] = int(request_max_response_tokens)
        model['max_prompt_tokens'] = model['max_tokens'] - model['max_response_tokens']
    return model


def get_api_key_from_setting() -> Optional[str]:
    row = Setting.objects.filter(name='openai_api_key').first()
    if row and row.value != '':
        return row.value
    return None


def get_api_key() -> Optional[ApiKey]:
    return ApiKey.objects.filter(is_enabled=True).order_by('token_used').first()


def num_tokens_from_text(text: str, model: str = "gpt-4o-mini") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    if model not in [
        "gpt-4o",
        "gpt-4o-2024-08-06",
        "gpt-4o-mini",
    ]:
        raise NotImplementedError(
            f"num_tokens_from_text() is not implemented for model {model}.")

    return len(encoding.encode(text))


def num_tokens_from_messages(messages: List[Dict[str, str]], model: str = "gpt-4o-mini") -> int:
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    if model in [
        "gpt-4o",
        "gpt-4o-2024-08-06",
        "gpt-4o-mini",
    ]:
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    else:
        raise NotImplementedError((
            f"num_tokens_from_messages() is not implemented for model {model}. "
            "See https://github.com/openai/openai-python/blob/main/chatml.md "
            "for information on how messages are converted to tokens."
        ))

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def get_openai(openai_api_key: str) -> Any:
    openai.api_key = openai_api_key
    proxy = os.getenv('OPENAI_API_PROXY')
    if proxy:
        openai.api_base = proxy
    return openai
