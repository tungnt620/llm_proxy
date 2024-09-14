import logging

from ..models import Setting
from ..utils import get_openai, get_current_model, get_api_key_from_setting

logger = logging.getLogger(__name__)


class OpenAIService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.my_openai = get_openai(api_key)

    def configure_model(self, model_name, max_tokens):
        model = get_current_model(model_name, max_tokens)
        self.my_openai.model = model
        return model

    def send_messages(self, model, messages, is_stream, top_p=None, frequency_penalty=None, presence_penalty=None):
        params = {
            'model': model['name'],
            'messages': messages,
            'max_completion_tokens': model['max_response_tokens'],
            'top_p': top_p,
            'frequency_penalty': frequency_penalty,
            'presence_penalty': presence_penalty,
            'stream': is_stream,
            'stream_options': {"include_usage": True} if is_stream else None
        }

        logger.debug('Sending message to OpenAI, params: %s', params)
        try:
            response = self.my_openai.ChatCompletion.create(**params)
            return response
        except Exception as e:
            logger.error('Error with OpenAI request: %s', e)
            raise e
