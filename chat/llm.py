import logging
import openai

logger = logging.getLogger(__name__)

openai_env = {
    'api_key': None,
    'api_base': None,
}

openai_model = {
    'name': 'gpt-3.5-turbo',
    'max_tokens': 4096,
    'max_prompt_tokens': 3096,
    'max_response_tokens': 1000
}


def setup_openai_env(api_base=None, api_key=None):
    if not openai_env['api_base']:
        openai_env['api_base'] = api_base
    if not openai_env['api_key']:
        openai_env['api_key'] = api_key
    openai.api_base = openai_env['api_base']
    openai.api_key = openai_env['api_key']
    openai.api_version = None
    return (openai_env['api_base'], openai_env['api_key'])


def setup_openai_model(model):
    logger.debug(model)
    openai_model.update(model)
    logger.debug(model)
