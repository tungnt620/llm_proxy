import logging
import openai

logger = logging.getLogger(__name__)

openai_env = {
    'api_key': None,
    'api_base': None,
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
