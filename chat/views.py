import os
import json
import openai
import tiktoken
import logging

from provider.models import ApiKey
from stats.models import TokenUsage
from .models import Conversation, Message, EmbeddingDocument, Setting, Prompt
from django.http import StreamingHttpResponse
from django.forms.models import model_to_dict
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes, action
from .serializers import ConversationSerializer, MessageSerializer, PromptSerializer, SettingSerializer
from .llm import setup_openai_env as llm_openai_env
from .llm import setup_openai_model as llm_openai_model

logger = logging.getLogger(__name__)


class SettingViewSet(viewsets.ModelViewSet):
    serializer_class = SettingSerializer

    # permission_classes = [IsAuthenticated]

    def get_queryset(self):
        available_names = [
            'open_registration',
            'open_api_key_setting',
            'open_frugal_mode_control',
        ]
        return Setting.objects.filter(name__in=available_names)

    def http_method_not_allowed(self, request, *args, **kwargs):
        if request.method != 'GET':
            return Response(status=status.HTTP_405_METHOD_NOT_ALLOWED)
        return super().http_method_not_allowed(request, *args, **kwargs)


class ConversationViewSet(viewsets.ModelViewSet):
    serializer_class = ConversationSerializer
    # authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Conversation.objects.filter(user=self.request.user).order_by('-created_at')

    @action(detail=False, methods=['delete'])
    def delete_all(self, request):
        queryset = self.filter_queryset(self.get_queryset())
        queryset.delete()
        return Response(status=204)


class MessageViewSet(viewsets.ModelViewSet):
    serializer_class = MessageSerializer
    # authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    # queryset = Message.objects.all()

    def get_queryset(self):
        queryset = Message.objects.filter(user=self.request.user).order_by('-created_at')
        conversationId = self.request.query_params.get('conversationId')
        if conversationId:
            queryset = queryset.filter(conversation_id=conversationId).order_by('created_at')
            return queryset
        return queryset


class PromptViewSet(viewsets.ModelViewSet):
    serializer_class = PromptSerializer
    # authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Prompt.objects.filter(user=self.request.user).order_by('-created_at')

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        serializer.validated_data['user'] = request.user

        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    @action(detail=False, methods=['delete'])
    def delete_all(self, request):
        queryset = self.filter_queryset(self.get_queryset())
        queryset.delete()
        return Response(status=204)


MODELS = {
    'gpt-3.5-turbo': {
        'name': 'gpt-3.5-turbo',
        'max_tokens': 4096,
        'max_prompt_tokens': 3096,
        'max_response_tokens': 1000
    },
    'gpt-4': {
        'name': 'gpt-4',
        'max_tokens': 8192,
        'max_prompt_tokens': 6192,
        'max_response_tokens': 2000
    },
    'gpt-3.5-turbo-16k': {
        'name': 'gpt-3.5-turbo-16k',
        'max_tokens': 16384,
        'max_prompt_tokens': 12384,
        'max_response_tokens': 4000
    },
    'gpt-4-32k': {
        'name': 'gpt-4-32k',
        'max_tokens': 32768,
        'max_prompt_tokens': 24768,
        'max_response_tokens': 8000
    },
    'gpt-4-1106-preview': {
        'name': 'gpt-4-1106-preview',
        'max_tokens': 131072,
        'max_prompt_tokens': 123072,
        'max_response_tokens': 8000,
    },
    'gpt-4o': {
        'name': 'gpt-4o',
        'max_tokens': 131072,
        'max_prompt_tokens': 123072,
        'max_response_tokens': 8000,
    }
}


def sse_pack(event, data):
    # Format data as an SSE message
    packet = "event: %s\n" % event
    packet += "data: %s\n" % json.dumps(data)
    packet += "\n"
    return packet


@api_view(['POST'])
# @authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def gen_title(request):
    conversation_id = request.data.get('conversationId')
    prompt = request.data.get('prompt')
    conversation_obj = Conversation.objects.get(id=conversation_id)
    message = Message.objects.filter(conversation_id=conversation_id).order_by('created_at').first()
    openai_api_key = request.data.get('openaiApiKey')
    api_key = None

    if openai_api_key is None:
        openai_api_key = get_api_key_from_setting()

    if openai_api_key is None:
        api_key = get_api_key()
        if api_key:
            openai_api_key = api_key.key
        else:
            return Response(
                {
                    'error': 'There is no available API key'
                },
                status=status.HTTP_400_BAD_REQUEST
            )

    if prompt is None:
        prompt = 'Generate a short title for the following content, no more than 10 words. \n\nContent: '

    messages = [
        {"role": "user", "content": prompt + message.message},
    ]

    my_openai = get_openai(openai_api_key)
    try:
        openai_response = my_openai.ChatCompletion.create(
            model='gpt-3.5-turbo-0301',
            messages=messages,
            max_tokens=256,
            temperature=0.5,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        completion_text = openai_response['choices'][0]['message']['content']
        title = completion_text.strip().replace('"', '')

        # increment the token count
        increase_token_usage(request.user, openai_response['usage']['total_tokens'], api_key)
    except Exception as e:
        print(e)
        title = 'Untitled Conversation'
    # update the conversation title
    conversation_obj.topic = title
    conversation_obj.save()

    return Response({
        'title': title
    })


@api_view(['POST'])
# @authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def upload_conversations(request):
    """allow user to import a list of conversations"""
    user = request.user
    import_err_msg = 'bad_import'
    conversation_ids = []
    try:
        imports = request.data.get('imports')
        # verify
        conversations = []
        for conversation in imports:
            topic = conversation.get('conversation_topic', None)
            messages = []
            for message in conversation.get('messages'):
                msg = {}
                msg['role'] = message['role']
                msg['content'] = message['content']
                messages.append(msg)
            if len(messages) > 0:
                conversations.append({
                    'topic': topic,
                    'messages': messages,
                })
        # dump
        for conversation in conversations:
            topic = conversation['topic']
            messages = conversation['messages']
            cobj = Conversation(
                topic=topic if topic else '',
                user=user,
            )
            cobj.save()
            conversation_ids.append(cobj.id)
            for idx, msg in enumerate(messages):
                try:
                    Message._meta.get_field('user')
                    mobj = Message(
                        user=user,
                        conversation=cobj,
                        message=msg['content'],
                        is_bot=msg['role'] != 'user',
                        messages=messages[:idx + 1],
                    )
                except:
                    mobj = Message(
                        conversation=cobj,
                        message=msg['content'],
                        is_bot=msg['role'] != 'user',
                        messages=messages[:idx + 1],
                    )
                mobj.save()
    except Exception as e:
        logger.debug(e)
        return Response(
            {'error': import_err_msg},
            status=status.HTTP_400_BAD_REQUEST
        )

    # return a list of new conversation id
    return Response(conversation_ids)


@api_view(['POST'])
# @authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def conversation(request):
    model_name = request.data.get('name')
    message_object_list = request.data.get('message')
    conversation_id = request.data.get('conversationId')
    request_max_response_tokens = request.data.get('max_tokens')
    system_content = request.data.get('system_content')
    if not system_content:
        system_content = "You are a helpful assistant."
    temperature = request.data.get('temperature', 0.7)
    top_p = request.data.get('top_p', 1)
    frequency_penalty = request.data.get('frequency_penalty', 0)
    presence_penalty = request.data.get('presence_penalty', 0)
    openai_api_key = request.data.get('openaiApiKey')
    frugal_mode = request.data.get('frugalMode', False)

    logger.debug('conversation_id = %s message_objects = %s', conversation_id, message_object_list)

    api_key = None

    if openai_api_key is None:
        openai_api_key = get_api_key_from_setting()

    if openai_api_key is None:
        api_key = get_api_key()
        if api_key:
            openai_api_key = api_key.key
        else:
            return Response(
                {
                    'error': 'There is no available API key'
                },
                status=status.HTTP_400_BAD_REQUEST
            )

    my_openai = get_openai(openai_api_key)
    llm_openai_env(my_openai.api_base, my_openai.api_key)

    model = get_current_model(model_name, request_max_response_tokens)
    llm_openai_model(model)

    try:
        messages = build_messages(model, conversation_id, message_object_list, system_content, frugal_mode)
        # message_object_list will be changed in build_messages

        new_doc_id = messages.get('doc_id', None)
        new_doc_title = messages.get('doc_title', None)
        logger.debug('messages: %s\n%s\n%s', messages, new_doc_id, new_doc_title)
    except Exception as e:
        print(e)
        return Response(
            {
                'error': e
            },
            status=status.HTTP_400_BAD_REQUEST
        )

    def stream_content():
        try:
            if messages['renew']:
                openai_response = my_openai.ChatCompletion.create(
                    model=model['name'],
                    messages=messages['messages'],
                    max_tokens=model['max_response_tokens'],
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stream=True,
                )
        except Exception as e:
            yield sse_pack('error', {
                'error': str(e)
            })
            print('openai error', e)
            return

        if conversation_id:
            # get the conversation
            conversation_obj = Conversation.objects.get(id=conversation_id)
        else:
            # create a new conversation
            conversation_obj = Conversation(user=request.user)
            conversation_obj.save()

        # insert new messages
        try:
            for m in message_object_list:
                message_obj = create_message(
                    user=request.user,
                    conversation_id=conversation_obj.id,
                    message=m['content'],
                    messages=messages['messages'],
                    tokens=messages['tokens'],
                    api_key=api_key
                )
                yield sse_pack('userMessageId', {
                    'userMessageId': message_obj.id,
                })
        except Exception as e:
            return Response(
                {
                    'error': e
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        collected_events = []
        completion_text = ''
        if messages['renew']:  # return LLM answer
            # iterate through the stream of events
            for event in openai_response:
                collected_events.append(event)  # save the event response
                # print(event)
                if event['choices'][0]['finish_reason'] is not None:
                    break
                if 'content' in event['choices'][0]['delta']:
                    event_text = event['choices'][0]['delta']['content']
                    completion_text += event_text  # append the text
                    yield sse_pack('message', {'content': event_text})
            bot_message_type = Message.plain_message_type
            ai_message_token = num_tokens_from_text(completion_text, model['name'])
        else:  # wait for process context
            if new_doc_title:
                completion_text = f'{new_doc_title} added.'
            else:
                completion_text = 'Context added.'
            yield sse_pack('message', {'content': completion_text})
            bot_message_type = Message.temp_message_type
            ai_message_token = 0

        ai_message_obj = create_message(
            user=request.user,
            conversation_id=conversation_obj.id,
            message=completion_text,
            message_type=bot_message_type,
            is_bot=True,
            tokens=ai_message_token,
            api_key=api_key
        )
        yield sse_pack('done', {
            'messageId': ai_message_obj.id,
            'conversationId': conversation_obj.id,
            'newDocId': new_doc_id,
        })

    response = StreamingHttpResponse(
        stream_content(),
        content_type='text/event-stream'
    )
    response['X-Accel-Buffering'] = 'no'
    response['Cache-Control'] = 'no-cache'
    return response


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def documents(request):
    pass


def create_message(user, conversation_id, message, is_bot=False, message_type=0, embedding_doc_id=None, messages='',
                   tokens=0, api_key=None):
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


def increase_token_usage(user, tokens, api_key=None):
    token_usage, created = TokenUsage.objects.get_or_create(user=user)
    token_usage.tokens += tokens
    token_usage.save()

    if api_key:
        api_key.token_used += tokens
        api_key.save()


def build_messages(model, conversation_id, new_messages, system_content, frugal_mode=False):
    if conversation_id:
        ordered_messages = Message.objects.filter(conversation_id=conversation_id).order_by('created_at')
        ordered_messages_list = list(ordered_messages)
    else:
        ordered_messages_list = []

    ordered_messages_list += [{
        'is_bot': False,
        'message': msg['content'],
    } for msg in new_messages]

    if frugal_mode:
        ordered_messages_list = ordered_messages_list[-1:]

    system_messages = [{"role": "system", "content": system_content}]

    current_token_count = num_tokens_from_messages(system_messages, model['name'])

    max_token_count = model['max_prompt_tokens']

    messages = []

    result = {
        'renew': True,
        'messages': messages,
        'tokens': 0,
    }

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

    result['messages'] = system_messages + messages
    result['tokens'] = current_token_count

    return result


def get_current_model(model_name, request_max_response_tokens):
    if model_name is None:
        model_name = "gpt-3.5-turbo"
    model = MODELS[model_name]
    if request_max_response_tokens is not None:
        model['max_response_tokens'] = int(request_max_response_tokens)
        model['max_prompt_tokens'] = model['max_tokens'] - model['max_response_tokens']
    return model


def get_api_key_from_setting():
    row = Setting.objects.filter(name='openai_api_key').first()
    if row and row.value != '':
        return row.value
    return None


def get_api_key():
    return ApiKey.objects.filter(is_enabled=True).order_by('token_used').first()


def num_tokens_from_text(text, model="gpt-3.5-turbo-0301"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    if model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"]:
        print(
            f"Warning: {model} may change over time.",
            f"Returning num tokens assuming {model}-0613."
        )
        return num_tokens_from_text(text, model=f"{model}-0613")

    if model not in [
        "gpt-3.5-turbo-0613",
        "gpt-4-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-32k-0613",
        "gpt-4-1106-preview",
        "gpt-4o"
    ]:
        raise NotImplementedError(
            f"num_tokens_from_text() is not implemented for model {model}.")

    return len(encoding.encode(text))


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    if model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"]:
        print(
            f"Warning: {model} may change over time.",
            f"Returning num tokens assuming {model}-0613."
        )
        return num_tokens_from_messages(messages, model=f"{model}-0613")

    if model in [
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-32k-0613",
        "gpt-4-1106-preview",
        "gpt-4o"
    ]:
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model in ["gpt-4-0613"]:
        tokens_per_message = 3
        tokens_per_name = 1
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


def get_openai(openai_api_key):
    openai.api_key = openai_api_key
    proxy = os.getenv('OPENAI_API_PROXY')
    if proxy:
        openai.api_base = proxy
    return openai
