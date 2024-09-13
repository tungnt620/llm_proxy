import json
import logging

from django.http import StreamingHttpResponse
from drf_spectacular.utils import extend_schema
from pydantic import BaseModel, ValidationError
from rest_framework import viewsets, status
from rest_framework.decorators import api_view, permission_classes, action
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response

from .llm import setup_openai_env as llm_openai_env
from .llm import setup_openai_model as llm_openai_model
from .models import Conversation, Message, Setting, Prompt
from .serializers import ConversationSerializer, MessageSerializer, PromptSerializer, SettingSerializer
from typing import List, Optional

from .utils import get_api_key_from_setting, get_openai, get_current_model, build_messages, create_message, \
    num_tokens_from_text

logger = logging.getLogger(__name__)


class SettingViewSet(viewsets.ModelViewSet):
    serializer_class = SettingSerializer

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
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        queryset = Message.objects.filter(user=self.request.user).order_by('-created_at')
        conversationId = self.request.query_params.get('conversationId')
        if conversationId:
            queryset = queryset.filter(conversation_id=conversationId).order_by('created_at')
            return queryset
        return queryset


class PromptViewSet(viewsets.ModelViewSet):
    serializer_class = PromptSerializer
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
    'gpt-4o': {
        'name': 'gpt-4o',
        'max_tokens': 144384,
        'max_prompt_tokens': 128000,
        'max_response_tokens': 16384,
    },
    'gpt-4o-mini': {
        'name': 'gpt-4o-mini',
        'max_tokens': 144384,
        'max_prompt_tokens': 128000,
        'max_response_tokens': 16384,
    }
}


def sse_pack(event, data):
    # Format data as an SSE message
    packet = "event: %s\n" % event
    packet += "data: %s\n" % json.dumps(data)
    packet += "\n"
    return packet


class ConversationRequest(BaseModel):
    name: Optional[str]
    messages: List[str]
    conversationId: Optional[int]
    max_tokens: Optional[int]
    system_content: Optional[str] = "You are a helpful assistant."
    top_p: Optional[float] = 1
    frequency_penalty: Optional[float] = 0
    presence_penalty: Optional[float] = 0
    frugalMode: Optional[bool] = False
    is_stream: Optional[bool] = False


@extend_schema(
    request=ConversationRequest
)
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def conversation(request: Request) -> Response | StreamingHttpResponse:
    """
    Handles a conversation request by interacting with the OpenAI API to generate responses.

    This function processes the incoming request to generate a conversation response using the OpenAI API.
    It supports streaming responses and handles various parameters such as model name, and penalties.

    Args:
        request (Request): The HTTP request object containing the conversation details and parameters.

    Returns:
        StreamingHttpResponse: A streaming HTTP response containing the conversation response or error messages.
    """
    try:
        # Validate request data using Pydantic
        data = ConversationRequest(**request.data)
    except ValidationError as e:
        return Response({'error': e.errors()}, status=status.HTTP_400_BAD_REQUEST)

    model_name = data.name
    raw_input_messages = data.messages
    conversation_id = data.conversationId
    request_max_response_tokens = data.max_tokens
    system_content = data.system_content
    top_p = data.top_p
    frequency_penalty = data.frequency_penalty
    presence_penalty = data.presence_penalty
    frugal_mode = data.frugalMode
    is_stream = data.is_stream

    logger.debug('conversation_id = %s raw_input_messages = %s', conversation_id, raw_input_messages)

    api_key = None

    openai_api_key = get_api_key_from_setting()

    if openai_api_key is None:
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
        messages = build_messages(model, conversation_id, raw_input_messages, system_content, frugal_mode)
        logger.debug('messages: %s', messages)
    except Exception as e:
        logger.error('Error building messages: %s', e)
        return Response(
            {
                'error': e
            },
            status=status.HTTP_400_BAD_REQUEST
        )

    def stream_content():
        try:
            if messages.renew:
                openai_response = my_openai.ChatCompletion.create(
                    model=model['name'],
                    messages=messages.messages,
                    max_completion_tokens=model['max_response_tokens'],
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stream=is_stream,
                    stream_options={"include_usage": True}
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
            for m in raw_input_messages:
                message_obj = create_message(
                    user=request.user,
                    conversation_id=conversation_obj.id,
                    message=m,
                    messages=messages.messages,
                    tokens=messages.tokens,
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
        if messages.renew:  # return LLM answer
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
        })

    response = StreamingHttpResponse(
        stream_content(),
        content_type='text/event-stream'
    )
    response['X-Accel-Buffering'] = 'no'
    response['Cache-Control'] = 'no-cache'
    return response
