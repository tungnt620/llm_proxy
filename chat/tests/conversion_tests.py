import json
from django.contrib.auth.models import User
from django.http import StreamingHttpResponse
from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient
from chat.tests.helpers import parse_sse_message, ParsedSSEMessage


class ConversationTestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='tester', password='password')
        self.client.force_authenticate(user=self.user)

    def make_conversation(self, request_payload) -> list[ParsedSSEMessage]:
        response: StreamingHttpResponse = self.client.post(reverse("conversation"), request_payload, format='json')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'text/event-stream')
        return [parse_sse_message(event.decode()) for event in response.streaming_content]

    def test_conversation(self):
        payload = {
            "messages": ["Hello", "How are you?"],
            "conversationId": None,
            "max_tokens": 100,
            "system_content": "You are a helpful assistant.",
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "frugalMode": False,
            "is_stream": True
        }
        messages = self.make_conversation(payload)

        self.assertEqual(messages[0].event, "userMessageId")
        self.assertEqual(json.loads(messages[0].data), {"userMessageId": 1})

        self.assertEqual(messages[1].event, "userMessageId")
        self.assertEqual(json.loads(messages[1].data), {"userMessageId": 2})

        last_message = messages[-1]
        self.assertEqual(last_message.event, "done")
        data = json.loads(last_message.data)
        self.assertGreater(data['messageId'], 0)
        self.assertEqual(data['conversationId'], 1)

        llm_answer = [json.loads(msg.data)["content"] for msg in messages if msg.event == "message"]
        print(llm_answer)

    def test_conversation_with_two_request_in_the_same_conversation(self):
        payload = {
            "messages": [
                "Vietnam is a country with a lot tech talent, they work hard and have a good attitude. Do you know that?"],
            "conversationId": None,
            "max_tokens": 100,
            "system_content": "You are a helpful assistant. Output must bellow 100 words",
            "frugalMode": False,
            "is_stream": True
        }
        messages = self.make_conversation(payload)
        llm_answer = [json.loads(msg.data)["content"] for msg in messages if msg.event == "message"]
        print(''.join(llm_answer))

        # Second request
        payload = {
            "messages": ["More information, please"],
            "conversationId": 1,
            "max_tokens": 100,
            "system_content": "You are a helpful assistant. Output must bellow 100 words",
            "frugalMode": False,
            "is_stream": True
        }
        messages = self.make_conversation(payload)
        llm_answer = [json.loads(msg.data)["content"] for msg in messages if msg.event == "message"]
        print(''.join(llm_answer))
