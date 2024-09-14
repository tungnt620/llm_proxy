import json

from django.contrib.auth.models import User
from django.http import StreamingHttpResponse
from django.test import TestCase
from rest_framework.test import APIClient

from chat.tests.helpers import parse_sse_message


class ConversationTestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.client.force_authenticate(user=self.user)

    def test_conversation(self):
        url = '/api/conversation/'
        data = {
            "messages": ["Hello", "How are you?"],
            "conversationId": None,
            "max_tokens": 50,
            "system_content": "You are a helpful assistant.",
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "frugalMode": False,
            "is_stream": True
        }
        response: StreamingHttpResponse = self.client.post(url, data, format='json')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'text/event-stream')

        streaming_content = list(response.streaming_content)
        llm_answer = ""

        for i, chunk in enumerate(streaming_content):
            sse_data = parse_sse_message(chunk.decode())
            print(sse_data)

            if i == 0:
                self.assertEqual(sse_data.event, "userMessageId")
                self.assertEqual(json.loads(sse_data.data), {"userMessageId": 1})
            elif i == 1:
                self.assertEqual(sse_data.event, "userMessageId")
                self.assertEqual(json.loads(sse_data.data), {"userMessageId": 2})
            elif i == len(streaming_content) - 1:
                self.assertEqual(sse_data.event, "done")
                message_data = json.loads(sse_data.data)
                self.assertGreater(message_data['messageId'], 0)
                self.assertEqual(message_data['conversationId'], 1)
            else:
                self.assertEqual(sse_data.event, "message")
                message_data = json.loads(sse_data.data)
                assert 'content' in message_data
                llm_answer += message_data['content']

        print(llm_answer)
