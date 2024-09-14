import unittest

from chat.tests.helpers import parse_sse_message


class TestSSEMessageParser(unittest.TestCase):

    def test_parse_sse_message(self):
        sse_message = """
        id: 123
        event: customEvent
        data: This is a custom event message
        data: It has multiple lines

        retry: 5000
        """

        # Parse the SSE message
        parsed_message = parse_sse_message(sse_message)

        # Assert parsed values
        self.assertEqual(parsed_message['id'], '123')
        self.assertEqual(parsed_message['event'], 'customEvent')
        self.assertEqual(parsed_message['data'], 'This is a custom event message\nIt has multiple lines')
        self.assertEqual(parsed_message['retry'], 5000)

    def test_parse_sse_message_with_no_event(self):
        sse_message = """
        data: Hello, world!
        """

        # Parse the SSE message
        parsed_message = parse_sse_message(sse_message)

        # Assert parsed values
        self.assertIsNone(parsed_message['id'])
        self.assertIsNone(parsed_message['event'])
        self.assertEqual(parsed_message['data'], 'Hello, world!')
        self.assertIsNone(parsed_message['retry'])
