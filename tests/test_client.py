import unittest
from insightfinderai.client import Client
from dotenv import load_dotenv
import os

class TestClient(unittest.TestCase):
    """
    Unit tests for the Client class.
    Tests the chat functionality with various scenarios.
    """

    def setUp(self):
        """
        Set up the test environment by loading environment variables and initializing the client.
        """
        load_dotenv()
        self.username = os.getenv("INSIGHTFINDER_USERNAME")
        self.api_key = os.getenv("INSIGHTFINDER_API_KEY")
        self.client = Client(self.username, self.api_key)

    def test_chat_valid(self):
        """
        Test the chat method with valid parameters.
        Asserts that the response contains expected attributes.
        """
        prompt = "Hello, how are you?"
        model_version = "TinyLlama-1.1B-Chat-v1.0"
        user_created_model_name = "test2"
        model_id_type = "TinyLlama"

        response = self.client.chat(prompt, model_version, user_created_model_name, model_id_type)
        # Check that the response object is not None and has required attributes
        self.assertIsNotNone(response)
        self.assertTrue(hasattr(response, "response"))
        self.assertTrue(hasattr(response, "trace_id"))
        self.assertTrue(hasattr(response, "model"))
        self.assertTrue(hasattr(response, "raw_chunks"))
        self.assertTrue(hasattr(response, "evaluations"))

    def test_chat_invalid_model(self):
        """
        Test the chat method with an invalid model version.
        Expects a ValueError to be raised.
        """
        prompt = "Hello, how are you?"
        model_version = "invalid-model"
        user_created_model_name = "test1"
        model_id_type = "default"

        # Expect ValueError for invalid model version
        with self.assertRaises(ValueError):
            self.client.chat(prompt, model_version, user_created_model_name, model_id_type)

    def test_chat_empty_prompt(self):
        """
        Test the chat method with an empty prompt.
        Expects a ValueError to be raised.
        """
        prompt = ""
        model_version = "TinyLlama-1.1B-Chat-v1.0"
        user_created_model_name = "test1"
        model_id_type = "default"

        # Expect ValueError for empty prompt
        with self.assertRaises(ValueError):
            self.client.chat(prompt, model_version, user_created_model_name, model_id_type)

if __name__ == '__main__':
    unittest.main()