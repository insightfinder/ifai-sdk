import unittest
from insightfinder.client import LLMLabsClient
from insightfinder.auth import Auth
from dotenv import load_dotenv
import os

class TestLLMLabsClient(unittest.TestCase):

    def setUp(self):
        load_dotenv()
        self.username = os.getenv("INSIGHTFINDER_USERNAME")
        self.api_key = os.getenv("INSIGHTFINDER_API_KEY")
        self.auth = Auth()
        self.auth.set_credentials(self.username, self.api_key)
        self.client = LLMLabsClient(auth=self.auth)

    def test_chat_valid(self):
        prompt = "Hello, how are you?"
        model_version = "TinyLlama-1.1B-Chat-v1.0"
        user_created_model_name = "test2"
        model_id_type = "TinyLlama"

        response = self.client.chat(prompt, model_version, user_created_model_name, model_id_type)
        self.assertIsNotNone(response)
        self.assertTrue(hasattr(response, "response"))
        self.assertTrue(hasattr(response, "trace_id"))
        self.assertTrue(hasattr(response, "model"))
        self.assertTrue(hasattr(response, "raw_chunks"))
        self.assertTrue(hasattr(response, "evaluations"))

    def test_chat_invalid_model(self):
        prompt = "Hello, how are you?"
        model_version = "invalid-model"
        user_created_model_name = "mustafaTest1"
        model_id_type = "default"

        with self.assertRaises(ValueError):
            self.client.chat(prompt, model_version, user_created_model_name, model_id_type)

    def test_chat_empty_prompt(self):
        prompt = ""
        model_version = "TinyLlama-1.1B-Chat-v1.0"
        user_created_model_name = "mustafaTest1"
        model_id_type = "default"

        with self.assertRaises(ValueError):
            self.client.chat(prompt, model_version, user_created_model_name, model_id_type)

if __name__ == '__main__':
    unittest.main()