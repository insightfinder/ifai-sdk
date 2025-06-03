import requests
import json
import logging
from types import SimpleNamespace
from .config import CHATBOT_API_URL

logger = logging.getLogger(__name__)

class Client:
    """
    Client for interacting with the LLMLabs InsightFinder chatbot API.
    Handles sending prompts to the API and streaming responses.
    """

    def __init__(self, username, api_key):
        """
        Initialize the client with user credentials.

        Args:
            username (str): The username for authentication.
            api_key (str): The API key for authentication.
        """
        if not username:
            raise ValueError("Username cannot be empty.")
        if not api_key:
            raise ValueError("API key cannot be empty.")
        self.username = username
        self.api_key = api_key

    def chat(self, prompt, model_version=None, user_created_model_name=None, model_id_type=None):
        """
        Send a prompt to the chatbot API and stream the response.

        Args:
            prompt (str): The prompt to send to the chatbot.
            model_version (str): The version of the model to use.
            user_created_model_name (str): The name of the user-created model.
            model_id_type (str): The type of model ID.

        Returns:
            SimpleNamespace: An object containing the stitched response, evaluations,
                             trace ID, model, and raw response chunks.

        Raises:
            ValueError: If required parameters are missing or if the API returns an error.
        """
        # Validate required parameters
        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        if model_version is None:
            raise ValueError("Model version must be specified.")

        if model_id_type is None:
            raise ValueError("Model ID type must be specified.")
        
        if user_created_model_name is None:
            raise ValueError("User created model name must be specified.")

        # Prepare request headers for authentication
        headers = {
            'X-Api-Key': self.api_key,
            'X-User-Name': self.username
        }
        
        # Prepare request payload
        data = {
            'prompt': prompt,
            'modelVersion': model_version,
            'userCreatedModelName': user_created_model_name,
            'modelIdType': model_id_type
        }
        
        # Send POST request to the chatbot API with streaming enabled
        response = requests.post(
            CHATBOT_API_URL,
            headers=headers,
            json=data,
            stream=True
        )

        # Handle HTTP response codes
        if 200 <= response.status_code < 300:
            # Success, continue processing
            pass
        elif 400 <= response.status_code < 500:
            raise ValueError(f"Client error {response.status_code}: {response.text}")
        elif 500 <= response.status_code < 600:
            raise ValueError(f"Server error {response.status_code}: {response.text}")
        else:
            raise ValueError(f"Unexpected status code {response.status_code}: {response.text}")

        # Initialize variables for processing the streamed response
        results = []
        stitched_response = ""
        evaluations = None
        trace_id = None
        model = None

        try:
            # Iterate over streamed lines from the response
            for line in response.iter_lines(decode_unicode=True):    
                if line and line.startswith('data:'):
                    json_part = line[5:].strip()
                    if json_part and json_part != '[START]':
                        try:
                            chunk = json.loads(json_part)
                            results.append(chunk)
                            # Extract model and trace_id if present
                            if not model and "model" in chunk:
                                model = chunk["model"]
                            if "id" in chunk:
                                trace_id = chunk["id"]
                            # Handle choices in the chunk
                            if "choices" in chunk:
                                for choice in chunk["choices"]:
                                    delta = choice.get("delta", {})
                                    content = delta.get("content", "")
                                    # If content looks like an evaluation JSON, parse it
                                    if content.startswith("{") and "evaluations" in content:
                                        try:
                                            eval_obj = json.loads(content)
                                            evaluations = eval_obj.get("evaluations")
                                            # Optionally, update trace_id if present in eval_obj
                                            trace_id = eval_obj.get("traceId", trace_id)
                                        except Exception:
                                            pass
                                    else:
                                        stitched_response += content
                        except Exception:
                            pass  # ignore invalid JSON
        except requests.exceptions.ChunkedEncodingError as e:
            logger.error(f"Stream broken: {e}")

        # Return the processed response and metadata as a SimpleNamespace
        return SimpleNamespace(
            response=stitched_response,
            evaluations=evaluations,
            trace_id=trace_id,
            model=model,
            raw_chunks=results
        )
