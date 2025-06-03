import requests
import json
from types import SimpleNamespace

class LLMLabsClient:
    def __init__(self, auth):
        self.auth = auth

    def chat(self, prompt, model_version=None, user_created_model_name=None, model_id_type=None):
        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        if model_version is None:
            raise ValueError("Model version must be specified.")

        if model_id_type is None:
            raise ValueError("Model ID type must be specified.")
        
        if user_created_model_name is None:
            raise ValueError("User created model name must be specified.")

        headers = {
            'X-Api-Key': self.auth.api_key,
            'X-User-Name': self.auth.username
        }
        
        data = {
            'prompt': prompt,
            'modelVersion': model_version,
            'userCreatedModelName': user_created_model_name,
            'modelIdType': model_id_type
        }
        
        response = requests.post(
            'https://ai.insightfinder.com/api/external/v1/chatbot/stream-with-type',
            headers=headers,
            json=data,
            stream=True
        )

        if 200 <= response.status_code < 300:
            # Success, continue processing
            pass
        elif 400 <= response.status_code < 500:
            raise ValueError(f"Client error {response.status_code}: {response.text}")
        elif 500 <= response.status_code < 600:
            raise ValueError(f"Server error {response.status_code}: {response.text}")
        else:
            raise ValueError(f"Unexpected status code {response.status_code}: {response.text}")

        results = []
        stitched_response = ""
        evaluations = None
        trace_id = None
        model = None

        try:
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
                            # Handle choices
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
            print(f"Stream broken: {e}")

        return SimpleNamespace(
            response=stitched_response,
            evaluations=evaluations,
            trace_id=trace_id,
            model=model,
            raw_chunks=results
        )
