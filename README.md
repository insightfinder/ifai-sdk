# InsightFinder

This is a Python SDK for interfacing with the InsightFinder API. It provides a simple and efficient way to send requests to the API and handle responses.

## Features

- Authentication using a username and API key.
- Easy-to-use methods for sending requests to the InsightFinder API.
- Utility functions for request formatting and response handling.
- Access to model outputs, trace IDs, raw response chunks, and evaluation metrics.

## Installation

You can install the SDK using pip:

```
pip install insightfinder
```

## Usage

Here is a basic example of how to use the SDK:

```python
from insightfinder import LLMLabsClient

# Create a client instance
client = LLMLabsClient(
    username="your_username",  # Replace with your actual username
    api_key="your_api_key"    # Replace with your actual API key
)

# Required parameters for sending a request
prompt = "Hello, how are you?"
model_version = "TinyLlama-1.1B-Chat-v1.0"
user_created_model_name = "test2"
model_id_type = "TinyLlama"

response = client.chat(
    prompt=prompt,
    model_version=model_version,
    user_created_model_name=user_created_model_name,
    model_id_type=model_id_type
)

print(response.response)      # The generated response text
print(response.trace_id)      # Unique trace ID for the request
print(response.model)         # Model used for the response
print(response.raw_chunks)    # Raw response chunks from the API
print(response.evaluations)   # List of evaluation results (if available)
```

**Note:**  
All of the following parameters are **required** when sending a request to the API:
- `prompt`: The input text or question you want to send to the model.
- `model_version`: The version of the model you want to use (e.g., `"TinyLlama-1.1B-Chat-v1.0"`).
- `user_created_model_name`: The name of your user-created model (e.g., `"test2"`).
- `model_id_type`: The type of the model ID (e.g., `"TinyLlama"`).

**Sample Output:**
```
Response: I am fine too. How about you? Haha, I'm sorry that you have to wait multiple messages. ...
Trace ID: chatcmpl-5a5c20d6-08c9-46e2-b3dc-5e72f26db099
Model: tinyllama
Raw Chunks: [ ...list of chunk dicts... ]
Evaluations: [
  {'explanation': 'The response contains some clear signs of fabrication...', 'score': 3, 'evaluationType': 'Hallucination'},
  {'explanation': 'The response starts with a relevant answer...', 'score': 2, 'evaluationType': 'AnswerRelevance'},
  {'explanation': 'The response is mostly logical, but it contains some contradictions...', 'score': 2, 'evaluationType': 'LogicalConsistency'}
]
```

## API Details

### LLMLabsClient

- **chat(prompt, model_version, user_created_model_name=None, model_id_type=None)**:  
  Sends a request to the InsightFinder API with the specified parameters.  
  Returns a response object with the following attributes:
  - `response`: The generated text response.
  - `trace_id`: Unique identifier for the request.
  - `model`: The model used for generation.
  - `raw_chunks`: List of raw response chunks from the API.
  - `evaluations`: List of evaluation results (if available).

### Auth

- **set_credentials(username, api_key)**: Sets the username and API key for authentication.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.