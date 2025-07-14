# InsightFinder AI SDK

A super user-friendly Python SDK for the InsightFinder AI platform. Designed for non-technical users who want powerful AI capabilities with clean, easy-to-read outputs.

## Features

- **Simple Setup**: Just provide your credentials and project name
- **Chat & Streaming**: Single chat with real-time streaming responses
- **Batch Processing**: Handle multiple requests efficiently 
- **Smart Evaluations**: Automatic bias, hallucination, and relevance analysis
- **Safety Checks**: Built-in PII/PHI detection and safety evaluation
- **Clean Output**: Formatted results with clear structure
- **Parallel Processing**: Fast batch operations with customizable workers

### Requirements Overview
- **For Chat Operations**: `session_name` is required
- **For Evaluation Operations**: `project_name` is required
- **For Both**: Both parameters needed if using chat and evaluation features together

## Installation

```bash
pip install insightfinderai
```

## Quick Start

### Basic Setup

```python
from insightfinderai import Client

# Method 1: Provide credentials directly
client = Client(
    project_name="my_ai_project",     # Required: Project name for evaluations
    session_name="chat_session_1",    # Required: Session name for chat operations
    username="your_username",         # Your username
    api_key="your_api_key",          # Your API key  
    enable_evaluations=True           # Optional: show evaluation results (default: False)
)

# Method 2: Use environment variables for credentials
# export INSIGHTFINDER_USERNAME="your_username"
# export INSIGHTFINDER_API_KEY="your_api_key"
client = Client(
    project_name="my_ai_project",     # Required for evaluations
    session_name="my_chat_session",   # Required for chat operations
    enable_evaluations=False          # Clean output without evaluations
)
```

### Simple Chat

```python
# Chat with clean formatted output including prompt
response = client.chat("What is artificial intelligence?")
print(response)  # Clean formatted output with prompt included
```

**Output:**
```
[Chat Response]
Trace ID : abc-123-def
Model    : tinyllama

Prompt:
>> What is artificial intelligence?

Response:
>> Artificial intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence, such as visual perception, speech recognition, decision-making, and language translation.

Evaluations:
----------------------------------------
1. Type        : AnswerRelevance
   Score       : 4
   Explanation : The response directly addresses the question about AI

2. Type        : Hallucination
   Score       : 5
   Explanation : The response contains accurate information
```

## All Features

### 1. Single Chat

```python
# With streaming disabled (default)
response = client.chat("Tell me about space exploration")

# With streaming enabled
response = client.chat("Hello!", stream=True)

# Skip safety evaluation
response = client.chat("What's 2+2?", include_safety=False)
```

### 2. Batch Chat

```python
# Process multiple questions at once
prompts = [
    "What's the weather like?",
    "Tell me a joke",
    "Explain quantum physics"
]

responses = client.batch_chat(prompts)
for i, response in enumerate(responses, 1):
    print(f"Response {i}: {response}")
```

### 3. Evaluation

```python
# Evaluate any prompt-response pair
result = client.evaluate(
    prompt="What's the capital of France?",
    response="The capital of France is Paris"
)
print(result)  # Shows bias, hallucination, relevance scores
```

### 4. Batch Evaluation

```python
# Evaluate multiple pairs efficiently
pairs = [
    ("What's 2+2?", "4"),
    ("Capital of Japan?", "Tokyo"),
    ("Tell me about AI", "AI stands for artificial intelligence")
]

results = client.batch_evaluate(pairs)
for result in results:
    print(result)
```

### 5. Safety Evaluation

```python
# Check for PII/PHI leakage and safety issues
safety_result = client.safety_evaluation("What's your social security number?")
print(safety_result)
```

### 6. Batch Safety Evaluation

```python
# Check multiple prompts for safety
prompts = [
    "Hello there!",
    "What's your credit card number?", 
    "Tell me your password"
]

safety_results = client.batch_safety_evaluation(prompts)
for result in safety_results:
    print(result)
```

## Customization Options

### Environment Variables
```bash
# Set once, use everywhere
export INSIGHTFINDER_USERNAME="your_username"
export INSIGHTFINDER_API_KEY="your_api_key"
```

```python
# No need to provide credentials in code
client = Client(
    project_name="my_project",        # Required for evaluations
    session_name="my_session"         # Required for chat operations
)
```

### Evaluation Display Control
```python
# Show evaluations and safety results in chat responses
client = Client(
    project_name="project_for_evals",    # Required for evaluations
    session_name="session_for_chat",     # Required for chat
    enable_evaluations=True
)
response = client.chat("Hello!")
# Output includes: response + evaluations + safety results

# Hide evaluations for clean output
client = Client(
    project_name="project_for_evals",    # Required for evaluations
    session_name="session_for_chat",     # Required for chat
    enable_evaluations=False
)
response = client.chat("Hello!")
# Output includes: response only (clean and minimal)
```

### Performance Tuning
```python
# Adjust parallel workers for batch operations
responses = client.batch_chat(prompts, max_workers=5)

# Control streaming and safety checks
response = client.chat(
    "Hello!",
    stream=True,        # Show real-time response
    include_safety=True # Run safety evaluation
)
```

### Custom Trace IDs
```python
# Use your own trace IDs for tracking
result = client.evaluate(
    prompt="Test question",
    response="Test answer", 
    trace_id="my-custom-trace-123"
)
```

## Understanding Evaluations

The SDK automatically evaluates responses for:

- **Answer Relevance**: How well the response answers the question
- **Hallucination**: Whether the response contains false information  
- **Logical Consistency**: How logical and coherent the response is
- **Bias**: Detection of potential bias in responses
- **PII/PHI Leakage**: Safety check for sensitive information exposure

Each evaluation includes:
- **Score**: Raw numeric score (as returned by the API)
- **Explanation**: Clear description of the evaluation
- **Type**: Category of evaluation performed

## Error Handling

The SDK provides clear, user-friendly error messages:

```python
try:
    response = client.chat("")  # Empty prompt
except ValueError as e:
    print(e)  # "Prompt cannot be empty"
```

## Custom API URL

```python
# Use a custom API endpoint
client = Client(
    project_name="project",           # Required for evaluations
    session_name="session",           # Required for chat operations
    username="user",
    api_key="key", 
    url="https://your-custom-api.com"
)
```

## Pro Tips

1. **Batch Processing**: Use batch methods for multiple requests - they're much faster!
2. **Stream Control**: Turn off streaming for batch operations to reduce noise
3. **Safety First**: Keep safety evaluation enabled for production use
4. **Project Organization**: Use descriptive project and session names for better tracking

## Requirements

- Python 3.7+
- `requests` library (automatically installed)

## License

This project is licensed under the MIT License.
