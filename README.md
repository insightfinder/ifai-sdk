# InsightFinder AI SDK

A super user-friendly Python SDK for the InsightFinder AI platform. Designed for non-technical users who want powerful AI capabilities with clean, easy-to-read outputs.

## Quick Start

### Basic Setup

```python
from insightfinderai import Client

# Method 1: Provide credentials directly
client = Client(
    session_name="llm-eval-test",             # Session name - also used for project name generation
    username="your_username",                 # Your username
    api_key="your_api_key",                  # Your API key  
    enable_chat_evaluation=True               # Optional: show evaluation results (default: True)
)

# Method 2: Use environment variables for credentials
# export INSIGHTFINDER_USERNAME="your_username"
# export INSIGHTFINDER_API_KEY="your_api_key"
client = Client(
    session_name="llm-eval-test",             # Session name
    enable_chat_evaluation=False              # Clean output without evaluations
)
```

### Simple Chat

```python
# Basic chat (no conversation history)
response = client.chat("What is artificial intelligence?")

# Access response as object
print(f"Response: {response.response}")
print(f"Prompt: {response.prompt}")
print(f"Evaluations: {response.evaluations}")
print(f"History: {response.history}")

# Print formatted output (same as before)
response.print()  # or print(response)

# Chat with conversation history (like ChatGPT)
response1 = client.chat("I'm learning Python", chat_history=True)
response2 = client.chat("What are lists?", chat_history=True)  # Uses context from first message

# Conversation history array (ChatGPT API style)
conversation = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there! How can I help?"},
    {"role": "user", "content": "Tell me about AI"}
]
response = client.chat(conversation)
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

### 1. Chat with Conversation History

```python
# Basic chat without history (default behavior)
response = client.chat("Tell me about space exploration")

# Chat with conversation history enabled
client.chat("I'm interested in machine learning", chat_history=True)
client.chat("What algorithms should I start with?", chat_history=True)  # Uses previous context

# Conversation array (ChatGPT API style)
conversation = [
    {"role": "user", "content": "What's the weather like?"},
    {"role": "assistant", "content": "I don't have real-time weather data. What's your location?"},
    {"role": "user", "content": "I'm in San Francisco"}
]
response = client.chat(conversation, chat_history=True)
```

### 2. Conversation Management

```python
# Retrieve current conversation history
history = client.retrieve_chat_history()
for msg in history:
    print(f"[{msg['role'].upper()}] {msg['content']}")

# Clear conversation history
client.clear_chat_history()

# Set conversation history manually
client.set_chat_history([
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
])
```

### 3. Save & Load Conversations

```python
# Save conversation to file
saved_file = client.save_chat_history("my_conversation.json")
print(f"Saved to: {saved_file}")

# Save with auto-generated filename
auto_file = client.save_chat_history()  # Creates timestamped file

# Load conversation from file
loaded_data = client.load_chat_history("my_conversation.json")
print(f"Loaded {loaded_data['message_count']} messages")

# Restore conversation (flexible input)
client.set_chat_history(loaded_data)  # Pass full loaded data
# or
client.set_chat_history(loaded_data["conversation"])  # Pass just messages
```

### 4. Batch Chat

```python
# Process multiple questions at once (parallel - default)
prompts = [
    "What's the weather like?",
    "Tell me a joke",
    "Explain quantum physics"
]

batch_response = client.batch_chat(prompts)

# Access as object
print(f"Processed {batch_response.summary['total_chats']} chats")
for i, response in enumerate(batch_response.response):
    print(f"Response {i+1}: {response.response[:50]}...")

# Print formatted output
batch_response.print()

# Sequential processing with conversation history
conversation_prompts = [
    "Hello, my name is John",
    "What's my name?",
    "Tell me about our conversation so far"
]

# Each prompt builds on the previous response
batch_with_history = client.batch_chat(conversation_prompts, enable_history=True)

# Access conversation flow
print("Conversation flow:")
for i, response in enumerate(batch_with_history.response):
    print(f"Prompt {i+1}: {response.prompt}")
    print(f"Response: {response.response[:100]}...")
    print(f"History length: {len(response.history)}")
    print()
```

### 5. Evaluation

```python
# Evaluate any prompt-response pair
result = client.evaluate(
    prompt="What's the capital of France?",
    response="The capital of France is Paris"
)

# Access as object
print(f"Evaluations: {result.summary['total_evaluations']}")
print(f"Passed: {result.summary['passed_evaluations']}")
print(f"Failed: {result.summary['failed_evaluations']}")

# Print formatted output
result.print()  # Shows evaluation breakdown
```

### 6. Batch Evaluation

```python
# Evaluate multiple prompt and response pairs efficiently
pairs = [
    ("What's 2+2?", "4"),
    ("Capital of Japan?", "Tokyo"),
    ("Tell me about AI", "AI stands for artificial intelligence")
]

results = client.batch_evaluate(pairs)
for result in results:
    print(result)
```

### 7. Safety Evaluation

```python
# Check for PII/PHI leakage and safety issues
safety_result = client.safety_evaluation("What's your social security number?")

# Access as object
print(f"Safety evaluations: {len(safety_result.evaluations)}")
print(f"Summary: {safety_result.summary}")

# Print formatted output
safety_result.print()
```

### 8. Batch Safety Evaluation

```python
# Check multiple prompts for safety
prompts = [
    "Hello there!",
    "What's your credit card number?", 
    "Tell me your password"
]

batch_safety = client.batch_safety_evaluation(prompts)

# Access summary statistics
summary = batch_safety.summary
print(f"Total prompts: {summary['total_prompts']}")
print(f"Passed safety: {summary['passed_evaluations']}")
print(f"Failed safety: {summary['failed_evaluations']}")

# Print formatted output
batch_safety.print()
```

### Object Properties

All responses now return rich objects with properties and methods while maintaining backward compatibility:

```python
# Chat Response
response = client.chat("Hello")
response.response      # AI response text
response.prompt        # Original prompt
response.evaluations   # Evaluation object (if enabled)
response.history       # Conversation history
response.trace_id      # Unique identifier
response.model         # Model name
response.is_passed     # True if no evaluations or all passed, False otherwise
response.print()       # Formatted output
print(response)        # Same as response.print() - backward compatible!

# Evaluation Response  
result = client.evaluate("prompt", "response")
result.prompt          # Original prompt
result.response        # Response evaluated
result.evaluations     # List of evaluation details
result.summary         # Statistics: total, passed, failed, top_failed
result.is_passed       # True if no evaluations (empty = passed), False if has evaluations
result.print()         # Formatted output

# Batch Responses
batch = client.batch_chat(["Hello", "Goodbye"])
batch.response         # List of ChatResponse objects
batch.summary          # Statistics: total_chats, successful, failed
batch.is_passed        # True if all individual responses passed, False otherwise
batch.print()          # Formatted output with summary

batch_eval = client.batch_evaluate([("Q", "A")])
batch_eval.evaluations # List of EvaluationResult objects
batch_eval.summary     # Statistics: total_prompts, passed, failed, top_failed
batch_eval.is_passed   # True if all evaluations passed, False otherwise
batch_eval.print()     # Formatted output with summary
```

## Enhanced Chat Features

The SDK supports multiple ways to manage conversation history:

```python
# Mode 1: No history (default) - each chat is independent
response = client.chat("Hello")  # chat_history=False (default)

# Mode 2: Automatic history - conversations build context
client.chat("I'm learning Python", chat_history=True)
client.chat("Explain functions", chat_history=True)  # Uses Python context

# Mode 3: Manual history - full control over conversation
conversation = [
    {"role": "user", "content": "What's machine learning?"},
    {"role": "assistant", "content": "ML is a subset of AI..."},
    {"role": "user", "content": "Give me an example"}
]
response = client.chat(conversation)
```

### History Persistence

```python
# Save any conversation for later use
client.chat("Discuss quantum computing", chat_history=True)
client.chat("Explain qubits", chat_history=True)

# Save conversation
filename = client.save_chat_history("quantum_discussion.json")

# Later... load and continue
data = client.load_chat_history("quantum_discussion.json")
client.set_chat_history(data)  # Restore context
client.chat("What about quantum entanglement?", chat_history=True)  # Continues from where you left off
```

## Customization Options
```bash
# Set once, use everywhere
export INSIGHTFINDER_USERNAME="your_username"
export INSIGHTFINDER_API_KEY="your_api_key"
```

```python
# No need to provide credentials in code
client = Client(
    session_name="llm-eval-test"              # Session name - also used for project auto-generation
)
```

### Evaluation Display Control
```python
# Show evaluations and safety results in chat responses
client = Client(
    session_name="llm-eval-test",              # Session name for chat and auto project generation
    enable_chat_evaluation=True
)
response = client.chat("Hello!")
# Output includes: response + evaluations + safety results

# Hide evaluations for clean output
client = Client(
    session_name="llm-eval-test",              # Session name for chat and auto project generation
    enable_chat_evaluation=False
)
response = client.chat("Hello!")
# Output includes: response only (clean and minimal)
```

### Performance Tuning
```python
# Adjust parallel workers for batch operations
batch_responses = client.batch_chat(prompts, max_workers=5)

# Enable sequential processing with conversation history
batch_with_context = client.batch_chat(prompts, enable_history=True)

# Control streaming and history
response = client.chat(
    "Hello!",
    stream=True,           # Show real-time response
    chat_history=True      # Enable conversation context
)

# Access response properties
print(f"Response: {response.response}")
print(f"History: {len(response.history)} messages")
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

### Session Name Override

All main methods (`chat`, `batch_chat`, `evaluate`, `batch_evaluate`, `safety_evaluation`, `batch_safety_evaluation`) now support per-call session name override:

```python
# Initialize client with default session name
client = Client(
    session_name="default-session",
    username="user",
    api_key="key"
)

# Override session name for specific operations
response = client.chat("Hello", session_name="custom-chat-session")
eval_result = client.evaluate("prompt", "response", session_name="custom-eval-session")
safety_result = client.safety_evaluation("prompt", session_name="custom-safety-session")

# Batch operations with custom session names
batch_chat = client.batch_chat(prompts, session_name="custom-batch-session")
batch_eval = client.batch_evaluate(pairs, session_name="custom-eval-session")
batch_safety = client.batch_safety_evaluation(prompts, session_name="custom-safety-session")

# When session_name is provided, it's used to generate the project name for evaluations
# If session_name is not provided, the default session name from Client() is used
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
    session_name="llm-eval-test",              # Session name for chat and auto project generation
    username="user",
    api_key="key", 
    url="https://your-custom-api.com"
)
```

## Pro Tips

1. **Batch Processing**: Use batch methods for multiple requests - they're much faster!
2. **Stream Control**: Turn off streaming for batch operations to reduce noise
3. **Safety First**: Keep safety evaluation enabled for production use

## Requirements

- Python 3.7+
- `requests` library (automatically installed)

## License

This project is licensed under the MIT License.