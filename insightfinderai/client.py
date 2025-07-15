import requests
import json
import logging
import uuid
import time
import os
from typing import List, Optional, Union, Callable, Any
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from .config import DEFAULT_API_URL, CHATBOT_ENDPOINT, EVALUATION_ENDPOINT, SAFETY_EVALUATION_ENDPOINT, TRACE_PROJECT_NAME_ENDPOINT

logger = logging.getLogger(__name__)

class EvaluationResult:
    """Represents an evaluation result with formatted display."""
    
    def __init__(self, evaluation_data: dict, trace_id: Optional[str] = None, prompt: Optional[str] = None, response: Optional[str] = None):
        self.evaluations = evaluation_data.get('evaluations', [])
        self.trace_id = trace_id or evaluation_data.get('traceId', '')
        self.prompt = prompt
        self.response = response
    
    def __str__(self):
        """Format evaluation results for clean display."""
        result = "[Evaluation Results]\n"
        result += f"Trace ID : {self.trace_id}\n"
        result += "\n"
        
        # Always show prompt and response if available
        if self.prompt:
            result += "Prompt:\n"
            result += f">> {self.prompt}\n"
            result += "\n"
        
        if self.response:
            result += "Response:\n"
            result += f">> {self.response}\n"
            result += "\n"
        
        # Show evaluations if available
        if self.evaluations:
            result += "Evaluations:\n"
            result += "-" * 40 + "\n"
            
            for i, eval_item in enumerate(self.evaluations, 1):
                eval_type = eval_item.get('evaluationType', 'Unknown')
                score = eval_item.get('score', 0)
                explanation = eval_item.get('explanation', 'No explanation provided')
                
                result += f"{i}. Type        : {eval_type}\n"
                result += f"   Score       : {score}\n"
                result += f"   Explanation : {explanation}\n"
                if i < len(self.evaluations):
                    result += "\n"
        else:
            result += "Evaluations:\n"
            result += "-" * 40 + "\n"
            result += "PASSED"
        
        return result

    def format_for_chat(self):
        """Format evaluation results for display within chat response (no prompt/response repetition)."""
        if not self.evaluations:
            return "Evaluations:\n" + "-" * 40 + "\nPASSED"
        
        result = "Evaluations:\n"
        result += "-" * 40 + "\n"
        
        for i, eval_item in enumerate(self.evaluations, 1):
            eval_type = eval_item.get('evaluationType', 'Unknown')
            score = eval_item.get('score', 0)
            explanation = eval_item.get('explanation', 'No explanation provided')
            
            result += f"{i}. Type        : {eval_type}\n"
            result += f"   Score       : {score}\n"
            result += f"   Explanation : {explanation}\n"
            if i < len(self.evaluations):
                result += "\n"
        
        return result

class ChatResponse:
    """Represents a chat response with formatted display."""
    
    def __init__(self, response: str, prompt: Optional[str] = None, evaluations: Optional[List[dict]] = None, trace_id: Optional[str] = None, model: Optional[str] = None, raw_chunks: Optional[List] = None, enable_evaluations: bool = False):
        self.response = response
        self.prompt = prompt
        self.evaluations = EvaluationResult({'evaluations': evaluations or []}, trace_id, prompt, response) if evaluations else None
        self.enable_evaluations = enable_evaluations
        self.trace_id = trace_id
        self.model = model
        self.raw_chunks = raw_chunks or []
    
    def __str__(self):
        """Format chat response for clean, user-friendly display."""
        result = "[Chat Response]\n"
        result += f"Trace ID : {self.trace_id or 'N/A'}\n"
        result += f"Model    : {self.model or 'Unknown'}\n"
        result += "\n"
        
        if self.prompt:
            result += "Prompt:\n"
            result += f">> {self.prompt}\n"
            result += "\n"
        
        result += "Response:\n"
        result += f">> {self.response}\n"
        
        # Show evaluations if they exist and enable_evaluations was enabled
        if self.evaluations and self.evaluations.evaluations:
            result += "\n" + self.evaluations.format_for_chat()
        elif self.enable_evaluations:
            # Show PASSED when evaluations are enabled but no evaluations were returned
            result += "\n\nEvaluations:\n"
            result += "-" * 40 + "\n"
            result += "PASSED"
        
        return result

class Client:
    """
    User-friendly client for InsightFinder AI SDK.
    
    This client provides easy-to-use methods for:
    - Single and batch chatting with streaming support
    - Evaluation of prompts and responses with automatic project name generation
    - Safety evaluation for prompts
    
    The client automatically generates project names for evaluations by calling the API
    with the session_name and appending "-Prompt" to the result.
    """

    def __init__(self, session_name: str, url: Optional[str] = None, username: Optional[str] = None, api_key: Optional[str] = None, enable_chat_evaluation: bool = True):
        """
        Initialize the client with user credentials and project settings.

        Args:
            session_name (str): Session name for chat requests and used to generate project name automatically
            url (str, optional): Custom API URL (defaults to https://ai.insightfinder.com)
            username (str, optional): Username for authentication (can be set via INSIGHTFINDER_USERNAME env var)
            api_key (str, optional): API key for authentication (can be set via INSIGHTFINDER_API_KEY env var)
            enable_chat_evaluation (bool): Whether to display evaluation and safety results in chat responses (default: True)
        
        Note:
            - session_name is used for both chat operations and to automatically generate the project_name for evaluation operations
            - The project name is automatically generated by calling the API with session_name and appending "-Prompt"
        
        Environment Variables:
            INSIGHTFINDER_USERNAME: Username for authentication
            INSIGHTFINDER_API_KEY: API key for authentication
        
        Example:
            # Using parameters
            client = Client(
                session_name="llm-eval-test", 
                username="john_doe",
                api_key="your_api_key_here",
                enable_chat_evaluation=True
            )
            
            # Using environment variables
            # export INSIGHTFINDER_USERNAME="john_doe"
            # export INSIGHTFINDER_API_KEY="your_api_key_here"
            client = Client(
                session_name="llm-eval-test",
                enable_chat_evaluation=True
            )
        """
        
        # Get credentials from parameters or environment variables
        self.username = username or os.getenv('INSIGHTFINDER_USERNAME')
        self.api_key = api_key or os.getenv('INSIGHTFINDER_API_KEY')
        
        if not self.username:
            raise ValueError("Username must be provided either as parameter or INSIGHTFINDER_USERNAME environment variable")
        if not self.api_key:
            raise ValueError("API key must be provided either as parameter or INSIGHTFINDER_API_KEY environment variable")
        if not session_name:
            raise ValueError("Session name cannot be empty")
        
        self.session_name = session_name
        self.enable_evaluations = enable_chat_evaluation
        
        # Set base URL with default fallback
        self.base_url = url if url else DEFAULT_API_URL
        if not self.base_url.endswith('/'):
            self.base_url += '/'
            
        # Construct API URLs
        self.chat_url = self.base_url + CHATBOT_ENDPOINT
        self.evaluation_url = self.base_url + EVALUATION_ENDPOINT  
        self.safety_url = self.base_url + SAFETY_EVALUATION_ENDPOINT
        self.trace_project_name_url = self.base_url + TRACE_PROJECT_NAME_ENDPOINT
        
        # Generate project name dynamically
        self.project_name = self._get_project_name()

    def _get_headers(self) -> dict:
        """Get authentication headers."""
        return {
            'X-Api-Key': self.api_key,
            'X-User-Name': self.username,
            'Content-Type': 'application/json'
        }

    def _generate_trace_id(self) -> str:
        """Generate a unique trace ID."""
        return str(uuid.uuid4())

    def _get_timestamp(self) -> int:
        """Get current timestamp in milliseconds."""
        return int(time.time() * 1000)

    def _get_project_name(self) -> str:
        """
        Get the project name by calling the trace project name API and appending '-Prompt'.
        
        Returns:
            str: The generated project name for evaluations
        """
        data = {
            "userCreatedModelName": self.session_name
        }
        
        try:
            response = requests.post(
                self.trace_project_name_url,
                headers=self._get_headers(),
                json=data
            )
            
            if not (200 <= response.status_code < 300):
                raise ValueError(f"Trace project name API error {response.status_code}: {response.text}")
            
            # The API returns raw text, not JSON
            trace_project_name = response.text.strip()
            
            # Append "-Prompt" to the trace project name
            return f"{trace_project_name}-Prompt"
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to get trace project name: {str(e)}")

    def chat(self, prompt: str, stream: bool = False) -> ChatResponse:
        """
        Send a single chat message and get response.
        
        Args:
            prompt (str): Your message/question
            stream (bool): Whether to show streaming response (default: False)
        
        Returns:
            ChatResponse: Response object with formatted display including evaluations (if enabled)
            
        Example:
            response = client.chat("What is the capital of France?")
            print(response)  # Clean formatted output with evaluations included
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
            
        # Prepare request data
        data = {
            'prompt': prompt,
            'userCreatedModelName': self.session_name,
        }
        
        try:
            response = requests.post(
                self.chat_url,
                headers=self._get_headers(),
                json=data,
                stream=True
            )
            
            if not (200 <= response.status_code < 300):
                raise ValueError(f"API error {response.status_code}: {response.text}")
            
            # Process streaming response
            results = []
            stitched_response = ""
            evaluations = None
            trace_id = None
            model = None
            evaluation_buffer = ""  # Buffer to accumulate evaluation JSON
            in_evaluation_block = False
            
            for line in response.iter_lines(decode_unicode=True):    
                if line and line.startswith('data:'):
                    json_part = line[5:].strip()
                    if json_part and json_part != '[START]':
                        try:
                            chunk = json.loads(json_part)
                            results.append(chunk)
                            
                            # Extract metadata
                            if not model and "model" in chunk:
                                model = chunk["model"]
                            if "id" in chunk:
                                trace_id = chunk["id"]
                                
                            # Process content
                            if "choices" in chunk:
                                for choice in chunk["choices"]:
                                    delta = choice.get("delta", {})
                                    content = delta.get("content", "")
                                    
                                    # Check if we're starting an evaluation block
                                    if content.startswith("{") and "evaluations" in content:
                                        in_evaluation_block = True
                                        evaluation_buffer = content
                                    elif in_evaluation_block:
                                        # We're in an evaluation block, accumulate content
                                        evaluation_buffer += content
                                        
                                        # Try to parse the accumulated JSON
                                        try:
                                            eval_obj = json.loads(evaluation_buffer)
                                            evaluations = eval_obj.get("evaluations")
                                            trace_id = eval_obj.get("traceId", trace_id)
                                            in_evaluation_block = False
                                            evaluation_buffer = ""
                                        except json.JSONDecodeError:
                                            # Not complete JSON yet, continue accumulating
                                            pass
                                    else:
                                        # Regular response content
                                        stitched_response += content
                                        if stream and content:
                                            print(content, end='', flush=True)
                        except:
                            pass
            
            # Create response object
            chat_response = ChatResponse(
                response=stitched_response,
                prompt=prompt,
                evaluations=evaluations if self.enable_evaluations else None,
                trace_id=trace_id,
                model=model,
                raw_chunks=results,
                enable_evaluations=self.enable_evaluations
            )
            
            return chat_response
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Request failed: {str(e)}")

    def batch_chat(self, prompts: List[str], stream: bool = False, max_workers: int = 3) -> List[ChatResponse]:
        """
        Send multiple chat messages in parallel.
        
        Args:
            prompts (List[str]): List of messages/questions
            stream (bool): Whether to show progress updates (default: False)
            max_workers (int): Number of parallel requests (default: 3)
        
        Returns:
            List[ChatResponse]: List of response objects with evaluations (if enabled)
            
        Example:
            prompts = ["Hello!", "What's the weather?", "Tell me a joke"]
            responses = client.batch_chat(prompts)
            for i, response in enumerate(responses):
                print(f"Response {i+1}: {response}")
        """
        if not prompts:
            raise ValueError("Prompts list cannot be empty")
        
        def process_single_chat(prompt_data):
            idx, prompt = prompt_data
            return idx, self.chat(prompt, stream=False)
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_prompt = {
                executor.submit(process_single_chat, (i, prompt)): i 
                for i, prompt in enumerate(prompts)
            }
            
            # Collect results in order
            results: List[Optional[ChatResponse]] = [None] * len(prompts)
            for future in as_completed(future_to_prompt):
                try:
                    idx, response = future.result()
                    results[idx] = response
                except Exception as e:
                    idx = future_to_prompt[future]
                    results[idx] = None
        
        return [r for r in results if r is not None]

    def evaluate(self, prompt: str, response: str, trace_id: Optional[str] = None) -> EvaluationResult:
        """
        Evaluate a prompt and response pair.
        
        Args:
            prompt (str): The original prompt/question
            response (str): The AI response to evaluate
            trace_id (str, optional): Custom trace ID (auto-generated if not provided)
        
        Returns:
            EvaluationResult: Evaluation results with formatted display
            
        Example:
            result = client.evaluate("What's 2+2?", "The answer is 4")
            print(result)  # Shows beautiful evaluation breakdown
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        if not response.strip():
            raise ValueError("Response cannot be empty")
            
        trace_id = trace_id or self._generate_trace_id()
        
        data = {
            "projectName": self.project_name,
            "traceId": trace_id,
            "prompt": prompt,
            "response": response,
            "timestamp": self._get_timestamp()
        }
        
        try:
            api_response = requests.post(
                self.evaluation_url,
                headers=self._get_headers(),
                json=data
            )
            
            if not (200 <= api_response.status_code < 300):
                raise ValueError(f"Evaluation API error {api_response.status_code}: {api_response.text}")
            
            result_data = api_response.json()
            return EvaluationResult(result_data, trace_id, prompt, response)
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Evaluation request failed: {str(e)}")

    def batch_evaluate(self, prompt_response_pairs: List[tuple], max_workers: int = 3) -> List[EvaluationResult]:
        """
        Evaluate multiple prompt-response pairs in parallel.
        
        Args:
            prompt_response_pairs (List[tuple]): List of (prompt, response) tuples
            max_workers (int): Number of parallel requests (default: 3)
        
        Returns:
            List[EvaluationResult]: List of evaluation results
            
        Example:
            pairs = [
                ("What's 2+2?", "4"),
                ("Capital of France?", "Paris"),
                ("Tell me a joke", "Why did the chicken cross the road?")
            ]
            results = client.batch_evaluate(pairs)
            for result in results:
                print(result)
        """
        if not prompt_response_pairs:
            raise ValueError("Prompt-response pairs list cannot be empty")
        
        def process_single_evaluation(pair_data):
            idx, (prompt, response) = pair_data
            return idx, self.evaluate(prompt, response)
        
        results: List[Optional[EvaluationResult]] = [None] * len(prompt_response_pairs)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pair = {
                executor.submit(process_single_evaluation, (i, pair)): i 
                for i, pair in enumerate(prompt_response_pairs)
            }
            
            for future in as_completed(future_to_pair):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    idx = future_to_pair[future]
                    results[idx] = None
        
        return [r for r in results if r is not None]

    def safety_evaluation(self, prompt: str, trace_id: Optional[str] = None) -> EvaluationResult:
        """
        Evaluate the safety of a prompt.
        
        Args:
            prompt (str): The prompt to evaluate for safety
            trace_id (str, optional): Custom trace ID (auto-generated if not provided)
        
        Returns:
            EvaluationResult: Safety evaluation results
            
        Example:
            result = client.safety_evaluation("What is your credit card number?")
            print(result)  # Shows safety evaluation with PII/PHI detection
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
            
        trace_id = trace_id or self._generate_trace_id()
        
        data = {
            "projectName": self.project_name,
            "traceId": trace_id,
            "prompt": prompt,
            "timestamp": self._get_timestamp()
        }
        
        try:
            api_response = requests.post(
                self.safety_url,
                headers=self._get_headers(),
                json=data
            )
            
            if not (200 <= api_response.status_code < 300):
                raise ValueError(f"Safety API error {api_response.status_code}: {api_response.text}")
            
            result_data = api_response.json()
            return EvaluationResult(result_data, trace_id, prompt, None)
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Safety evaluation request failed: {str(e)}")

    def batch_safety_evaluation(self, prompts: List[str], max_workers: int = 3) -> List[EvaluationResult]:
        """
        Evaluate the safety of multiple prompts in parallel.
        
        Args:
            prompts (List[str]): List of prompts to evaluate
            max_workers (int): Number of parallel requests (default: 3)
        
        Returns:
            List[EvaluationResult]: List of safety evaluation results
            
        Example:
            prompts = ["Hello", "What's your SSN?", "Tell me about AI"]
            results = client.batch_safety_evaluation(prompts)
            for result in results:
                print(result)
        """
        if not prompts:
            raise ValueError("Prompts list cannot be empty")
        
        def process_single_safety(prompt_data):
            idx, prompt = prompt_data
            return idx, self.safety_evaluation(prompt)
        
        results: List[Optional[EvaluationResult]] = [None] * len(prompts)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_prompt = {
                executor.submit(process_single_safety, (i, prompt)): i 
                for i, prompt in enumerate(prompts)
            }
            
            for future in as_completed(future_to_prompt):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    idx = future_to_prompt[future]
                    results[idx] = None
        
        return [r for r in results if r is not None]
