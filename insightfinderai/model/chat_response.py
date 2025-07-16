"""
ChatResponse model for the InsightFinder AI SDK.
"""
from typing import List, Optional, Union, Dict, Any
from .evaluation_result import EvaluationResult


class ChatResponse:
    """Represents a chat response with formatted display and object access."""
    
    def __init__(self, response: str, prompt: Optional[Union[str, List[Dict[str, str]]]] = None, evaluations: Optional[List[dict]] = None, trace_id: Optional[str] = None, model: Optional[str] = None, raw_chunks: Optional[List] = None, enable_evaluations: bool = False, history: Optional[List[Dict[str, str]]] = None):
        self.response = response
        self.prompt = prompt
        self.history = history or []
        # Convert prompt to string for evaluation result if it's a list
        prompt_str = self._format_prompt_for_display() if isinstance(prompt, list) else prompt
        self.evaluations = EvaluationResult({'evaluations': evaluations or []}, trace_id, prompt_str, response) if evaluations else None
        self.enable_evaluations = enable_evaluations
        self.trace_id = trace_id
        self.model = model
        self.raw_chunks = raw_chunks or []
        self.is_passed = self.evaluations is None or self.evaluations.is_passed
    
    def _format_prompt_for_display(self) -> str:
        """Format conversation history for display."""
        if not isinstance(self.prompt, list):
            return str(self.prompt) if self.prompt else ""
        
        formatted = []
        for msg in self.prompt:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            formatted.append(f"[{role.upper()}] {content}")
        return "\n".join(formatted)
    
    def print(self) -> str:
        """Print and return chat response for clean, user-friendly display."""
        result = self.__str__()
        print(result)
        return result
    
    def __str__(self):
        """Format chat response for clean, user-friendly display."""
        result = "[Chat Response]\n"
        result += f"Trace ID : {self.trace_id or 'N/A'}\n"
        result += f"Model    : {self.model or 'Unknown'}\n"
        result += "\n"
        
        if self.prompt:
            result += "Prompt:\n"
            if isinstance(self.prompt, list):
                # Format conversation history nicely
                for i, msg in enumerate(self.prompt):
                    role = msg.get('role', 'unknown').upper()
                    content = msg.get('content', '')
                    result += f">> [{role}] {content}\n"
            else:
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
