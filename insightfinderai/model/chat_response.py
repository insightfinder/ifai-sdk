"""
ChatResponse model for the InsightFinder AI SDK.
"""
from typing import List, Optional, Union, Dict, Any
from .evaluation_result import EvaluationResult


class ChatResponse:
    """Represents a chat response with formatted display and object access."""
    
    def __init__(self, response: str, prompt: Optional[Union[str, List[Dict[str, str]]]] = None, evaluations: Optional[List[dict]] = None, trace_id: Optional[str] = None, model: Optional[str] = None, raw_chunks: Optional[List] = None, enable_evaluations: bool = False, history: Optional[List[Dict[str, str]]] = None, project_name: Optional[str] = None, session_name: Optional[str] = None):
        self.response = response
        self.prompt = prompt
        self.history = history or []
        # Convert prompt to string for evaluation result if it's a list
        prompt_str = self._format_prompt_for_display() if isinstance(prompt, list) else prompt
        
        # Store evaluations both as EvaluationResult object and as direct list
        self._evaluation_result = EvaluationResult({'evaluations': evaluations or []}, trace_id, prompt_str, response) if evaluations else None
        self.evaluations = evaluations or []  # Direct access to evaluations list
        
        self.enable_evaluations = enable_evaluations
        self.trace_id = trace_id
        self.model = model
        self.project_name = project_name
        self.session_name = session_name
        self.model_version = self._extract_model_version(project_name, session_name) if project_name else None
        self.raw_chunks = raw_chunks or []
        self.is_passed = self._evaluation_result is None or self._evaluation_result.is_passed
    
    def _extract_model_version(self, project_name: str, session_name: Optional[str] = None) -> Optional[str]:
        """
        Extract model version from project name by removing session name and suffixes.
        Expected format: session_name-ModelVersion-llmTrace
        Example: TinyLLama-TinyLLma-v1-TinyLlama-1-1B-Chat-v1-0-llmTrace
        Returns: TinyLlama-1-1B-Chat-v1-0
        """
        if not project_name:
            return None
        
        try:
            original_project_name = project_name
            
            # Remove "-Prompt" suffix if present (added in _get_project_name)
            if project_name.endswith("-Prompt"):
                project_name = project_name[:-7]  # Remove "-Prompt"
            
            # Remove "-llmTrace" suffix if present
            if project_name.endswith("-llmTrace"):
                project_name = project_name[:-9]  # Remove "-llmTrace"
            
            # If we have the session name, remove it from the beginning
            if session_name and project_name.startswith(session_name + "-"):
                # Remove session_name and the following hyphen
                model_version = project_name[len(session_name) + 1:]
                return model_version if model_version else None
            
            # Fallback: If no session name provided or session name doesn't match
            # This should rarely happen in normal usage, but provides a safety net
            parts = project_name.split('-')
            
            if len(parts) < 2:
                return project_name  # Return as-is if can't split
            
            # Very simple fallback: assume the first part is session-related
            # and return everything else
            return '-'.join(parts[1:]) if len(parts) > 1 else None
                
        except Exception:
            return None

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
        # result += f"Trace ID      : {self.trace_id or 'N/A'}\n"  # Commented out as requested
        result += f"Model         : {self.model or 'Unknown'}\n"
        result += f"Model Version : {self.model_version or 'Unknown'}\n"
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
        if self.evaluations and self._evaluation_result:
            result += "\n" + self._evaluation_result.format_for_chat()
        elif self.enable_evaluations:
            # Show PASSED when evaluations are enabled but no evaluations were returned
            result += "\n\nEvaluations:\n"
            result += "-" * 40 + "\n"
            result += "PASSED"
        
        return result
