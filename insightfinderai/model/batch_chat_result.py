"""
BatchChatResult model for the InsightFinder AI SDK.
"""
from typing import List, Optional, Union, Dict, Any
from .chat_response import ChatResponse


class BatchChatResult:
    """Represents batch chat results with object access."""
    
    def __init__(self, chat_responses: List[ChatResponse]):
        self.response = chat_responses
        self.evaluations = [resp.evaluations for resp in chat_responses if resp.evaluations]
        self.history = []  # Batch chat typically doesn't maintain conversation history
        self.summary = self._generate_summary()
        self.is_passed = all(resp.is_passed for resp in chat_responses)
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics for batch chat."""
        total_chats = len(self.response)
        successful_chats = sum(1 for resp in self.response if resp.response)
        
        return {
            'total_chats': total_chats,
            'successful_chats': successful_chats,
            'failed_chats': total_chats - successful_chats
        }
    
    @property
    def prompt(self) -> List[Union[str, List[Dict[str, str]]]]:
        """Get all prompts from the batch chat."""
        return [resp.prompt for resp in self.response if resp.prompt]
    
    def print(self) -> str:
        """Print and return batch chat results."""
        result = self.__str__()
        print(result)
        return result
    
    def __str__(self):
        """Format batch chat results for display."""
        result = "[Batch Chat Results]\n"
        result += f"Total Chats    : {self.summary['total_chats']}\n"
        result += f"Successful     : {self.summary['successful_chats']}\n"
        result += f"Failed         : {self.summary['failed_chats']}\n"
        result += "\n" + "="*60 + "\n\n"
        
        for i, chat_response in enumerate(self.response, 1):
            result += f"--- Response {i} ---\n"
            result += str(chat_response) + "\n\n"
        
        return result
