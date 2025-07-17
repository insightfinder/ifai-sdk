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
        self.evaluation_summary = self._generate_evaluation_summary()
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
    
    def _generate_evaluation_summary(self) -> Dict[str, Any]:
        """Generate evaluation summary with same structure as EvaluationResult."""
        total_prompts = len(self.response)
        passed_evaluations = sum(1 for resp in self.response if resp.is_passed)
        failed_evaluations = total_prompts - passed_evaluations
        
        # Count evaluation types across all responses to find top failed evaluation
        eval_type_counts = {}
        for response in self.response:
            if response.evaluations:  # response.evaluations is now a list
                for eval_item in response.evaluations:
                    eval_type = eval_item.get('evaluationType', 'Unknown')
                    eval_type_counts[eval_type] = eval_type_counts.get(eval_type, 0) + 1
        
        # Find top failed evaluation type(s)
        top_failed_evaluation = None
        if eval_type_counts:
            max_count = max(eval_type_counts.values())
            top_failed_types = [eval_type for eval_type, count in eval_type_counts.items() if count == max_count]
            top_failed_evaluation = top_failed_types if len(top_failed_types) > 1 else top_failed_types[0]
        
        return {
            'total_prompts': total_prompts,
            'passed_evaluations': passed_evaluations,
            'failed_evaluations': failed_evaluations,
            'top_failed_evaluation': top_failed_evaluation
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
