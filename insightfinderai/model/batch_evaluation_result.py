"""
BatchEvaluationResult model for the InsightFinder AI SDK.
"""
from typing import List, Optional, Union, Dict, Any
from .evaluation_result import EvaluationResult

class BatchEvaluationResult:
    """Represents batch evaluation results with summary statistics and object access."""
    
    def __init__(self, evaluation_results: List[EvaluationResult]):
        self.evaluations = evaluation_results
        self.response = evaluation_results  # Alias for consistency
        self.summary = self._generate_summary()
        self.is_passed = all(eval_result.is_passed for eval_result in self.evaluations)
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics for batch evaluations."""
        if not self.evaluations:
            return {
                'total_prompts': 0,
                'passed_evaluations': 0,
                'failed_evaluations': 0,
                'top_failed_evaluation': None
            }
        
        total_prompts = len(self.evaluations)
        passed_evaluations = 0
        failed_evaluations = 0
        
        # Count evaluation types across all failed prompts
        eval_type_counts = {}
        
        for eval_result in self.evaluations:
            if not eval_result.evaluations:
                # Empty evaluations = PASS
                passed_evaluations += 1
            else:
                # Has evaluations = FAIL (regardless of scores)
                failed_evaluations += 1
                
                # Count evaluation types for this failed prompt
                for eval_item in eval_result.evaluations:
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
    def prompt(self) -> List[str]:
        """Get all prompts from the batch evaluations."""
        return [eval_result.prompt for eval_result in self.evaluations if eval_result.prompt]
    
    def print(self) -> str:
        """Print and return batch evaluation results."""
        result = self.__str__()
        print(result)
        return result
    
    def __str__(self):
        """Format batch evaluation results for display."""
        result = "[Batch Evaluation Results]\n"
        result += f"Total Prompts     : {self.summary['total_prompts']}\n"
        result += f"Passed Evaluations: {self.summary['passed_evaluations']}\n"
        result += f"Failed Evaluations: {self.summary['failed_evaluations']}\n"
        
        if self.summary['top_failed_evaluation']:
            top_failed = self.summary['top_failed_evaluation']
            if isinstance(top_failed, list):
                result += f"Top Failed        : {', '.join(top_failed)}\n"
            else:
                result += f"Top Failed        : {top_failed}\n"
        
        result += "\n" + "="*60 + "\n\n"
        
        for i, eval_result in enumerate(self.evaluations, 1):
            result += f"--- Evaluation {i} ---\n"
            result += str(eval_result) + "\n\n"
        
        return result