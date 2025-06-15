from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ContractAnalysisContext(BaseModel):
    """Context for contract analysis operations with improved structure alignment"""
    
    # Document storage
    documents: List[Dict[str, Any]] = Field(default_factory=list)
    processed_documents: List[Dict[str, Any]] = Field(default_factory=list)
    contract_text: str = ""
    
    # Analysis results with consistent naming
    analysis_results: List[Dict[str, Any]] = Field(default_factory=list)
    multi_document_analysis_results: Dict[str, Any] = Field(default_factory=lambda: {
        "issues": [],
        "consistency_score": 0,
        "total_issues": 0,
        "issues_by_severity": {},
        "issues_by_category": {},
        "detailed_issues": [],
        "recommendations": []
    })
    single_document_analysis_results: Dict[str, Any] = Field(default_factory=dict)
    comprehensive_report: Dict[str, Any] = Field(default_factory=dict)
    
    # Tool execution tracking
    tool_results: Dict[str, Any] = Field(default_factory=dict)
    tool_execution_order: List[str] = Field(default_factory=list)
    
    # Step tracking
    step_1_complete: bool = False
    step_2_complete: bool = False
    step_3_complete: bool = False
    step_4_complete: bool = False
    step_5_complete: bool = False
    step_6_complete: bool = False
    all_steps_complete: bool = False
    
    # Issue tracking
    total_issues: int = 0
    issues_by_severity: Dict[str, int] = Field(default_factory=dict)
    issues_by_category: Dict[str, int] = Field(default_factory=dict)
    detailed_issues: List[Dict[str, Any]] = Field(default_factory=list)
    detailed_inconsistencies: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Recommendations and insights
    recommendations: List[str] = Field(default_factory=list)
    
    # Progress tracking
    progress: float = 0.0
    current_step: str = ""
    completed_steps: List[str] = Field(default_factory=list)
    
    # Error handling
    error: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Analysis timing
    analysis_start_time: str = Field(default_factory=lambda: datetime.now().isoformat())
    total_documents_analyzed: int = 0

    def add_document(self, document: Dict[str, Any]):
        """Add a document to the context with validation"""
        if not document.get("content"):
            raise ValueError("Document must have content")
        
        doc_info = {
            "name": document.get("name", f"document_{len(self.documents)+1}"),
            "content": document["content"],
            "word_count": len(document["content"].split()),
            "timestamp": datetime.now().isoformat(),
            "type": document.get("type", "unknown")
        }
        
        self.documents.append(doc_info)
        self.processed_documents.append(doc_info)
        self.contract_text += document["content"] + "\n\n"
        self.total_documents_analyzed += 1
        
        logger.info(f"Added document: {doc_info['name']} ({doc_info['word_count']} words)")

    def clear_documents(self):
        """Clear all documents and reset context"""
        self.documents = []
        self.processed_documents = []
        self.contract_text = ""
        self.total_documents_analyzed = 0
        logger.info("Documents cleared from context")

    def store_tool_result(self, tool_name: str, result: Any):
        """Store the result of a specific tool execution with enhanced error handling"""
        try:
            # Handle RunResult object
            if hasattr(result, 'final_output'):
                result_data = result.final_output
            elif hasattr(result, 'raw_result'):
                result_data = result.raw_result
            else:
                result_data = result

            # Store the result with timestamp
            self.tool_results[tool_name] = {
                "result": result_data,
                "timestamp": datetime.now().isoformat(),
                "success": True if result_data else False
            }
            
            # Track execution order
            if tool_name not in self.tool_execution_order:
                self.tool_execution_order.append(tool_name)
            
            # Update step completion status
            if tool_name == "get_all_documents_from_rag":
                self.step_1_complete = True
                self.completed_steps.append("Document Retrieval")
            elif tool_name == "check_multi_document_consistency":
                self.step_2_complete = True
                self.completed_steps.append("Multi-Document Consistency Analysis")
            elif tool_name == "enhanced_document_search":
                self.step_3_complete = True
                self.completed_steps.append("Enhanced Document Search")
            elif tool_name == "analyze_contract_consistency":
                self.step_4_complete = True
                self.completed_steps.append("Contract Consistency Analysis")
            elif tool_name == "deep_case_alignment_analysis":
                self.step_5_complete = True
                self.completed_steps.append("Deep Case Alignment Analysis")
            elif tool_name == "generate_consistency_report":
                self.step_6_complete = True
                self.completed_steps.append("Final Report Generation")
                self.all_steps_complete = True
            
            # Update analysis results if available
            if isinstance(result_data, dict):
                self._update_context_from_result(tool_name, result_data)
            
            logger.info(f"Successfully stored result for tool: {tool_name}")
            
        except Exception as e:
            error_msg = f"Error storing tool result for {tool_name}: {str(e)}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            self.tool_results[tool_name] = {
                "result": str(result),
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": str(e)
            }

    def _update_context_from_result(self, tool_name: str, result_data: Dict[str, Any]):
        """Update context fields based on tool results with proper field mapping"""
        try:
            if tool_name == "check_multi_document_consistency":
                if "consistency_analysis" in result_data:
                    self.multi_document_analysis_results.update(result_data["consistency_analysis"])
                if "analysis_results" in result_data:
                    self.multi_document_analysis_results["detailed_analysis"] = result_data["analysis_results"]
                if "issues" in result_data:
                    self.detailed_inconsistencies.extend(result_data["issues"])
                if "detailed_inconsistencies" in result_data:
                    self.detailed_inconsistencies.extend(result_data["detailed_inconsistencies"])
            
            elif tool_name == "analyze_contract_consistency":
                if "issues" in result_data:
                    self.multi_document_analysis_results["issues"] = result_data["issues"]
                if "consistency_score" in result_data:
                    self.multi_document_analysis_results["consistency_score"] = result_data["consistency_score"]
                if "total_issues" in result_data:
                    self.multi_document_analysis_results["total_issues"] = result_data["total_issues"]
                if "issues_by_severity" in result_data:
                    self.multi_document_analysis_results["issues_by_severity"] = result_data["issues_by_severity"]
                if "issues_by_category" in result_data:
                    self.multi_document_analysis_results["issues_by_category"] = result_data["issues_by_category"]
                if "detailed_issues" in result_data:
                    self.multi_document_analysis_results["detailed_issues"] = result_data["detailed_issues"]
                if "recommendations" in result_data:
                    self.multi_document_analysis_results["recommendations"] = result_data["recommendations"]
            
            # Common fields for all tools
            if "analysis_results" in result_data:
                if isinstance(result_data["analysis_results"], list):
                    self.analysis_results.extend(result_data["analysis_results"])
                else:
                    self.analysis_results.append(result_data["analysis_results"])
            
            if "issues" in result_data:
                self.detailed_issues.extend(result_data["issues"])
                self.total_issues += len(result_data["issues"])
            
            if "recommendations" in result_data:
                if isinstance(result_data["recommendations"], list):
                    self.recommendations.extend(result_data["recommendations"])
                else:
                    self.recommendations.append(result_data["recommendations"])
            
            if "issues_by_severity" in result_data:
                for severity, count in result_data["issues_by_severity"].items():
                    self.issues_by_severity[severity] = self.issues_by_severity.get(severity, 0) + count
            
            if "issues_by_category" in result_data:
                for category, count in result_data["issues_by_category"].items():
                    self.issues_by_category[category] = self.issues_by_category.get(category, 0) + count
            
        except Exception as e:
            error_msg = f"Error updating context from {tool_name} result: {str(e)}"
            logger.error(error_msg)
            self.warnings.append(error_msg)

    def update_progress(self, step: str, progress: float):
        """Update the current progress of the analysis"""
        self.current_step = step
        self.progress = progress
        
        if step not in self.completed_steps:
            self.completed_steps.append(step)
        
        logger.info(f"Progress updated: {step} ({progress*100:.1f}%)")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current context state"""
        return {
            "documents_count": len(self.documents),
            "processed_documents_count": len(self.processed_documents),
            "total_issues": self.total_issues,
            "issues_by_severity": self.issues_by_severity,
            "issues_by_category": self.issues_by_category,
            "recommendations_count": len(self.recommendations),
            "progress": self.progress,
            "current_step": self.current_step,
            "completed_steps": self.completed_steps,
            "tools_executed": self.tool_execution_order,
            "errors_count": len(self.errors),
            "warnings_count": len(self.warnings),
            "has_error": bool(self.error)
        }