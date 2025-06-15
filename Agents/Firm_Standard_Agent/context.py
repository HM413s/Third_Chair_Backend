from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime

class FirmStandardContext(BaseModel):
    """Context for tracking firm standard analysis state and results"""
    
    # Document tracking
    processed_documents: List[Dict[str, Any]] = []
    total_documents_analyzed: int = 0
    
    # Analysis results
    punctuation_issues: List[Dict[str, Any]] = []
    capitalization_problems: List[Dict[str, Any]] = []
    sentence_structure_issues: List[Dict[str, Any]] = []
    word_choice_inconsistencies: List[Dict[str, Any]] = []
    style_compliance_report: Dict[str, Any] = {}
    
    # Step completion tracking
    step_1_complete: bool = False
    step_2_complete: bool = False
    step_3_complete: bool = False
    step_4_complete: bool = False
    step_5_complete: bool = False
    step_6_complete: bool = False
    all_steps_complete: bool = False
    
    # Progress tracking
    progress: float = 0.0
    current_step: str = "Initializing"
    error: Optional[str] = None
    
    # Tool results storage
    tool_results: Dict[str, Any] = {}
    
    # Timing
    analysis_start_time: Optional[str] = None
    analysis_end_time: Optional[str] = None
    final_Output_Report : Dict[str,Any]={}
    