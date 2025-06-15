from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import json
import os
from dotenv import load_dotenv
from datetime import datetime
import logging

import re

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class CoherenceAnalysisReport(BaseModel):
    consistency_score: float = Field(description="Overall consistency score")
    total_issues: int = Field(description="Total number of issues found")
    documents_analyzed: int = Field(description="Total number of documents analyzed")
    issues_by_severity: Dict[str, int] = Field(description="Breakdown of issues by severity")
    issues_by_category: Dict[str, int] = Field(description="Breakdown of issues by category")
    recommendations: List[str] = Field(description="List of recommendations")
    detailed_issues: List[Dict[str, Any]] = Field(description="Detailed analysis of issues")
    summary: str = Field(description="Executive summary of the analysis")

def process_coherence_report(report_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process the coherence analysis report using LangChain and Google GenAI
    """
    try:
        # Initialize the LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.1
        )
        
        # Create a prompt template that enforces JSON output
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert document analysis assistant. 
            Format the provided coherence analysis report into a clean, well-structured JSON response.
            IMPORTANT RULES:
            1. Use ONLY the actual content from the provided report_data
            2. DO NOT generate or add any new issues or recommendations
            3. Keep all original issues exactly as they are
            4. Keep all original recommendations exactly as they are
            5. Only format and structure the data for better display
            6. Ensure all fields are properly formatted for frontend display
            7. Maintain the exact severity levels and categories from the original data
            8. Do not modify any issue descriptions or recommendations
            9. Do not add any example or hypothetical content
            
            Return a JSON response with this structure:
            {{
                "consistency_score": number,
                "total_issues": number,
                "documents_analyzed": number,
                "issues_by_severity": {{
                    "critical": number,
                    "high": number,
                    "medium": number,
                    "low": number,
                    "info": number
                }},
                "issues_by_category": {{
                    "case_alignment": number,
                    "cross_references": number,
                    "defined_terms": number,
                    "structural": number,
                    "numbering": number
                }},
                "recommendations": string[],
                "detailed_issues": [
                    {{
                        "id": string,
                        "category": string,
                        "description": string,
                        "status": "success" | "error" | "warning",
                        "severity": "critical" | "high" | "medium" | "low" | "info",
                        "location": {{
                            "document": string,
                            "line": number
                        }},
                        "details": string,
                        "confidence_score": number,
                        "suggested_fix": string,
                        "affected_documents": string[]
                    }}
                ],
                "summary": string
            }}
            
            DO NOT include any markdown formatting or additional text. Return ONLY the JSON object."""),
            ("human", "Format this coherence analysis report into a clean JSON structure. Use ONLY the actual content from the report: {report_data}")
        ])

        # Format the report data for the prompt
        formatted_report = json.dumps(report_data, indent=2)
        
        # Generate the analysis
        chain = prompt | llm
        
        # Process the report
        result = chain.invoke({"report_data": formatted_report})
        
        # Parse the JSON response
        try:
            processed_report = json.loads(result.content)
        except json.JSONDecodeError:
            # If the response isn't valid JSON, try to extract JSON from the text
            json_match = re.search(r'\{.*\}', result.content, re.DOTALL)
            if json_match:
                processed_report = json.loads(json_match.group())
            else:
                raise ValueError("Could not extract valid JSON from LLM response")

        # Add metadata
        processed_report["metadata"] = {
            "processed_at": datetime.now().isoformat(),
            "total_documents": report_data.get("document_stats", {}).get("total_documents"),
            "documents_with_issues": report_data.get("document_stats", {}).get("documents_with_issues")
        }

        # Log the processed report for debugging
        logger.info(f"Processed report structure: {json.dumps(processed_report, indent=2)}")

        return {
            "success": True,
            "processed_report": processed_report
        }

    except Exception as e:
        logger.error(f"Error processing coherence report: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    # Example usage
    sample_report = {
        "success": True,
        "analysis_complete": True,
        "consistency_score": 85.0,
        "total_issues": 10,
        "documents_analyzed": 5,
        "issues_by_severity": {
            "critical": 2,
            "high": 3,
            "medium": 3,
            "low": 2,
            "info": 0
        },
        "issues_by_category": {
            "case_alignment": 3,
            "cross_references": 2,
            "defined_terms": 2,
            "structural": 2,
            "numbering": 1
        },
        "recommendations": [
            "Ensure consistent case references across all documents",
            "Standardize cross-reference formatting",
            "Maintain consistent defined terms usage"
        ],
        "detailed_issues": []
    }
    
    result = process_coherence_report(sample_report)
    print(json.dumps(result, indent=2)) 