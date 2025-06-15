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

class StyleAnalysisReport(BaseModel):
    compliance_score: float = Field(description="Overall compliance score")
    compliance_level: str = Field(description="Compliance level (e.g., Good, Excellent, Needs Improvement)")
    total_issues: int = Field(description="Total number of issues found")
    total_documents_analyzed: int = Field(description="Total number of documents analyzed")
    issue_breakdown: Dict[str, int] = Field(description="Breakdown of issues by category")
    recommendations: List[str] = Field(description="List of recommendations")
    detailed_analysis: Dict[str, List[Dict[str, Any]]] = Field(description="Detailed analysis of issues")
    summary: str = Field(description="Executive summary of the analysis")

def process_style_report(report_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process the style analysis report using LangChain and Google GenAI
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
            Format the provided style analysis report into a clean, well-structured JSON response.
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
                "compliance_score": number,
                "compliance_level": string,
                "total_issues": number,
                "total_documents_analyzed": number,
                "issue_breakdown": {{
                    "punctuation": number,
                    "capitalization": number,
                    "sentence_structure": number,
                    "word_choice": number
                }},
                "recommendations": string[],
                "detailed_analysis": {{
                    "punctuation_issues": [
                        {{
                            "description": string,
                            "severity": "critical" | "high" | "medium" | "low",
                            "context": string,
                            "location": string,
                            "recommendation": string,
                            "affected_documents": string[],
                            "additional_details": string
                        }}
                    ],
                    "capitalization_issues": [...],
                    "structure_issues": [...],
                    "word_choice_issues": [...]
                }},
                "summary": string
            }}
            
            DO NOT include any markdown formatting or additional text. Return ONLY the JSON object."""),
            ("human", "Format this style analysis report into a clean JSON structure. Use ONLY the actual content from the report: {report_data}")
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
        logger.error(f"Error processing style report: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    # Example usage
    sample_report = {
        "success": True,
        "analysis_complete": True,
        "compliance_score": 87.0,
        "compliance_level": "Good",
        "total_issues": 13,
        "report": {
            "executive_summary": {
                "compliance_score": 87.0,
                "compliance_level": "Good",
                "total_issues_found": 13,
                "total_documents_analyzed": 10,
                "issue_breakdown": {
                    "punctuation": 3,
                    "capitalization": 3,
                    "sentence_structure": 7,
                    "word_choice": 0
                }
            },
            "recommendations": [
                "Standardize punctuation usage, particularly for Oxford commas and quotation marks",
                "Ensure consistent capitalization of legal terms and proper nouns",
                "Review and revise long sentences and sentence fragments"
            ],
            "detailed_analysis": {
                "punctuation_issues": [],
                "capitalization_issues": [],
                "structure_issues": [],
                "word_choice_issues": []
            }
        }
    }
    
    result = process_style_report(sample_report)
    print(json.dumps(result, indent=2)) 