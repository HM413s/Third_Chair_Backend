from agents import Agent, Runner, OpenAIChatCompletionsModel, set_default_openai_client, set_tracing_disabled, RunContextWrapper
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio
from typing import Dict, Any
import logging
from datetime import datetime
from .tools import (
    get_documents_for_style_analysis,
    analyze_punctuation_consistency,
    analyze_capitalization_patterns,
    analyze_sentence_structure_consistency,
    analyze_word_choice_consistency,
    generate_style_compliance_report
)
from .context import FirmStandardContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('style_analysis.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('StyleAnalysisAgent')

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

set_default_openai_client(client)
set_tracing_disabled(True)

# Define the Firm Standard Agent
firm_standard_agent = Agent[FirmStandardContext](
    name="Firm Standard Language & Style Compliance Agent",
    handoff_description="Execute style analysis steps in strict sequence.",
    instructions="""You are a document analysis agent. Execute these tools in EXACT sequence:

            1. First Tool - get_documents_for_style_analysis:
            - Call this tool ONCE at the start
            - Store documents in context
            - Only proceed if successful

            2. Second Tool - analyze_punctuation_consistency:
            - Use the documents ALREADY in context from step 1
            - DO NOT call get_documents_for_style_analysis again
            - Store results in context
            - Only proceed if successful

            3. Third Tool - analyze_capitalization_patterns:
            - Use the SAME documents from step 1
            - DO NOT call get_documents_for_style_analysis again
            - Store results in context
            - Only proceed if successful

            4. Fourth Tool - analyze_sentence_structure_consistency:
            - Use the SAME documents from step 1
            - DO NOT call get_documents_for_style_analysis again
            - Store results in context
            - Only proceed if successful

            5. Fifth Tool - analyze_word_choice_consistency:
            - Use the SAME documents from step 1
            - DO NOT call get_documents_for_style_analysis again
            - Store results in context
            - Only proceed if successful

            6. Final Tool - generate_style_compliance_report:
            - Use ALL results from previous steps
            - Generate final report
            - Store report in context

            CRITICAL RULES:
            0. Execute the only tool that user ask you to and store tools result in context
            1. Do NOT call get_documents_for_style_analysis more than once
            2. Use the documents from step 1 for ALL subsequent steps
            3. Stop if any tool fails
            4. If context.all_steps_complete is True, DO NOT restart the analysis
            5. If context.step_6_complete is True, DO NOT restart the analysis
            6. If final_output contains analysis_complete=True, DO NOT restart the analysis
        """,
    model=model,
    tools=[
        get_documents_for_style_analysis,
        analyze_punctuation_consistency,
        analyze_capitalization_patterns,
        analyze_sentence_structure_consistency,
        analyze_word_choice_consistency,
        generate_style_compliance_report
    ]
)

async def enhanced_automated_style_analysis() -> Dict[str, Any]:
    """Execute all six analysis steps in strict sequence"""
    try:
        # Create analysis context
        context = FirmStandardContext()
        context_wrapper = RunContextWrapper(context)
        
        # Step 1: Document Retrieval
        print("\nINFO:StyleAnalysisAgent:Step 1: Document Retrieval")
        step1_result = await Runner.run(
            firm_standard_agent,
            "Execute step 1: get_documents_for_style_analysis",
            context=context_wrapper
        )
        print("Step 1 Result Type:", type(step1_result))
        print("Step 1 Final Output:", step1_result.final_output)
        
        # if not step1_result.final_output or not step1_result.final_output.get("success", False):
        #     raise Exception("Document retrieval failed")
        
        # Step 2: Punctuation Analysis
        print("\nINFO:StyleAnalysisAgent:Step 2: Punctuation Analysis")
        step2_result = await Runner.run(
            firm_standard_agent,
            "Execute step 2: analyze_punctuation_consistency",
            context=context_wrapper
        )
        print("Step 2 Result Type:", type(step2_result))
        print("Step 2 Final Output:", step2_result.final_output)
        
        # Check if punctuation analysis was successful
        if not step2_result.final_output:
            raise Exception("Punctuation analysis failed - no output")
            
        # Step 3: Capitalization Analysis
        print("\nINFO:StyleAnalysisAgent:Step 3: Capitalization Analysis")
        step3_result = await Runner.run(
            firm_standard_agent,
            "Execute step 3: analyze_capitalization_patterns",
            context=context_wrapper
        )
        print("Step 3 Result Type:", type(step3_result))
        print("Step 3 Final Output:", step3_result.final_output)
        
        # if not step3_result.final_output or not step3_result.final_output.get("success", False):
        #     raise Exception("Capitalization analysis failed")
        
        # Step 4: Sentence Structure Analysis
        print("\nINFO:StyleAnalysisAgent:Step 4: Sentence Structure Analysis")
        step4_result = await Runner.run(
            firm_standard_agent,
            "Execute step 4: analyze_sentence_structure_consistency",
            context=context_wrapper
        )
        print("Step 4 Result Type:", type(step4_result))
        print("Step 4 Final Output:", step4_result.final_output)
        
        # if not step4_result.final_output or not step4_result.final_output.get("success", False):
        #     raise Exception("Sentence structure analysis failed")
        
        # Step 5: Word Choice Analysis
        print("\nINFO:StyleAnalysisAgent:Step 5: Word Choice Analysis")
        step5_result = await Runner.run(
            firm_standard_agent,
            "Execute step 5: analyze_word_choice_consistency",
            context=context_wrapper
        )
        print("Step 5 Result Type:", type(step5_result))
        print("Step 5 Final Output:", step5_result.final_output)
        
        # if not step5_result.final_output or not step5_result.final_output.get("success", False):
        #     raise Exception("Word choice analysis failed")
        
        # Step 6: Generate Final Report
        print("\nINFO:StyleAnalysisAgent:Step 6: Generate Final Report")
        step6_result = await Runner.run(
            firm_standard_agent,
            "Execute step 6: generate_style_compliance_report",
            context=context_wrapper
        )
        print("Step 6 Result Type:", type(step6_result))
        print("Step 6 Final Output:", step6_result.final_output)
        
        # if not step6_result.final_output or not step6_result.final_output.get("success", False):
        #     raise Exception("Report generation failed")
        
        # All steps completed successfully
        context.all_steps_complete = True
        context.step_6_complete = True
        
        # Create final output
        final_output = {
            "success": True,
            "analysis_complete": True,
            "execution_tracking": {
                "step1_complete": context.step_1_complete,
                "step2_complete": context.step_2_complete,
                "step3_complete": context.step_3_complete,
                "step4_complete": context.step_4_complete,
                "step5_complete": context.step_5_complete,
                "step6_complete": context.step_6_complete,
                "all_steps_complete": context.all_steps_complete
            },
            "analysis_timing": {
                "start_time": context.analysis_start_time,
                "end_time": datetime.now().isoformat()
            },
            "document_stats": {
                "total_documents": context.total_documents_analyzed,
                "documents_with_issues": len(context.punctuation_issues) + 
                                       len(context.capitalization_problems) + 
                                       len(context.sentence_structure_issues) + 
                                       len(context.word_choice_inconsistencies)
            },
            "final_report": step6_result.final_output
        }

        context.final_Output_Report = final_output
                
        return {
            "success": True,
            "final_output": final_output
        }
        
    except Exception as e:
        logger.error(f"Error in style analysis: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "final_output": {
                "success": False,
                "error": str(e),
                "analysis_complete": False,
                "execution_state": {
                    "steps_completed": [
                        context.step_1_complete,
                        context.step_2_complete,
                        context.step_3_complete,
                        context.step_4_complete,
                        context.step_5_complete,
                        context.step_6_complete
                    ],
                    "progress": context.progress,
                    "current_step": context.current_step,
                    "error_step": context.current_step
                }
            }
        }

if __name__ == "__main__":
    asyncio.run(enhanced_automated_style_analysis())