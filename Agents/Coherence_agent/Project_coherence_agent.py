from agents import Agent, Runner, OpenAIChatCompletionsModel, set_default_openai_client, set_tracing_disabled, RunContextWrapper
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio
from typing import Dict, List, Any
from .tools import (
    get_all_documents_from_rag,
    check_multi_document_consistency,
    generate_consistency_report,
    analyze_contract_consistency,
    enhanced_document_search,
    deep_case_alignment_analysis,
)
from datetime import datetime
import PyPDF2
from .context import ContractAnalysisContext
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('contract_analysis.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('CoherenceAgent')

# Load environment variables
load_dotenv()

# Initialize Gemini client
gemini_api_key = os.getenv("MISTRAL_API_KEY")
if not gemini_api_key:
    raise ImportError("SET MISTRAL_API_KEY ENV")

client = AsyncOpenAI(
    api_key=os.getenv('MISTRAL_API_KEY'),  # Make sure to set this in your .env file
    base_url="https://api.mistral.ai/v1"   # Mistral AI endpoint
)

# Initialize Mistral model
model_mistral = OpenAIChatCompletionsModel(
    model='mistral-large-latest',  # Mistral's latest large model
    openai_client=client
)

def read_document(file_path: str) -> str:
    """Read document content based on file type"""
    try:
        if file_path.lower().endswith('.pdf'):
            # Handle PDF files
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        else:
            # Handle text files
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return ""

def chunk_document(content: str, max_chunk_size: int = 2000) -> List[str]:
    """Split document into smaller chunks for processing"""
    # First, split by paragraphs to maintain context
    paragraphs = content.split('\n\n')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        # Rough estimate: 1 word ≈ 1.3 tokens
        para_tokens = len(para.split()) * 1.3
        
        if current_size + para_tokens > max_chunk_size:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_size = para_tokens
        else:
            current_chunk.append(para)
            current_size += para_tokens
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

def filter_relevant_documents(documents: List[Dict[str, str]], max_docs: int = 5) -> List[Dict[str, str]]:
    """Filter and prioritize the most relevant documents"""
    # Sort by word count to prioritize substantial documents
    sorted_docs = sorted(documents, key=lambda x: len(x['content'].split()), reverse=True)
    
    # Take the top N documents
    return sorted_docs[:max_docs]

# Initialize the Clause Consistency Agent with improved instructions matching first agent pattern
clause_consistency_agent = Agent[ContractAnalysisContext](
    name="Clause Consistency Analysis Agent",
    handoff_description="Execute consistency analysis steps in strict sequence.",
    instructions="""You are a document consistency analysis agent. Execute these tools in EXACT sequence:

            1. First Tool - get_all_documents_from_rag:
            - Call this tool ONCE at the start
            - Store documents in context
            - Only proceed if successful

            2. Second Tool - check_multi_document_consistency:
            - Use the documents ALREADY in context from step 1
            - DO NOT call get_all_documents_from_rag again
            - Store results in context
            - Only proceed if successful

            3. Third Tool - enhanced_document_search:
            - Use the SAME documents from step 1
            - DO NOT call get_all_documents_from_rag again
            - Store results in context
            - Only proceed if successful

            4. Fourth Tool - analyze_contract_consistency:
            - Use the SAME documents from step 1
            - DO NOT call get_all_documents_from_rag again
            - Store results in context
            - Only proceed if successful

            5. Fifth Tool - deep_case_alignment_analysis:
            - Use the SAME documents from step 1
            - DO NOT call get_all_documents_from_rag again
            - Store results in context
            - Only proceed if successful

            6. Final Tool - generate_consistency_report:
            - Use ALL results from previous steps
            - Generate final report
            - Store report in context

        CRITICAL RULES:
        0. Execute the only tool that user ask you to and store tools result in context
            1. Do NOT call get_all_documents_from_rag more than once
        2. Use the documents from step 1 for ALL subsequent steps
        3. Stop if any tool fails
        4. If context.all_steps_complete is True, DO NOT restart the analysis
        5. If context.step_6_complete is True, DO NOT restart the analysis
        6. If final_output contains analysis_complete=True, DO NOT restart the analysis
        """,
    model=model_mistral,
    tools=[
        get_all_documents_from_rag,
        check_multi_document_consistency,
        enhanced_document_search,
        analyze_contract_consistency,
        deep_case_alignment_analysis,
        generate_consistency_report
    ]
)

async def enhanced_automated_consistency_analysis(documents: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """Execute all six analysis steps in strict sequence - matching first agent pattern"""
    try:
        # Create analysis context using the proper class
        context = ContractAnalysisContext()
        
        # Update context with uploaded documents if provided
        if documents:
            logger.info(f"Processing {len(documents)} uploaded documents")
            
            # Filter documents first
            filtered_docs = filter_relevant_documents(documents)
            logger.info(f"Filtered to {len(filtered_docs)} most relevant documents")
            
            # Process and add each document to context
            for doc in filtered_docs:
                # Split document into smaller chunks
                chunks = chunk_document(doc["content"])
                for i, chunk in enumerate(chunks):
                    doc_info = {
                        "name": f"{doc['name']}_chunk_{i+1}",
                        "content": chunk,
                        "word_count": len(chunk.split()),
                        "timestamp": datetime.now().isoformat(),
                        "original_doc": doc["name"],
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "type": "chunk"
                    }
                    context.add_document(doc_info)
            
            logger.info(f"Processed {len(context.documents)} document chunks")
        
        context_wrapper = RunContextWrapper(context)
        
        # Step 1: Document Retrieval with chunking
        print("\nINFO:CoherenceAgent:Step 1: Document Retrieval")
        step1_result = await Runner.run(
            clause_consistency_agent,
            "Execute step 1: get_all_documents_from_rag with chunking",
            context=context_wrapper
        )
        context.store_tool_result("get_all_documents_from_rag", step1_result)
        context.update_progress("Document Retrieval", 0.16)  # 1/6 steps complete
        
        # Ensure we don't exceed token limits
        if context.documents:
            total_tokens = sum(len(doc['content'].split()) * 1.3 for doc in context.documents)
            if total_tokens > 100000:  # Leave room for instructions and context
                logger.warning(f"Total tokens ({total_tokens}) exceeds safe limit, reducing document set")
                # Keep only the first few chunks of each document
                reduced_docs = []
                current_doc = None
                for doc in context.documents:
                    if current_doc != doc['original_doc']:
                        current_doc = doc['original_doc']
                        reduced_docs.append(doc)
                
                # Clear and re-add reduced documents
                context.clear_documents()
                for doc in reduced_docs:
                    context.add_document(doc)
                logger.info(f"Reduced to {len(reduced_docs)} document chunks")
        
        # Step 2: Multi-Document Consistency Analysis
        print("\nINFO:CoherenceAgent:Step 2: Multi-Document Consistency Analysis")
        try:
            step2_result = await Runner.run(
                clause_consistency_agent,
                "Execute step 2: check_multi_document_consistency",
                context=context_wrapper
            )
            context.store_tool_result("check_multi_document_consistency", step2_result)
            context.update_progress("Multi-Document Consistency Analysis", 0.33)  # 2/6 steps complete
        except Exception as e:
            logger.error(f"Error in step 2: {str(e)}")
            raise
        
        # Step 3: Enhanced Document Search
        print("\nINFO:CoherenceAgent:Step 3: Enhanced Document Search")
        step3_result = await Runner.run(
            clause_consistency_agent,
            "Execute step 3: enhanced_document_search",
            context=context_wrapper
        )
        context.store_tool_result("enhanced_document_search", step3_result)
        context.update_progress("Enhanced Document Search", 0.50)  # 3/6 steps complete
        
        # Step 4: Contract Consistency Analysis
        print("\nINFO:CoherenceAgent:Step 4: Contract Consistency Analysis")
        step4_result = await Runner.run(
            clause_consistency_agent,
            "Execute step 4: analyze_contract_consistency",
            context=context_wrapper
        )
        context.store_tool_result("analyze_contract_consistency", step4_result)
        context.update_progress("Contract Consistency Analysis", 0.66)  # 4/6 steps complete
        
        # Step 5: Deep Case Alignment Analysis
        print("\nINFO:CoherenceAgent:Step 5: Deep Case Alignment Analysis")
        step5_result = await Runner.run(
            clause_consistency_agent,
            "Execute step 5: deep_case_alignment_analysis",
            context=context_wrapper
        )
        context.store_tool_result("deep_case_alignment_analysis", step5_result)
        context.update_progress("Deep Case Alignment Analysis", 0.83)  # 5/6 steps complete
        
        # Step 6: Generate Final Report
        print("\nINFO:CoherenceAgent:Step 6: Generate Final Report")
        step6_result = await Runner.run(
            clause_consistency_agent,
            "Execute step 6: generate_consistency_report",
            context=context_wrapper
        )
        context.store_tool_result("generate_consistency_report", step6_result)
        context.update_progress("Final Report Generation", 1.0)  # 6/6 steps complete
        
        # Create final output using context summary
        context_summary = context.get_summary()
        final_output = {
            "success": True,
            "analysis_complete": True,
            "execution_tracking": {
                "step1_complete": "Document Retrieval" in context.completed_steps,
                "step2_complete": "Multi-Document Consistency Analysis" in context.completed_steps,
                "step3_complete": "Enhanced Document Search" in context.completed_steps,
                "step4_complete": "Contract Consistency Analysis" in context.completed_steps,
                "step5_complete": "Deep Case Alignment Analysis" in context.completed_steps,
                "step6_complete": "Final Report Generation" in context.completed_steps,
                "all_steps_complete": len(context.completed_steps) == 6
            },
            "analysis_timing": {
                "start_time": context.tool_results.get("get_all_documents_from_rag", {}).get("timestamp"),
                "end_time": context.tool_results.get("generate_consistency_report", {}).get("timestamp")
            },
            "document_stats": {
                "total_documents": context_summary["documents_count"],
                "documents_with_issues": context_summary["total_issues"]
            },
            "consistency_stats": {
                "total_issues": context_summary["total_issues"],
                "issues_by_severity": context_summary["issues_by_severity"],
                "issues_by_category": context_summary["issues_by_category"],
                "detailed_issues": context.detailed_issues,
                "recommendations": context.recommendations
            },
            "final_report": context.tool_results.get("generate_consistency_report", {}).get("result")
        }
                
        return {
            "success": True,
            "final_output": final_output
        }
        
    except Exception as e:
        logger.error(f"Error in consistency analysis: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "final_output": {
                "success": False,
                "error": str(e),
                "analysis_complete": False,
                "execution_state": {
                    "steps_completed": context.completed_steps,
                    "progress": context.progress,
                    "current_step": context.current_step,
                    "error_step": context.current_step
                }
            }
        }

def clause_consistency_agent_auto(documents: List[str] = None):
    """
    ONE-CLICK CONTRACT ANALYSIS - Now following first agent pattern
    
    This function performs complete automated contract consistency analysis:
    - Processes uploaded documents or extracts from RAG
    - Analyzes each document for consistency issues
    - Searches for specific content patterns
    - Checks multi-document consistency
    - Generates comprehensive report
    
    Args:
        documents (List[str], optional): List of file paths to analyze. If None, uses RAG documents.
    """
    print("🚀 ONE-CLICK CONTRACT ANALYSIS STARTING...")
    print("🔍 This will analyze the provided contract documents automatically")
    print("⏱️  Please wait while the complete analysis is performed...")
    print("\n" + "🟦" * 50)
    
    try:
        # Process documents if provided
        processed_documents = []
        if documents:
            print(f"📁 Processing {len(documents)} uploaded documents...")
            # Read and process each document
            for doc_path in documents:
                try:
                    content = read_document(doc_path)
                    if content:
                        doc_info = {
                            'name': os.path.basename(doc_path),
                            'content': content,
                            'path': doc_path,
                            'type': 'pdf' if doc_path.lower().endswith('.pdf') else 'text'
                        }
                        processed_documents.append(doc_info)
                        print(f"✅ Successfully processed: {os.path.basename(doc_path)}")
                except Exception as e:
                    print(f"⚠️ Error processing document {doc_path}: {str(e)}")
                    continue
            
            print(f"✅ Successfully processed {len(processed_documents)} documents")
        else:
            print("📁 No documents provided, using RAG documents...")
    
        # Run the enhanced automated analysis with processed documents
        result = asyncio.run(enhanced_automated_consistency_analysis(processed_documents if processed_documents else None))
        
        print(f"\nDEBUG: Final result type: {type(result)}")
        print(f"DEBUG: Final result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
        
        # Enhanced output handling matching first agent
        if result is None:
            print("❌ No final output. The analysis failed or was interrupted.")
            print("\n" + "🟥" * 50)
            print("❌ Analysis failed - Please check configuration")
            print("🟥" * 50)
            return None
        
        # Handle successful result
        if isinstance(result, dict) and result.get("success", False):
            final_output = result.get("final_output", {})
            
            print("📊 ANALYSIS RESULTS:")
            print("=" * 50)
            
            # Show execution tracking
            if "execution_tracking" in final_output:
                tracking = final_output["execution_tracking"]
                print(f"📈 Analysis Status: {'✅ Complete' if tracking.get('all_steps_complete') else '⏳ In Progress'}")
                
                steps = ["Document Retrieval", "Consistency Analysis", "Document Search", 
                        "Contract Analysis", "Case Alignment", "Report Generation"]
                step_keys = ["step1_complete", "step2_complete", "step3_complete", 
                           "step4_complete", "step5_complete", "step6_complete"]
                
                for i, (step_name, step_key) in enumerate(zip(steps, step_keys)):
                    status = "✅" if tracking.get(step_key, False) else "❌"
                    print(f"  {status} Step {i+1}: {step_name}")
            
            # Show document stats
            if "document_stats" in final_output:
                stats = final_output["document_stats"]
                print(f"\n📁 Documents: {stats.get('total_documents', 0)} processed")
                print(f"⚠️  Issues Found: {stats.get('documents_with_issues', 0)}")
            
            # Show consistency stats
            if "consistency_stats" in final_output:
                consistency = final_output["consistency_stats"]
                print(f"\n🔍 Total Issues: {consistency.get('total_issues', 0)}")
                
                if consistency.get('issues_by_severity'):
                    print("📊 Issues by Severity:")
                    for severity, count in consistency['issues_by_severity'].items():
                        print(f"  {severity}: {count}")
                
                if consistency.get('recommendations'):
                    print("💡 RECOMMENDATIONS:")
                    for rec in consistency['recommendations'][:5]:
                        print(f"  • {rec}")
                    
            # Show timing info
            if "analysis_timing" in final_output:
                timing = final_output["analysis_timing"]
                print(f"\n⏱️  Analysis completed at: {timing.get('end_time', 'Unknown')}")
            
            print("\n" + "🟩" * 50)
            print("📊 ANALYSIS COMPLETE - Report Generated Successfully!")
            print("🟩" * 50)
            
            return {"final_output": final_output, "success": True}
            
        else:
            # Handle failed result
            error_msg = result.get("error", "Unknown error occurred")
            print(f"❌ Analysis failed: {error_msg}")
            return {"error": error_msg, "success": False}
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        return {"error": str(e), "success": False}

# For backward compatibility - keeping the original main function but enhanced
def main():
    """Enhanced main function with both hardcoded and RAG analysis options"""
    print("Select Analysis Mode:")
    print("1. Automated RAG-based Analysis (Recommended)")
    print("2. Hardcoded Contract Analysis (Testing - requires at least 2 documents)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Run automated RAG-based analysis
        return clause_consistency_agent_auto()
    else:
        # Original hardcoded analysis for testing with multiple documents
        raise ValueError("Please provide at least two documents for comparison. Use the following format:\n"
                        "documents = ['path/to/document1.pdf', 'path/to/document2.pdf']\n"
                        "clause_consistency_agent_auto(documents)\n\n"
                        "Example usage:\n"
                        "from your_module import clause_consistency_agent_auto\n"
                        "documents = [\n"
                        "    '/path/to/contract1.pdf',\n"
                        "    '/path/to/contract2.pdf',\n"
                        "    '/path/to/agreement1.txt'\n"
                        "]\n"
                        "result = clause_consistency_agent_auto(documents)")

# Export the main automation function for easy access
__all__ = [
    'clause_consistency_agent_auto',  # Main one-click function
    'clause_consistency_agent',       # Agent instance
    'main'                           # Original main function
]

if __name__ == "__main__":
    # When script is run directly, execute the automated analysis
    clause_consistency_agent_auto()