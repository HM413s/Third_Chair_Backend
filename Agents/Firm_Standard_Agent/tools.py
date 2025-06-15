from agents import function_tool, RunContextWrapper
from typing import Dict, Any, List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from RAG import rag_service
from Agents.Firm_Standard_Agent.context import FirmStandardContext
import re
import logging
from functools import wraps
from fastapi import WebSocket
from websocket_manager import active_connections, send_websocket_message
from datetime import datetime

# Simple logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rag_service = rag_service.RAGService()

@function_tool(strict_mode=False)
async def get_documents_for_style_analysis(context: RunContextWrapper[FirmStandardContext]) -> Dict[str, Any]:
    """Get documents for style analysis"""
    try:
        # Get all available documents
        stats = rag_service.get_storage_stats()
        documents = []
        
        for source, source_stats in stats.get("sources", {}).items():
            docs = rag_service.get_documents_by_source(source)
            if docs:
                documents.extend(docs)
        
        if not documents:
            return {"success": False, "error": "No documents found for analysis"}
        
        # Update context properly
        context.context.total_documents_analyzed = len(documents)
        context.context.processed_documents = documents
        
        # Update progress and completion
        context.context.progress = 0.167
        context.context.current_step = "Document Retrieval & Setup"
        context.context.step_1_complete = True
        
        print(f"📁 Found {len(documents)} documents for analysis")
        return {
            "success": True,
            "documents": documents,
            "total_documents": len(documents)
        }
        
    except Exception as e:
        logger.error(f"Error in document retrieval: {str(e)}")
        return {"success": False, "error": str(e)}

@function_tool(strict_mode=False)
def analyze_punctuation_consistency(context: RunContextWrapper[FirmStandardContext]) -> Dict[str, Any]:
    """Analyze punctuation consistency across documents"""
    try:
        # Get documents from context
        documents = context.context.processed_documents
        if not documents:
            print("⚠️ No documents found in context")
            return {
                "success": False, 
                "error": "No documents available for analysis",
                "final_output": {
                    "success": False,
                    "error": "No documents available for analysis"
                }
            }
            
        print(f"🔍 Starting punctuation analysis on {len(documents)} documents...")
        
        punctuation_issues = []
        
        for i, doc in enumerate(documents):
            try:
                doc_issues = []
                content = doc.get("content", "")
                
                if not content or not isinstance(content, str):
                    print(f"⚠️  Skipping document {i+1}: No valid content")
                    continue
                
                print(f"📝 Analyzing document {i+1}: {doc.get('source_file', 'unknown')[:50]}...")
                
                # Check for Oxford comma patterns - more robust approach
                try:
                    # Look for series without Oxford comma: "A, B and C"
                    oxford_missing = re.findall(r'\w+,\s+\w+\s+and\s+\w+', content)
                    # Look for series with Oxford comma: "A, B, and C"  
                    oxford_present = re.findall(r'\w+,\s+\w+,\s+and\s+\w+', content)
                    
                    if oxford_missing and oxford_present:
                        doc_issues.append({
                            "category": "oxford_comma",
                            "inconsistency_type": "mixed_usage",
                            "details": f"Found {len(oxford_missing)} instances without Oxford comma and {len(oxford_present)} with Oxford comma",
                            "examples": {
                                "without_oxford": oxford_missing[:3],  # First 3 examples
                                "with_oxford": oxford_present[:3]
                            }
                        })
                except Exception as regex_error:
                    print(f"⚠️  Oxford comma check failed: {regex_error}")
                
                # Check for quotation mark consistency - simplified
                try:
                    straight_quotes = len(re.findall(r'"[^"]*"', content))
                    curly_quotes = len(re.findall(r'[""][^""]*[""]', content))
                    
                    if straight_quotes > 0 and curly_quotes > 0:
                        doc_issues.append({
                            "category": "quotation_marks",
                            "inconsistency_type": "mixed_quote_styles",
                            "details": f"Found {straight_quotes} straight quotes and {curly_quotes} curly quotes",
                            "recommendation": "Use consistent quotation mark style throughout document"
                        })
                except Exception as regex_error:
                    print(f"⚠️  Quote check failed: {regex_error}")
                
                # Check for semicolon vs comma consistency in lists
                try:
                    semicolon_lists = len(re.findall(r'\w+;\s+\w+;\s+', content))
                    comma_lists = len(re.findall(r'\w+,\s+\w+,\s+', content))
                    
                    if semicolon_lists > 0 and comma_lists > semicolon_lists * 2:
                        doc_issues.append({
                            "category": "list_punctuation",
                            "inconsistency_type": "mixed_list_separators",
                            "details": f"Found {semicolon_lists} semicolon lists and {comma_lists} comma lists",
                            "recommendation": "Consider consistent use of semicolons for complex lists"
                        })
                except Exception as regex_error:
                    print(f"⚠️  List punctuation check failed: {regex_error}")
                
                if doc_issues:
                    punctuation_issues.append({
                        "document_name": doc.get("source_file", f"document_{i+1}"),
                        "issues": doc_issues
                    })
                    print(f"🔴 Found {len(doc_issues)} punctuation issues in document {i+1}")
                else:
                    print(f"✅ No punctuation issues in document {i+1}")
                    
            except Exception as doc_error:
                print(f"⚠️  Error processing document {i+1}: {doc_error}")
                continue
        
        # Update context with results
        context.context.punctuation_issues = punctuation_issues
        context.context.step_2_complete = True
        
        # Store tool result
        if not hasattr(context.context, 'tool_results'):
            context.context.tool_results = {}
        context.context.tool_results["analyze_punctuation_consistency"] = {
            "success": True,
            "total_issues": len(punctuation_issues),
            "issues_by_document": punctuation_issues
        }
        
        total_issues = len(punctuation_issues)
        if total_issues == 0:
            print("✅ STEP 2/6: No punctuation inconsistencies found")
        else:
            print(f"🔴 STEP 2/6: Found {total_issues} documents with punctuation issues")
        
        # Create the final output structure
        final_output = {
            "success": True,
            "total_issues_found": total_issues,
            "punctuation_issues": punctuation_issues,
            "summary": f"Found {total_issues} documents with punctuation issues" if total_issues > 0 else "No punctuation issues found"
        }
        
        return {
            "success": True,
            "total_issues_found": total_issues,
            "punctuation_issues": punctuation_issues,
            "final_output": final_output
        }
        
    except Exception as e:
        error_msg = f"Error in punctuation analysis: {str(e)}"
        logger.error(error_msg)
        print(f"❌ CRITICAL ERROR in punctuation analysis: {str(e)}")
        return {
            "success": False, 
            "error": str(e),
            "final_output": {
                "success": False,
                "error": str(e)
            }
        }

@function_tool(strict_mode=False)
async def analyze_capitalization_patterns(context: RunContextWrapper[FirmStandardContext]) -> Dict[str, Any]:
    """Analyze capitalization patterns and consistency"""
    try:
        # Get documents from context
        documents = context.context.processed_documents
        if not documents:
            print("⚠️ No documents found in context")
            return {
                "success": False, 
                "error": "No documents available for analysis",
                "final_output": {
                    "success": False,
                    "error": "No documents available for analysis"
                }
            }
        
        # Broadcast start of analysis
        await send_websocket_message(
            "Starting capitalization analysis...",
            "Capitalization Analysis",
            45
        )
        
        print(f"🔍 Starting capitalization analysis on {len(documents)} documents...")
        capitalization_problems = []
        
        for i, doc in enumerate(documents):
            try:
                doc_problems = []
                content = doc.get("content", "")
                
                if not content or not isinstance(content, str):
                    print(f"⚠️  Skipping document {i+1}: No valid content")
                    continue
                
                print(f"📝 Analyzing document {i+1}: {doc.get('source_file', 'unknown')[:50]}...")
                
                # Legal terms that should be consistently capitalized
                legal_terms = ['agreement', 'contract', 'party', 'parties', 'client', 'section', 'clause', 'exhibit']
                
                for term in legal_terms:
                    # Find all variations of the term
                    variations = set()
                    for match in re.finditer(r'\b' + re.escape(term) + r'\b', content, re.IGNORECASE):
                        variations.add(match.group())
                    
                    if len(variations) > 1:
                        # Found inconsistent capitalization
                        doc_problems.append({
                            "issue_type": "inconsistent_legal_term",
                            "term": term,
                            "variations_found": list(variations),
                            "locations": [match.start() for match in re.finditer(r'\b' + re.escape(term) + r'\b', content, re.IGNORECASE)]
                        })
                
                if doc_problems:
                    capitalization_problems.append({
                        "document_name": doc.get("source_file", f"document_{i+1}"),
                        "capitalization_problems": doc_problems
                    })
                    print(f"🔴 Found {len(doc_problems)} capitalization issues in document {i+1}")
                else:
                    print(f"✅ No capitalization issues in document {i+1}")
                    
            except Exception as doc_error:
                print(f"⚠️  Error processing document {i+1}: {doc_error}")
                continue
        
        # Update context
        context.context.capitalization_problems = capitalization_problems
        context.context.step_3_complete = True
        
        # Store tool result directly in context
        context.context.tool_results["analyze_capitalization_patterns"] = {
            "success": True,
            "total_issues": len(capitalization_problems),
            "issues_by_document": capitalization_problems
        }
        
        total_issues = len(capitalization_problems)
        if total_issues == 0:
            await send_websocket_message(
                "✅ No capitalization issues found",
                "Capitalization Analysis",
                50
            )
            print("✅ STEP 3/6: No capitalization issues found")
        else:
            await send_websocket_message(
                f"🔴 Found {total_issues} documents with capitalization issues",
                "Capitalization Analysis",
                50
            )
            print(f"🔴 STEP 3/6: Found {total_issues} documents with capitalization issues")
        
        # Create the final output structure
        final_output = {
            "success": True,
            "total_issues_found": total_issues,
            "capitalization_problems": capitalization_problems,
            "summary": f"Found {total_issues} documents with capitalization issues" if total_issues > 0 else "No capitalization issues found"
        }
        
        return {
            "success": True,
            "total_issues_found": total_issues,
            "capitalization_problems": capitalization_problems,
            "final_output": final_output
        }
        
    except Exception as e:
        error_msg = f"Error in capitalization analysis: {str(e)}"
        logger.error(error_msg)
        await send_websocket_message(
            f"Error in capitalization analysis: {str(e)}",
            "Error Handler",
            0
        )
        return {
            "success": False, 
            "error": str(e),
            "final_output": {
                "success": False,
                "error": str(e)
            }
        }

@function_tool(strict_mode=False)
def analyze_sentence_structure_consistency(context: RunContextWrapper[FirmStandardContext]) -> Dict[str, Any]:
    """Analyze sentence structure consistency"""
    try:
        # Get documents from context
        documents = context.context.processed_documents
        if not documents:
            print("⚠️ No documents found in context")
            return {
                "success": False, 
                "error": "No documents available for analysis",
                "final_output": {
                    "success": False,
                    "error": "No documents available for analysis"
                }
            }
            
        print(f"🔍 Starting sentence structure analysis on {len(documents)} documents...")
        sentence_structure_issues = []
        
        for i, doc in enumerate(documents):
            try:
                content = doc.get("content", "")
                
                if not content or not isinstance(content, str):
                    print(f"⚠️  Skipping document {i+1}: No valid content")
                    continue
                
                print(f"📝 Analyzing document {i+1}: {doc.get('source_file', 'unknown')[:50]}...")
                
                sentences = re.split(r'[.!?]+', content)
                
                # Analyze sentence structure
                total_sentences = len(sentences)
                total_words = sum(len(sent.split()) for sent in sentences)
                avg_length = total_words / total_sentences if total_sentences > 0 else 0
                
                # Find potential issues
                structure_issues_list = []
                
                # Check for long sentences (>40 words)
                for j, sent in enumerate(sentences):
                    words = sent.split()
                    if len(words) > 40:
                        structure_issues_list.append({
                            "type": "long_sentence",
                            "sentence_index": j,
                            "word_count": len(words),
                            "sentence": sent.strip()
                        })
                
                # Check for sentence fragments
                for j, sent in enumerate(sentences):
                    if not sent.strip():
                        continue
                    if not any(c.isupper() for c in sent.strip()):
                        structure_issues_list.append({
                            "type": "sentence_fragment",
                            "sentence_index": j,
                            "sentence": sent.strip()
                        })
                
                if structure_issues_list:
                    sentence_structure_issues.append({
                        "document_name": doc.get("source_file", f"document_{i+1}"),
                        "sentence_analysis": {
                            "total_sentences": total_sentences,
                            "avg_sentence_length": avg_length,
                            "structure_issues": structure_issues_list
                        }
                    })
                    print(f"🔴 Found {len(structure_issues_list)} sentence structure issues in document {i+1}")
                else:
                    print(f"✅ No sentence structure issues in document {i+1}")
                    
            except Exception as doc_error:
                print(f"⚠️  Error processing document {i+1}: {doc_error}")
                continue
        
        # Update context
        context.context.sentence_structure_issues = sentence_structure_issues
        context.context.step_4_complete = True
        
        # Store tool result
        context.context.tool_results["analyze_sentence_structure_consistency"] = {
            "success": True,
            "total_issues": len(sentence_structure_issues),
            "issues_by_document": sentence_structure_issues
        }
        
        total_issues = len(sentence_structure_issues)
        if total_issues == 0:
            print("✅ STEP 4/6: No sentence structure issues found")
        else:
            print(f"🔴 STEP 4/6: Found {total_issues} documents with sentence structure issues")
        
        # Create the final output structure
        final_output = {
            "success": True,
            "total_issues_found": total_issues,
            "structure_issues": sentence_structure_issues,
            "summary": f"Found {total_issues} documents with sentence structure issues" if total_issues > 0 else "No sentence structure issues found"
        }
        
        return {
            "success": True,
            "total_issues_found": total_issues,
            "structure_issues": sentence_structure_issues,
            "final_output": final_output
        }
        
    except Exception as e:
        error_msg = f"Error in sentence structure analysis: {str(e)}"
        logger.error(error_msg)
        print(f"❌ CRITICAL ERROR in sentence structure analysis: {str(e)}")
        return {
            "success": False, 
            "error": str(e),
            "final_output": {
                "success": False,
                "error": str(e)
            }
        }

@function_tool(strict_mode=False)
def analyze_word_choice_consistency(context: RunContextWrapper[FirmStandardContext]) -> Dict[str, Any]:
    """Analyze word choice consistency"""
    try:
        # Get documents from context
        documents = context.context.processed_documents
        if not documents:
            print("⚠️ No documents found in context")
            return {
                "success": False, 
                "error": "No documents available for analysis",
                "final_output": {
                    "success": False,
                    "error": "No documents available for analysis"
                }
            }
            
        print(f"🔍 Starting word choice analysis on {len(documents)} documents...")
        word_choice_inconsistencies = []
        
        for i, doc in enumerate(documents):
            try:
                doc_issues = []
                content = doc.get("content", "")
                
                if not content or not isinstance(content, str):
                    print(f"⚠️  Skipping document {i+1}: No valid content")
                    continue
                
                print(f"📝 Analyzing document {i+1}: {doc.get('source_file', 'unknown')[:50]}...")
                
                # Define preferred terms and their variations
                preferred_terms = {
                    "shall": ["must", "will", "should"],
                    "party": ["parties", "side", "sides"],
                    "agreement": ["contract", "deal", "arrangement"],
                    "terminate": ["end", "stop", "cease"],
                    "pursuant to": ["under", "in accordance with", "as per"]
                }
                
                for preferred, variations in preferred_terms.items():
                    # Find all variations
                    found_variations = {}
                    for term in [preferred] + variations:
                        matches = list(re.finditer(r'\b' + re.escape(term) + r'\b', content, re.IGNORECASE))
                        if matches:
                            found_variations[term] = [match.start() for match in matches]
                    
                    if len(found_variations) > 1:
                        # Found inconsistent usage
                        doc_issues.append({
                            "category": "word_choice",
                            "preferred_term": preferred,
                            "found_variations": list(found_variations.keys()),
                            "locations": found_variations
                        })
                
                if doc_issues:
                    word_choice_inconsistencies.append({
                        "document_name": doc.get("source_file", f"document_{i+1}"),
                        "terminology_inconsistencies": doc_issues
                    })
                    print(f"🔴 Found {len(doc_issues)} word choice issues in document {i+1}")
                else:
                    print(f"✅ No word choice issues in document {i+1}")
                    
            except Exception as doc_error:
                print(f"⚠️  Error processing document {i+1}: {doc_error}")
                continue
        
        # Update context
        context.context.word_choice_inconsistencies = word_choice_inconsistencies
        context.context.step_5_complete = True
        
        # Store tool result
        context.context.tool_results["analyze_word_choice_consistency"] = {
            "success": True,
            "total_issues": len(word_choice_inconsistencies),
            "issues_by_document": word_choice_inconsistencies
        }
        
        total_issues = len(word_choice_inconsistencies)
        if total_issues == 0:
            print("✅ STEP 5/6: No word choice inconsistencies found")
        else:
            print(f"🔴 STEP 5/6: Found {total_issues} documents with word choice issues")
        
        # Create the final output structure
        final_output = {
            "success": True,
            "total_issues_found": total_issues,
            "word_choice_issues": word_choice_inconsistencies,
            "summary": f"Found {total_issues} documents with word choice issues" if total_issues > 0 else "No word choice issues found"
        }
        
        return {
            "success": True,
            "total_issues_found": total_issues,
            "word_choice_issues": word_choice_inconsistencies,
            "final_output": final_output
        }
        
    except Exception as e:
        error_msg = f"Error in word choice analysis: {str(e)}"
        logger.error(error_msg)
        print(f"❌ CRITICAL ERROR in word choice analysis: {str(e)}")
        return {
            "success": False, 
            "error": str(e),
            "final_output": {
                "success": False,
                "error": str(e)
            }
        }

@function_tool(strict_mode=False)
def generate_style_compliance_report(context: RunContextWrapper[FirmStandardContext]) -> Dict[str, Any]:
    """Generate comprehensive style compliance report"""
    try:
        # Get all analysis results from context
        punctuation_issues = getattr(context.context, 'punctuation_issues', [])
        capitalization_issues = getattr(context.context, 'capitalization_problems', [])
        structure_issues = getattr(context.context, 'sentence_structure_issues', [])
        word_choice_issues = getattr(context.context, 'word_choice_inconsistencies', [])
        
        # Calculate issue counts correctly
        total_punctuation = sum(len(doc.get("issues", [])) for doc in punctuation_issues)
        total_capitalization = sum(len(doc.get("capitalization_problems", [])) for doc in capitalization_issues)
        total_structure = sum(len(doc.get("sentence_analysis", {}).get("structure_issues", [])) for doc in structure_issues)
        total_word_choice = sum(len(doc.get("terminology_inconsistencies", [])) for doc in word_choice_issues)
        
        total_issues = total_punctuation + total_capitalization + total_structure + total_word_choice
        
        # Calculate compliance score (0-100)
        total_documents = len(getattr(context.context, 'processed_documents', []))
        if total_documents > 0:
            issues_per_doc = total_issues / total_documents
            compliance_score = max(0, 100 - (issues_per_doc * 10))  # Each issue reduces score by 10 points
        else:
            compliance_score = 0
        
        # Determine compliance level
        if compliance_score >= 90:
            compliance_level = "Excellent"
        elif compliance_score >= 75:
            compliance_level = "Good"
        elif compliance_score >= 60:
            compliance_level = "Fair"
        else:
            compliance_level = "Needs Improvement"
        
        # Generate recommendations
        recommendations = []
        if total_punctuation > 0:
            recommendations.append("Standardize punctuation usage, particularly for Oxford commas and quotation marks")
        if total_capitalization > 0:
            recommendations.append("Ensure consistent capitalization of legal terms and proper nouns")
        if total_structure > 0:
            recommendations.append("Review and revise long sentences and sentence fragments")
        if total_word_choice > 0:
            recommendations.append("Standardize terminology usage across documents")
        
        # Create executive summary
        executive_summary = {
            "compliance_score": round(compliance_score, 1),
            "compliance_level": compliance_level,
            "total_issues_found": total_issues,
            "total_documents_analyzed": total_documents,
            "issue_breakdown": {
                "punctuation": total_punctuation,
                "capitalization": total_capitalization,
                "sentence_structure": total_structure,
                "word_choice": total_word_choice
            }
        }
        
        # Compile final report
        report = {
            "executive_summary": executive_summary,
            "recommendations": recommendations,
            "detailed_analysis": {
                "punctuation_issues": punctuation_issues,
                "capitalization_issues": capitalization_issues,
                "structure_issues": structure_issues,
                "word_choice_issues": word_choice_issues
            }
        }
        
        # Update context
        context.context.style_compliance_report = report
        context.context.step_6_complete = True
        context.context.all_steps_complete = True
        
        # Store tool result
        context.context.tool_results["generate_style_compliance_report"] = {
            "success": True,
            "report": report
        }
        
        print(f"📊 Final Compliance Score: {compliance_score:.1f}% ({compliance_level})")
        print(f"📋 Total Issues Found: {total_issues}")
        print(f"📄 Documents Analyzed: {total_documents}")
        
        # Create the final output structure
        final_output = {
            "success": True,
            "analysis_complete": True,
            "compliance_score": compliance_score,
            "compliance_level": compliance_level,
            "total_issues": total_issues,
            "report": report,
            "summary": f"Style compliance report generated. Analysis complete.",
            "execution_tracking": {
                "step1_complete": getattr(context.context, 'step_1_complete', False),
                "step2_complete": getattr(context.context, 'step_2_complete', False),
                "step3_complete": getattr(context.context, 'step_3_complete', False),
                "step4_complete": getattr(context.context, 'step_4_complete', False),
                "step5_complete": getattr(context.context, 'step_5_complete', False),
                "step6_complete": getattr(context.context, 'step_6_complete', False),
                "all_steps_complete": getattr(context.context, 'all_steps_complete', False)
            },
            "analysis_timing": {
                "start_time": getattr(context.context, 'analysis_start_time', None),
                "end_time": datetime.now().isoformat()
            },
            "document_stats": {
                "total_documents": total_documents,
                "documents_with_issues": len(punctuation_issues) + 
                                       len(capitalization_issues) + 
                                       len(structure_issues) + 
                                       len(word_choice_issues)
            }
        }

        print(f"Final Output: {final_output}")
        
        return {
            "success": True,
            "analysis_complete": True,
            "compliance_score": compliance_score,
            "compliance_level": compliance_level,
            "total_issues": total_issues,
            "report": report,
            "final_output": final_output
        }
        
    except Exception as e:
        error_msg = f"Error in report generation: {str(e)}"
        logger.error(error_msg)
        print(f"❌ CRITICAL ERROR in report generation: {str(e)}")
        return {
            "success": False, 
            "error": str(e),
            "final_output": {
                "success": False,
                "error": str(e),
                "analysis_complete": False
            }
        }

@function_tool(strict_mode=False)
def get_multi_document_comparison(context: RunContextWrapper[FirmStandardContext], document_sources: List[str] = None) -> Dict[str, Any]:
    """Get documents for cross-document comparison analysis"""
    try:
        if not document_sources:
            # Get all available documents grouped by source
            stats = rag_service.get_storage_stats()
            document_sources = list(stats.get("sources", {}).keys())
        
        if len(document_sources) < 2:
            return {
                "success": False,
                "error": "At least 2 documents are required for comparison",
                "available_sources": document_sources
            }
        
        # Get documents for each source
        comparison_data = []
        for source in document_sources:
            docs = rag_service.search_by_location({"source_file": source}, limit=100)
            if docs["text_docs"]:
                comparison_data.append({
                    "source": source,
                    "documents": docs["text_docs"],
                    "total_chunks": len(docs["text_docs"])
                })
        
        # Perform initial comparison analysis
        comparison_result = rag_service.compare_documents(comparison_data, "style and formatting analysis")
        
        # Update context
        if context and hasattr(context, "context"):
            context.context.comparison_data = comparison_data
            context.context.comparison_result = comparison_result
            context.context.multi_document_analysis_complete = True
        
        print(f"📊 Multi-document comparison prepared: {len(comparison_data)} documents")
        print(f"📄 Documents: {[data['source'] for data in comparison_data]}")
        
        return {
            "success": True,
            "documents_to_compare": len(comparison_data),
            "document_sources": [data["source"] for data in comparison_data],
            "total_chunks": sum(data["total_chunks"] for data in comparison_data),
            "comparison_analysis": comparison_result
        }
        
    except Exception as e:
        logger.error(f"Error in multi-document comparison: {str(e)}")
        return {"success": False, "error": str(e)}

@function_tool(strict_mode=False)

def analyze_cross_document_terminology(context: RunContextWrapper[FirmStandardContext]) -> Dict[str, Any]:
    """Analyze terminology consistency across multiple documents"""
    try:
        comparison_data = getattr(context.context, 'comparison_data', [])
        if not comparison_data:
            return {"success": False, "error": "No comparison data available. Run get_multi_document_comparison first."}
        
        # Legal terms to analyze
        legal_terms = ['agreement', 'contract', 'party', 'parties', 'client', 'section', 'clause', 'exhibit', 'shall', 'will', 'must']
        
        terminology_inconsistencies = []
        
        for term in legal_terms:
            term_usage_by_document = {}
            
            for doc_data in comparison_data:
                source = doc_data["source"]
                chunks = doc_data["documents"]
                
                variations_found = set()
                locations = []
                
                for chunk in chunks:
                    content = chunk.get("content", "").lower()
                    location = chunk.get("location", {})
                    
                    if term in content:
                        # Find exact variations of the term
                        pattern = rf'\b{re.escape(term)}\w*\b'
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        for match in matches:
                            variations_found.add(match)
                            locations.append({
                                "variation": match,
                                "file_name": source,
                                "line_number": location.get("line_number", 0),
                                "section_path": location.get("section_path", ""),
                                "content_preview": chunk["content"][:100] + "..." if len(chunk["content"]) > 100 else chunk["content"]
                            })
                
                if variations_found:
                    term_usage_by_document[source] = {
                        "variations": list(variations_found),
                        "locations": locations
                    }
            
            # Check for inconsistencies across documents
            if len(term_usage_by_document) > 1:
                all_variations = set()
                for doc_usage in term_usage_by_document.values():
                    all_variations.update(doc_usage["variations"])
                
                if len(all_variations) > 1:
                    # Found inconsistency
                    terminology_inconsistencies.append({
                        "term": term,
                        "inconsistency_type": "cross_document_variation",
                        "variations_found": list(all_variations),
                        "usage_by_document": term_usage_by_document,
                        "recommendation": f"Standardize usage of '{term}' across all documents",
                        "total_occurrences": sum(len(doc["locations"]) for doc in term_usage_by_document.values())
                    })
        
        # Update context
        if context and hasattr(context, "context"):
            context.context.cross_document_terminology = terminology_inconsistencies
        
        total_issues = len(terminology_inconsistencies)
        
        if total_issues == 0:
            print("✅ Cross-document terminology analysis: No inconsistencies found")
        else:
            print(f"🔴 Cross-document terminology: Found {total_issues} terminology inconsistencies")
            for issue in terminology_inconsistencies[:3]:  # Show first 3
                print(f"  • {issue['term']}: {len(issue['variations_found'])} variations across documents")
        
        return {
            "success": True,
            "total_inconsistencies": total_issues,
            "terminology_issues": terminology_inconsistencies,
            "documents_analyzed": [data["source"] for data in comparison_data]
        }
        
    except Exception as e:
        logger.error(f"Error in cross-document terminology analysis: {str(e)}")
        return {"success": False, "error": str(e)}

@function_tool(strict_mode=False)

def analyze_cross_document_formatting(context: RunContextWrapper[FirmStandardContext]) -> Dict[str, Any]:
    """Analyze formatting consistency across multiple documents"""
    try:
        comparison_data = getattr(context.context, 'comparison_data', [])
        if not comparison_data:
            return {"success": False, "error": "No comparison data available. Run get_multi_document_comparison first."}
        
        formatting_inconsistencies = []
        
        # Analyze each document's formatting patterns
        document_patterns = {}
        
        for doc_data in comparison_data:
            source = doc_data["source"]
            chunks = doc_data["documents"]
            
            patterns = {
                "section_numbering": set(),
                "bullet_styles": set(),
                "quotation_marks": set(),
                "date_formats": set(),
                "section_headers": []
            }
            
            for chunk in chunks:
                content = chunk.get("content", "")
                location = chunk.get("location", {})
                
                # Check section numbering patterns
                section_numbers = re.findall(r'(?:Section|Sec\.?)\s+(\d+(?:\.\d+)*)', content, re.IGNORECASE)
                patterns["section_numbering"].update(section_numbers)
                
                # Check bullet point styles
                bullet_matches = re.findall(r'^[\s]*([•\-\*]\s)', content, re.MULTILINE)
                patterns["bullet_styles"].update(bullet_matches)
                
                # Check quotation mark styles
                quote_matches = re.findall(r'[""\'\'"]', content)
                patterns["quotation_marks"].update(quote_matches)
                
                # Check date formats
                date_matches = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', content)
                patterns["date_formats"].update(date_matches)
                
                # Collect section headers
                if location.get("section_title"):
                    patterns["section_headers"].append({
                        "title": location["section_title"],
                        "type": location.get("section_type", ""),
                        "line_number": location.get("line_number", 0)
                    })
            
            document_patterns[source] = patterns
        
        # Compare patterns across documents
        for pattern_type in ["section_numbering", "bullet_styles", "quotation_marks", "date_formats"]:
            all_patterns = set()
            pattern_by_doc = {}
            
            for source, patterns in document_patterns.items():
                doc_patterns = patterns[pattern_type]
                if doc_patterns:
                    all_patterns.update(doc_patterns)
                    pattern_by_doc[source] = list(doc_patterns)
            
            # If multiple patterns found across documents, it's inconsistent
            if len(all_patterns) > 1 and len(pattern_by_doc) > 1:
                formatting_inconsistencies.append({
                    "type": pattern_type,
                    "inconsistency": f"Multiple {pattern_type.replace('_', ' ')} styles found across documents",
                    "patterns_found": list(all_patterns),
                    "usage_by_document": pattern_by_doc,
                    "recommendation": f"Standardize {pattern_type.replace('_', ' ')} across all documents"
                })
        
        # Analyze section header consistency
        all_section_types = set()
        section_styles_by_doc = {}
        
        for source, patterns in document_patterns.items():
            headers = patterns["section_headers"]
            if headers:
                section_types = set(header["type"] for header in headers if header["type"])
                all_section_types.update(section_types)
                section_styles_by_doc[source] = list(section_types)
        
        if len(all_section_types) > 1:
            formatting_inconsistencies.append({
                "type": "section_header_styles",
                "inconsistency": "Different section header styles found across documents",
                "patterns_found": list(all_section_types),
                "usage_by_document": section_styles_by_doc,
                "recommendation": "Standardize section header formatting across all documents"
            })
        
        # Update context
        if context and hasattr(context, "context"):
            context.context.cross_document_formatting = formatting_inconsistencies
        
        total_issues = len(formatting_inconsistencies)
        
        if total_issues == 0:
            print("✅ Cross-document formatting analysis: No inconsistencies found")
        else:
            print(f"🔴 Cross-document formatting: Found {total_issues} formatting inconsistencies")
            for issue in formatting_inconsistencies:
                print(f"  • {issue['type'].replace('_', ' ').title()}: {len(issue['patterns_found'])} different styles")
        
        return {
            "success": True,
            "total_inconsistencies": total_issues,
            "formatting_issues": formatting_inconsistencies,
            "documents_analyzed": [data["source"] for data in comparison_data]
        }
        
    except Exception as e:
        logger.error(f"Error in cross-document formatting analysis: {str(e)}")
        return {"success": False, "error": str(e)}

@function_tool(strict_mode=False)

def generate_cross_document_compliance_report(context: RunContextWrapper[FirmStandardContext]) -> Dict[str, Any]:
    """Generate comprehensive cross-document compliance report"""
    try:
        # Get all analysis results from context
        comparison_result = getattr(context.context, 'comparison_result', {})
        terminology_issues = getattr(context.context, 'cross_document_terminology', [])
        formatting_issues = getattr(context.context, 'cross_document_formatting', [])
        
        # Get basic style analysis results
        punctuation_issues = getattr(context.context, 'punctuation_issues', [])
        capitalization_issues = getattr(context.context, 'capitalization_problems', [])
        structure_issues = getattr(context.context, 'sentence_structure_issues', [])
        word_choice_issues = getattr(context.context, 'word_choice_inconsistencies', [])
        
        # Calculate issue counts
        total_terminology = len(terminology_issues)
        total_formatting = len(formatting_issues)
        total_punctuation = sum(len(doc.get("issues", [])) for doc in punctuation_issues)
        total_capitalization = sum(len(doc.get("issues", [])) for doc in capitalization_issues)
        total_structure = sum(len(doc.get("issues", [])) for doc in structure_issues)
        total_word_choice = sum(len(doc.get("issues", [])) for doc in word_choice_issues)
        
        total_cross_doc_issues = total_terminology + total_formatting
        total_all_issues = total_cross_doc_issues + total_punctuation + total_capitalization + total_structure + total_word_choice
        
        # Get document information
        documents_analyzed = comparison_result.get("document_sources", [])
        total_docs = len(documents_analyzed)
        
        # Calculate compliance scores
        if total_docs > 0:
            # Cross-document specific score
            cross_doc_score = max(0, 100 - (total_cross_doc_issues * 15))  # Heavier penalty for cross-doc issues
            
            # Overall compliance score
            overall_score = max(0, 100 - (total_all_issues / total_docs * 8))
        else:
            cross_doc_score = overall_score = 100
        
        # Determine compliance levels
        def get_compliance_level(score):
            if score >= 95: return "Excellent"
            elif score >= 85: return "Good"
            elif score >= 70: return "Fair"
            else: return "Needs Improvement"
        
        cross_doc_level = get_compliance_level(cross_doc_score)
        overall_level = get_compliance_level(overall_score)
        
        # Generate detailed report
        report = {
            "success": True,
            "analysis_type": "cross_document_compliance",
            "documents_analyzed": documents_analyzed,
            "total_documents": total_docs,
            "cross_document_analysis": {
                "compliance_score": cross_doc_score,
                "compliance_level": cross_doc_level,
                "total_issues": total_cross_doc_issues,
                "issue_breakdown": {
                    "terminology_inconsistencies": total_terminology,
                    "formatting_inconsistencies": total_formatting
                }
            },
            "overall_analysis": {
                "compliance_score": overall_score,
                "compliance_level": overall_level,
                "total_issues": total_all_issues,
                "issue_breakdown": {
                    "cross_document_issues": total_cross_doc_issues,
                    "punctuation_issues": total_punctuation,
                    "capitalization_issues": total_capitalization,
                    "structure_issues": total_structure,
                    "word_choice_issues": total_word_choice
                }
            },
            "detailed_findings": {
                "terminology_inconsistencies": terminology_issues,
                "formatting_inconsistencies": formatting_issues,
                "document_comparison": comparison_result
            },
            "recommendations": []
        }
        
        # Generate recommendations
        if total_terminology > 0:
            report["recommendations"].append({
                "priority": "High",
                "category": "Terminology",
                "issue": f"{total_terminology} terminology inconsistencies found across documents",
                "action": "Standardize legal terminology usage across all documents"
            })
        
        if total_formatting > 0:
            report["recommendations"].append({
                "priority": "Medium",
                "category": "Formatting",
                "issue": f"{total_formatting} formatting inconsistencies found across documents",
                "action": "Establish consistent formatting standards for all document types"
            })
        
        if total_punctuation > 0:
            report["recommendations"].append({
                "priority": "Medium",
                "category": "Punctuation",
                "issue": f"{total_punctuation} punctuation issues found",
                "action": "Review and standardize punctuation usage"
            })
        
        # Print summary
        print(f"\n📊 CROSS-DOCUMENT COMPLIANCE REPORT")
        print(f"📄 Documents Analyzed: {total_docs}")
        print(f"📈 Cross-Document Score: {cross_doc_score:.1f}% ({cross_doc_level})")
        print(f"📈 Overall Score: {overall_score:.1f}% ({overall_level})")
        print(f"🔍 Total Issues Found: {total_all_issues}")
        
        if total_cross_doc_issues > 0:
            print(f"\n🔴 Cross-Document Issues:")
            if total_terminology > 0:
                print(f"  • Terminology: {total_terminology} inconsistencies")
            if total_formatting > 0:
                print(f"  • Formatting: {total_formatting} inconsistencies")
        
        if total_all_issues == 0:
            print("✅ Excellent! All documents maintain consistent style and formatting.")
        
        # Update context
        if context and hasattr(context, "context"):
            context.context.cross_document_report = report
            context.context.cross_document_analysis_complete = True
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating cross-document compliance report: {str(e)}")
        return {"success": False, "error": str(e)} 