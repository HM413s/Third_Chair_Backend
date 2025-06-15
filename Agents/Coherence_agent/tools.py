from agents import function_tool
from typing import Dict, Any, List, Optional
from RAG import rag_service
from .clause_utils import AdvancedClauseAnalyzer
from agents import RunContextWrapper
from .context import ContractAnalysisContext
import re
import logging
import traceback
import time
from functools import wraps
import json
from RAG.rag_service import RAGService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('contract_analysis.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Error handling decorator
def handle_errors(func):
    """Decorator to handle errors and log them consistently"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func_name = func.__name__
        
        try:
            logger.info(f"Starting {func_name} with args: {len(args)} positional, {len(kwargs)} keyword")
            result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            logger.info(f"Successfully completed {func_name} in {execution_time:.2f} seconds")

            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Error in {func_name} after {execution_time:.2f} seconds: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            return {
                "success": False,
                "error": str(e),
                "function": func_name,
                "execution_time": execution_time,
                "traceback": traceback.format_exc()
            }
    
    return wrapper

rag_service = rag_service.RAGService()

@function_tool(strict_mode=False)
@handle_errors
def generate_consistency_report(context: RunContextWrapper[ContractAnalysisContext], analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a comprehensive consistency report"""
    logger.info("Generating consistency report")
    
    if not analysis_results:
        logger.warning("No analysis results provided for report generation")
        raise ValueError("No analysis results provided")
    
    report = {
        "executive_summary": {},
        "detailed_findings": {},
        "recommendations": [],
        "consistency_score": 0
    }
    
    # Calculate consistency score (0-100)
    total_issues = 0
    max_possible_issues = 0
    
    logger.debug("Calculating consistency score from analysis results")
    
    # Count issues from various analyses
    if "multi_doc_analysis" in analysis_results:
        multi_doc = analysis_results["multi_doc_analysis"]
        if multi_doc.get("success"):
            inconsistent_terms = len(multi_doc.get("term_consistency", {}).get("inconsistent_terms", {}))
            orphaned_refs = sum(len(refs) for refs in multi_doc.get("orphaned_references", {}).values())
            total_issues += inconsistent_terms + orphaned_refs
            max_possible_issues += 20
            logger.debug(f"Multi-doc analysis: {inconsistent_terms} inconsistent terms, {orphaned_refs} orphaned refs")
        else:
            logger.warning("Multi-document analysis was not successful")
    
    if "case_analysis" in analysis_results:
        case_analysis = analysis_results["case_analysis"]
        if case_analysis.get("success"):
            case_issues = len(case_analysis.get("case_issues", {}))
            formatting_issues = sum(1 for issues in case_analysis.get("formatting_issues", {}).values() if issues)
            total_issues += case_issues + formatting_issues
            max_possible_issues += 10
            logger.debug(f"Case analysis: {case_issues} case issues, {formatting_issues} formatting issues")
        else:
            logger.warning("Case analysis was not successful")
    
    # Calculate score
    if max_possible_issues > 0:
        consistency_score = max(0, 100 - (total_issues / max_possible_issues * 100))
    else:
        consistency_score = 100
        logger.info("No issues found, setting consistency score to 100")
    
    report["consistency_score"] = round(consistency_score, 2)
    logger.info(f"Calculated consistency score: {consistency_score}")
    
    # Executive summary
    rating = ("Excellent" if consistency_score >= 90 else
              "Good" if consistency_score >= 75 else
              "Fair" if consistency_score >= 60 else "Poor")
    
    report["executive_summary"] = {
        "total_issues_found": total_issues,
        "consistency_rating": rating,
        "primary_concerns": []
    }
    
    # Add primary concerns
    concerns_added = 0
    if total_issues > 0:
        if "multi_doc_analysis" in analysis_results:
            multi_doc = analysis_results["multi_doc_analysis"]
            if multi_doc.get("term_consistency", {}).get("inconsistent_count", 0) > 0:
                report["executive_summary"]["primary_concerns"].append("Inconsistent term definitions across documents")
                concerns_added += 1
            if any(multi_doc.get("orphaned_references", {}).values()):
                report["executive_summary"]["primary_concerns"].append("References to non-existent documents")
                concerns_added += 1
        
        if "case_analysis" in analysis_results:
            case_analysis = analysis_results["case_analysis"]
            if case_analysis.get("needs_alignment"):
                report["executive_summary"]["primary_concerns"].append("Inconsistent case and formatting")
                concerns_added += 1
    
    logger.info(f"Added {concerns_added} primary concerns to executive summary")
    
    # Recommendations
    recommendations = []
    if total_issues > 0:
        recommendations.extend([
            "Standardize clause reference formatting across all documents",
            "Create a master definitions document to ensure term consistency",
            "Implement document review checklist for case and formatting consistency",
            "Establish cross-reference validation process"
        ])
    else:
        recommendations.append("Document consistency is excellent - maintain current standards")
    
    report["recommendations"] = recommendations
    report["detailed_findings"] = analysis_results
    
    # Update context
    try:
        context.context.comprehensive_report = report
        logger.debug("Updated context with comprehensive report")
    except Exception as e:
        logger.warning(f"Failed to update context: {str(e)}")
    
    logger.info("Successfully generated consistency report")
    return {
        "success": True,
        "report": report,
        "final_output": json.dumps(report, indent=2)
    }

@function_tool(strict_mode=False)
@handle_errors
def get_all_documents_from_rag(context: RunContextWrapper[ContractAnalysisContext]) -> Dict[str, Any]:
    """Get all documents stored in RAG for multi-document analysis with enhanced location tracking"""
    logger.info("Retrieving all documents from RAG service with location metadata")
    
    try:
        # Get documents from context
        if not context or not hasattr(context, "context"):
            logger.error("No context found")
            return {
                "success": False,
                "error": "No context found",
                "documents": [],
                "total_documents": 0,
                "final_output": {
                    "error": "No context found"
                }
            }

        # Get documents from RAG service with retry logic
        max_retries = 1
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                rag_service = RAGService()
                all_docs = rag_service.get_all_documents()
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    logger.warning(f"Rate limited, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                raise
        
        if not all_docs:
            logger.warning("No documents found in RAG service")
            return {
                "success": True,  # Changed to True since this is a valid state
                "documents": [],
                "total_documents": 0,
                "final_output": {
                    "message": "No documents found in RAG service",
                    "documents": []
                }
            }

        # Process the RAG documents
        processed_docs = []
        location_summary = {}
        
        for doc in all_docs:
            try:
                processed_doc = {
                    "content": doc.get("content", ""),
                    "type": doc.get("type", "text"),
                    "metadata": doc.get("metadata", {}),
                    "location": doc.get("location", {}),
                    "source_file": doc.get("source_file", "unknown"),
                    "created_at": doc.get("created_at", "")
                }
                processed_docs.append(processed_doc)
                
                # Update location summary
                source = processed_doc["source_file"]
                if source not in location_summary:
                    location_summary[source] = 0
                location_summary[source] += 1
                
            except Exception as e:
                logger.error(f"Error processing document: {str(e)}")
                continue
        
        logger.info(f"Successfully processed {len(processed_docs)} RAG-processed documents with location tracking")
        
        result = {
            "success": True,
            "documents": processed_docs,
            "total_documents": len(processed_docs),
            "location_summary": location_summary,
            "final_output": {
                "documents": processed_docs,
                "total_documents": len(processed_docs),
                "location_summary": location_summary
            }
        }
        return result
        
    except Exception as e:
        logger.error(f"Error in get_all_documents_from_rag: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "documents": [],
            "total_documents": 0,
            "final_output": {
                "error": str(e)
            }
        }

@function_tool(strict_mode=False)
@handle_errors
def check_multi_document_consistency(context: RunContextWrapper[ContractAnalysisContext], documents: List[Dict[str, str]]) -> Dict[str, Any]:
    """Check consistency across multiple documents with precise location tracking"""
    logger.info(f"Checking consistency across {len(documents)} documents with enhanced location reporting")
    
    if not documents:
        logger.error("No documents provided for consistency check")
        raise ValueError("No documents provided")
    
    analyzer = AdvancedClauseAnalyzer()
    
    # Enhanced analysis with precise location tracking
    detailed_inconsistencies = []
    document_summaries = []
    location_based_issues = []
    
    # Analyze each document for inconsistencies with location data
    for i, doc in enumerate(documents):
        doc_name = doc.get("name", f"Document_{i+1}")
        content = doc.get("content", "")
        location_info = doc.get("location", {})
        content_flags = doc.get("content_flags", {})
        
        # Track various inconsistency types with precise source info
        doc_analysis = {
            "document_name": doc_name,
            "document_index": i + 1,
            "content_length": len(content),
            "location_metadata": location_info,
            "content_flags": content_flags,
            "case_issues": [],
            "reference_issues": [],
            "formatting_issues": [],
            "location_based_issues": []
        }
        
        # Check for inconsistent terminology with location tracking
        term_patterns = [
            (r'\b(?:agreement|Agreement|AGREEMENT)\b', 'agreement'),
            (r'\b(?:party|Party|PARTY)\b', 'party'), 
            (r'\b(?:section|Section|SECTION)\b', 'section'),
            (r'\b(?:clause|Clause|CLAUSE)\b', 'clause'),
            (r'\b(?:exhibit|Exhibit|EXHIBIT)\b', 'exhibit'),
            (r'\b(?:appendix|Appendix|APPENDIX)\b', 'appendix')
        ]
        
        for pattern, term_type in term_patterns:
            matches = list(re.finditer(pattern, content))
            if len(set(match.group() for match in matches)) > 1:
                # Found inconsistent capitalization
                variations = list(set(match.group() for match in matches))
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    char_position = match.start()
                    
                    # Extract surrounding context
                    context_start = max(0, char_position - 100)
                    context_end = min(len(content), char_position + 100)
                    surrounding_context = content[context_start:context_end]
                    
                    # Create precise location reference
                    location_ref = {
                        "file": doc_name,
                        "line_number": line_num,
                        "char_position": char_position,
                        "section_path": location_info.get("section_path", "Unknown section"),
                        "absolute_position": f"Line {line_num}, Char {char_position}",
                        "surrounding_context": surrounding_context.strip()
                    }
                    
                    issue = {
                        "term": match.group(),
                        "term_type": term_type,
                        "variations_found": variations,
                        "location": location_ref,
                        "issue_type": "case_inconsistency",
                        "severity": "medium"
                    }
                    doc_analysis["case_issues"].append(issue)
                    location_based_issues.append(issue)
        
        # Check for broken references with enhanced location tracking
        ref_patterns = [
            (r'(?:Section|Clause|Article)\s+(\d+(?:\.\d+)*)', 'section_reference'),
            (r'(?:Exhibit|Appendix)\s+([A-Z])', 'exhibit_reference'),
            (r'(?:Schedule|Attachment)\s+(\w+)', 'schedule_reference')
        ]
        
        for pattern, ref_type in ref_patterns:
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                char_position = match.start()
                ref_text = match.group()
                
                # Check if this reference exists as an actual section
                ref_exists = bool(re.search(rf'^{re.escape(ref_text)}', content, re.MULTILINE | re.IGNORECASE))
                
                if not ref_exists:
                    # Extract surrounding context
                    context_start = max(0, char_position - 150)
                    context_end = min(len(content), char_position + 150)
                    surrounding_context = content[context_start:context_end]
                    
                    location_ref = {
                        "file": doc_name,
                        "line_number": line_num,
                        "char_position": char_position,
                        "section_path": location_info.get("section_path", "Unknown section"),
                        "absolute_position": f"Line {line_num}, Char {char_position}",
                        "surrounding_context": surrounding_context.strip()
                    }
                    
                    issue = {
                        "broken_reference": ref_text,
                        "reference_type": ref_type,
                        "location": location_ref,
                        "issue_type": "broken_reference",
                        "severity": "high"
                    }
                    doc_analysis["reference_issues"].append(issue)
                    location_based_issues.append(issue)
        
        # Check for formatting inconsistencies with location details
        numbering_patterns = [
            (r'^\s*\d+\.\s+', 'numbered_list'),
            (r'^\s*\(\d+\)\s+', 'parenthetical_numbered'),
            (r'^\s*[A-Z]\.\s+', 'lettered_list'),
            (r'^\s*\([a-z]\)\s+', 'parenthetical_lettered')
        ]
        
        numbering_styles = {}
        for pattern, style_name in numbering_patterns:
            matches = list(re.finditer(pattern, content, re.MULTILINE))
            if matches:
                numbering_styles[style_name] = []
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    numbering_styles[style_name].append({
                        "line_number": line_num,
                        "text": match.group().strip(),
                        "char_position": match.start()
                    })
        
        if len(numbering_styles) > 1:
            location_ref = {
                "file": doc_name,
                "section_path": location_info.get("section_path", "Document-wide"),
                "styles_found": list(numbering_styles.keys()),
                "locations": numbering_styles
            }
            
            issue = {
                "issue_type": "mixed_numbering_styles",
                "styles_found": list(numbering_styles.keys()),
                "location": location_ref,
                "severity": "medium",
                "style_details": numbering_styles
            }
            doc_analysis["formatting_issues"].append(issue)
            location_based_issues.append(issue)
        
        document_summaries.append(doc_analysis)
        
        # Add to detailed inconsistencies if issues found
        total_issues = (len(doc_analysis["case_issues"]) + 
                       len(doc_analysis["reference_issues"]) + 
                       len(doc_analysis["formatting_issues"]))
        
        if total_issues > 0:
            detailed_inconsistencies.append({
                "document": doc_name,
                "total_issues": total_issues,
                "location_info": location_info,
                "breakdown": {
                    "case_issues": len(doc_analysis["case_issues"]),
                    "reference_issues": len(doc_analysis["reference_issues"]),
                    "formatting_issues": len(doc_analysis["formatting_issues"])
                },
                "details": doc_analysis,
                "location_summary": {
                    "file_path": location_info.get("absolute_position", "Unknown location"),
                    "section_coverage": location_info.get("section_path", "Unknown section"),
                    "line_range": f"Lines around {location_info.get('line_number', 'Unknown')}"
                }
            })
    
    # Cross-document analysis with location tracking
    all_terms = {}
    for doc in documents:
        content = doc.get("content", "")
        doc_name = doc.get("name", "unknown")
        location_info = doc.get("location", {})
        
        # Extract defined terms from each document with locations
        definition_pattern = r'"([^"]+)"\s+(?:means|shall mean|is defined as)'
        for match in re.finditer(definition_pattern, content, re.IGNORECASE):
            term = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            
            if term not in all_terms:
                all_terms[term] = []
            
            all_terms[term].append({
                "document": doc_name,
                "line_number": line_num,
                "section_path": location_info.get("section_path", "Unknown section"),
                "definition_text": match.group(),
                "char_position": match.start()
            })
    
    # Find terms defined differently across documents
    inconsistent_definitions = {}
    for term, locations in all_terms.items():
        if len(set(loc["document"] for loc in locations)) > 1:
            inconsistent_definitions[term] = locations
    
    # Generate location-aware recommendations
    location_based_recommendations = []
    if location_based_issues:
        # Group issues by type and location
        issues_by_type = {}
        for issue in location_based_issues:
            issue_type = issue.get("issue_type", "unknown")
            if issue_type not in issues_by_type:
                issues_by_type[issue_type] = []
            issues_by_type[issue_type].append(issue)
        
        for issue_type, issues in issues_by_type.items():
            files_affected = set(issue["location"]["file"] for issue in issues)
            location_based_recommendations.append(
                f"Fix {len(issues)} {issue_type} issues across {len(files_affected)} files: {', '.join(files_affected)}"
            )
    
    # Use the comprehensive_document_analysis method for additional insights
    try:
        if hasattr(analyzer, 'comprehensive_document_analysis'):
            analysis_results = analyzer.comprehensive_document_analysis(documents)
        else:
            logger.warning("AdvancedClauseAnalyzer does not have comprehensive_document_analysis method")
            analysis_results = {
                "summary": f"Analyzed {len(documents)} documents using enhanced location-aware analysis",
                "detailed_analysis": detailed_inconsistencies,
                "cross_document_analysis": inconsistent_definitions,
                "location_based_issues": location_based_issues
            }
    except Exception as e:
        logger.warning(f"AdvancedClauseAnalyzer failed: {str(e)}")
        analysis_results = {
            "summary": f"Analyzed {len(documents)} documents with enhanced location tracking",
            "detailed_analysis": detailed_inconsistencies,
            "cross_document_analysis": inconsistent_definitions,
            "location_based_issues": location_based_issues,
            "error": str(e)
        }
    
    # Update context with enhanced location data
    try:
        if (context and hasattr(context, "context") and 
            hasattr(context.context, "multi_document_analysis_results") and
            isinstance(context.context.multi_document_analysis_results, dict)):
            
            context.context.multi_document_analysis_results["comprehensive_analysis"] = analysis_results
            context.context.multi_document_analysis_results["detailed_inconsistencies"] = detailed_inconsistencies
            context.context.multi_document_analysis_results["location_based_issues"] = location_based_issues
            logger.debug("Updated context with comprehensive location-aware analysis results")
        else:
            logger.warning("Context structure is not as expected for multi_document_analysis_results update")
    except Exception as e:
        logger.warning(f"Failed to update context: {str(e)}")
    
    # Print location-aware issue summary
    print(f"\n📍 LOCATION-AWARE CONSISTENCY ANALYSIS RESULTS:")
    print(f"📊 Total Issues Found: {len(location_based_issues)}")
    
    for issue in location_based_issues[:5]:  # Show first 5 issues
        location = issue.get("location", {})
        print(f"  🔴 {issue.get('issue_type', 'Unknown Issue')}")
        print(f"    📄 File: {location.get('file', 'Unknown')}")
        print(f"    📍 Location: {location.get('absolute_position', 'Unknown position')}")
        print(f"    📑 Section: {location.get('section_path', 'Unknown section')}")
        if issue.get('term'):
            print(f"    🔤 Term: '{issue['term']}' (found variations: {', '.join(issue.get('variations_found', []))})")
        elif issue.get('broken_reference'):
            print(f"    🔗 Broken Reference: '{issue['broken_reference']}'")
        print(f"    ⚠️  Severity: {issue.get('severity', 'unknown').upper()}")
        print()
    
    if len(location_based_issues) > 5:
        print(f"    ... and {len(location_based_issues) - 5} more issues")
    
    report = {
        "summary": f"Analyzed {len(documents)} documents with {len(detailed_inconsistencies)} documents containing {len(location_based_issues)} total issues",
        "total_documents_analyzed": len(documents),
        "documents_with_issues": len(detailed_inconsistencies),
        "total_location_tracked_issues": len(location_based_issues),
        "detailed_inconsistencies": detailed_inconsistencies,
        "location_based_issues": location_based_issues,
        "document_summaries": document_summaries,
        "cross_document_issues": {
            "inconsistent_definitions": inconsistent_definitions,
            "definition_conflicts": len(inconsistent_definitions)
        },
        "recommendations": [
            f"Review {len(detailed_inconsistencies)} documents with consistency issues",
            "Standardize terminology capitalization across all documents",
            "Fix broken cross-references and ensure all referenced sections exist",
            "Establish consistent formatting and numbering schemes"
        ] + location_based_recommendations,
        "details": analysis_results
    }
    return {"final_output": report, **report}

@function_tool(strict_mode=False)
@handle_errors
def check_case_alignment(context: RunContextWrapper[ContractAnalysisContext], contract_text: str) -> Dict[str, Any]:
    """Check case consistency and alignment in contract text"""
    logger.info("Checking case alignment and consistency")
    
    if not contract_text:
        logger.error("No contract text provided for case alignment check")
        raise ValueError("No contract text provided")
    
    logger.debug(f"Analyzing contract text of {len(contract_text)} characters")
    
    analyzer = AdvancedClauseAnalyzer()
    
    try:
        case_issues = analyzer.check_case_consistency(contract_text)
        logger.info(f"Case consistency check completed: {len(case_issues)} issues found")
    except Exception as e:
        logger.error(f"Failed to check case consistency: {str(e)}")
        raise
    
    # Additional case checks
    formatting_issues = {
        'mixed_numbering': [],
        'inconsistent_headers': [],
        'capitalization_issues': []
    }
    
    # Check section numbering consistency
    logger.debug("Checking section numbering consistency")
    section_patterns = [
        r'Section\s+\d+',
        r'SECTION\s+\d+', 
        r'section\s+\d+',
        r'Sec\.\s+\d+',
        r'SEC\.\s+\d+'
    ]
    
    numbering_styles = {}
    for pattern in section_patterns:
        matches = re.findall(pattern, contract_text)
        if matches:
            numbering_styles[pattern] = len(matches)
            logger.debug(f"Found {len(matches)} matches for pattern: {pattern}")
    
    if len(numbering_styles) > 1:
        formatting_issues['mixed_numbering'] = numbering_styles
        logger.warning(f"Mixed numbering styles detected: {list(numbering_styles.keys())}")
    
    # Check header consistency
    logger.debug("Checking header consistency")
    header_patterns = [
        r'^[A-Z\s]+',
        r'^[A-Z][a-z\s]+',
        r'^\d+\.\s+[A-Z][A-Z\s]+'
    ]
    
    header_issues = {}
    for pattern in header_patterns:
        matches = re.findall(pattern, contract_text, re.MULTILINE)
        if matches:
            header_issues[pattern] = len(matches)
            logger.debug(f"Found {len(matches)} header matches for pattern: {pattern}")
    
    if len(header_issues) > 1:
        formatting_issues['inconsistent_headers'] = header_issues
        logger.warning(f"Inconsistent header styles detected: {list(header_issues.keys())}")
    
    # Check capitalization consistency
    logger.debug("Checking capitalization consistency")
    capitalization_issues = {}
    
    try:
        for term_type, variations in analyzer.case_patterns.items():
            found_variations = {}
            for variation in variations:
                matches = re.findall(variation, contract_text)
                if matches:
                    found_variations[variation] = len(matches)
            
            if len(found_variations) > 1:
                capitalization_issues[term_type] = found_variations
                logger.debug(f"Capitalization issues for {term_type}: {list(found_variations.keys())}")
    except AttributeError:
        logger.warning("AdvancedClauseAnalyzer does not have case_patterns attribute")
    
    if capitalization_issues:
        formatting_issues['capitalization_issues'] = capitalization_issues
    
    needs_alignment = bool(case_issues or any(formatting_issues.values()))
    logger.info(f"Case alignment check completed. Needs alignment: {needs_alignment}")
    
    result = {
        "success": True,
        "case_issues": case_issues,
        "formatting_issues": formatting_issues,
        "needs_alignment": needs_alignment,
        "final_output": {
            "case_issues": case_issues,
            "formatting_issues": formatting_issues,
            "needs_alignment": needs_alignment
        }
    }
    
    # Update context
    try:
        context.context.single_document_analysis_results= result
        logger.debug("Updated context with clause reference analysis")
    except Exception as e:
        logger.warning(f"Failed to update context: {str(e)}")
    
    return result

@function_tool(strict_mode=False)
@handle_errors
def search_contract_content(context: RunContextWrapper[ContractAnalysisContext], query: str) -> Dict[str, Any]:
    """Search contract content using RAG for clause analysis"""
    logger.info(f"Searching contract content with query: '{query}'")
    
    if not query or not query.strip():
        logger.error("Empty query provided for contract search")
        raise ValueError("Query cannot be empty")
    
    try:
        result = rag_service.chat(query)
        logger.info(f"Search completed successfully. Found {result.get('metadata', {}).get('total_docs', 0)} documents")
        logger.debug(f"Search response length: {len(result.get('response', ''))}")
        
        return {
            "success": True,
            "response": result["response"],
            "total_docs": result["metadata"]["total_docs"],
            "text_docs": result["metadata"]["text_docs"]
        }
    except Exception as e:
        logger.error(f"RAG service search failed: {str(e)}")
        raise

@function_tool(strict_mode=False)
@handle_errors
def analyze_clause_references(context: RunContextWrapper[ContractAnalysisContext], contract_text: str) -> Dict[str, Any]:
    """Analyze clause references in the contract text"""
    logger.info("Analyzing clause references")
    
    if not contract_text:
        logger.error("No contract text provided for clause reference analysis")
        raise ValueError("No contract text provided")
    
    logger.debug(f"Analyzing contract text of {len(contract_text)} characters")
    
    analyzer = AdvancedClauseAnalyzer()
    
    # Extract all references
    logger.debug("Extracting clause references")
    try:
        references = analyzer.extract_clause_references(contract_text)
        total_refs = sum(len(refs) for refs in references.values())
        logger.info(f"Extracted {total_refs} clause references")
    except Exception as e:
        logger.error(f"Failed to extract clause references: {str(e)}")
        raise
    
    logger.debug("Extracting defined terms")
    try:
        defined_terms = analyzer.extract_defined_terms(contract_text)
        logger.info(f"Extracted {len(defined_terms)} defined terms")
    except Exception as e:
        logger.error(f"Failed to extract defined terms: {str(e)}")
        raise
    
    logger.debug("Finding definition sections")
    try:
        definition_sections = analyzer.find_definition_sections(contract_text)
        logger.info(f"Found {len(definition_sections)} definition sections")
    except Exception as e:
        logger.error(f"Failed to find definition sections: {str(e)}")
        raise
    
    # Find existing sections
    logger.debug("Finding existing sections")
    try:
        existing_sections = re.findall(r'(?:Section|Clause|Article)\s+(\d+(?:\.\d+)*)', contract_text, re.IGNORECASE)
        existing_sections = list(set(existing_sections))
        logger.info(f"Found {len(existing_sections)} existing sections")
    except Exception as e:
        logger.error(f"Failed to find existing sections: {str(e)}")
        existing_sections = []
    
    result = {
        "success": True,
        "references": references,
        "defined_terms": defined_terms,
        "definition_sections": len(definition_sections),
        "existing_sections": existing_sections,
        "total_references": sum(len(refs) for refs in references.values()),
        "final_output": {
            "references": references,
            "defined_terms": defined_terms,
            "existing_sections": existing_sections
        }
    }
    
    # Update context
    try:
        context.context.single_document_analysis_results= result
        logger.debug("Updated context with clause reference analysis")
    except Exception as e:
        logger.warning(f"Failed to update context: {str(e)}")
    
    return result

@function_tool(strict_mode=False)
@handle_errors
def check_reference_validity(context: RunContextWrapper[ContractAnalysisContext], section_refs: List[str], existing_sections: List[str]) -> Dict[str, Any]:
    """Check if section references are valid (exist in document)"""
    logger.info(f"Checking validity of {len(section_refs)} section references against {len(existing_sections)} existing sections")
    
    if not section_refs:
        logger.warning("No section references provided for validity check")
        return {
            "success": True,
            "valid_references": [],
            "missing_references": [],
            "validity_rate": 1.0,
            "final_output": {
                "valid_references": [],
                "missing_references": [],
                "validity_rate": 1.0
            }
        }
    
    missing_refs = []
    valid_refs = []
    
    for ref in section_refs:
        if ref in existing_sections:
            valid_refs.append(ref)
        else:
            missing_refs.append(ref)
    
    validity_rate = len(valid_refs) / len(section_refs) if section_refs else 1.0
    
    logger.info(f"Reference validity check: {len(valid_refs)} valid, {len(missing_refs)} missing (rate: {validity_rate:.2%})")
    
    if missing_refs:
        logger.warning(f"Missing references: {missing_refs}")
    
    result = {
        "success": True,
        "valid_references": valid_refs,
        "missing_references": missing_refs,
        "validity_rate": validity_rate,
        "final_output": {
            "valid_references": valid_refs,
            "missing_references": missing_refs,
            "validity_rate": validity_rate
        }
    }
    return result

@function_tool(strict_mode=False)
@handle_errors
def find_circular_references(context: RunContextWrapper[ContractAnalysisContext], contract_text: str) -> Dict[str, Any]:
    """Detect circular references between clauses"""
    logger.info("Detecting circular references")
    
    if not contract_text:
        logger.error("No contract text provided for circular reference detection")
        raise ValueError("No contract text provided")
    
    # Extract section content with their references
    sections = {}
    section_pattern = r'(?:Section|Clause|Article)\s+(\d+(?:\.\d+)*)[^\n]*\n(.*?)(?=(?:Section|Clause|Article)\s+\d+|\Z)'
    
    logger.debug("Extracting sections and their references")
    try:
        matches = re.finditer(section_pattern, contract_text, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            section_num = match.group(1)
            section_content = match.group(2)
            
            # Find references in this section
            refs = re.findall(r'(?:Section|Clause|Article)\s+(\d+(?:\.\d+)*)', section_content, re.IGNORECASE)
            sections[section_num] = refs
            logger.debug(f"Section {section_num}: {len(refs)} references")
    except Exception as e:
        logger.error(f"Failed to extract section references: {str(e)}")
        raise
    
    logger.info(f"Analyzing {len(sections)} sections for circular references")
    
    # Check for circular references
    circular_refs = []
    for section, refs in sections.items():
        for ref in refs:
            if ref in sections and section in sections[ref]:
                circular_refs.append((section, ref))
                logger.warning(f"Circular reference detected: {section} <-> {ref}")
    
    logger.info(f"Found {len(circular_refs)} circular references")
    
    result = {
        "success": True,
        "circular_references": circular_refs,
        "total_sections_analyzed": len(sections),
        "sections_with_refs": sum(1 for refs in sections.values() if refs),
        "final_output": {
            "circular_references": circular_refs
        }
    }
    return result

@function_tool(strict_mode=False)
@handle_errors
def validate_defined_terms(context: RunContextWrapper[ContractAnalysisContext], defined_terms: List[str], contract_text: str) -> Dict[str, Any]:
    """Validate that defined terms are actually defined in the contract"""
    logger.info(f"Validating {len(defined_terms)} defined terms")
    
    if not defined_terms:
        logger.warning("No defined terms provided for validation")
        return {
            "success": True,
            "properly_defined": [],
            "undefined_terms": [],
            "definition_rate": 1.0,
            "final_output": {
                "properly_defined": [],
                "undefined_terms": [],
                "definition_rate": 1.0
            }
        }
    
    if not contract_text:
        logger.error("No contract text provided for term validation")
        raise ValueError("No contract text provided")
    
    undefined_terms = []
    defined_properly = []
    
    for term in defined_terms:
        logger.debug(f"Validating term: {term}")
        
        # Look for definition patterns
        definition_patterns = [
            rf'"{re.escape(term)}"\s+means',
            rf'"{re.escape(term)}"\s+shall mean',
            rf'"{re.escape(term)}"\s+is defined as',
            rf'"{re.escape(term)}"\s+refers to',
            rf'the term "{re.escape(term)}"'
        ]
        
        found_definition = False
        for pattern in definition_patterns:
            if re.search(pattern, contract_text, re.IGNORECASE):
                found_definition = True
                logger.debug(f"Found definition for {term} using pattern: {pattern}")
                break
        
        if found_definition:
            defined_properly.append(term)
        else:
            undefined_terms.append(term)
            logger.warning(f"No definition found for term: {term}")
    
    definition_rate = len(defined_properly) / len(defined_terms) if defined_terms else 1.0
    logger.info(f"Term validation: {len(defined_properly)} properly defined, {len(undefined_terms)} undefined (rate: {definition_rate:.2%})")
    
    result = {
        "success": True,
        "properly_defined": defined_properly,
        "undefined_terms": undefined_terms,
        "definition_rate": definition_rate,
        "final_output": {
            "properly_defined": defined_properly,
            "undefined_terms": undefined_terms,
            "definition_rate": definition_rate
        }
    }
    return result

@function_tool(strict_mode=False)
@handle_errors
def check_duplicate_definitions(context: RunContextWrapper[ContractAnalysisContext], contract_text: str) -> Dict[str, Any]:
    """Check for duplicate definitions of the same term"""
    logger.info("Checking for duplicate definitions")
    
    if not contract_text:
        logger.error("No contract text provided for duplicate definition check")
        raise ValueError("No contract text provided")
    
    # Find all definition instances
    definition_pattern = r'"([^"]+)"\s+(?:means|shall mean|is defined as|refers to)'
    
    try:
        definitions = re.findall(definition_pattern, contract_text, re.IGNORECASE)
        logger.info(f"Found {len(definitions)} total definitions")
    except Exception as e:
        logger.error(f"Failed to extract definitions: {str(e)}")
        raise
    
    # Count occurrences
    term_counts = {}
    for term in definitions:
        term_lower = term.lower()
        term_counts[term_lower] = term_counts.get(term_lower, 0) + 1
    
    # Find duplicates
    duplicates = {term: count for term, count in term_counts.items() if count > 1}
    
    logger.info(f"Found {len(duplicates)} terms with duplicate definitions")
    for term, count in duplicates.items():
        logger.warning(f"Term '{term}' defined {count} times")
    
    result = {
        "success": True,
        "duplicate_definitions": duplicates,
        "total_definitions": len(definitions),
        "unique_terms": len(term_counts),
        "final_output": {
            "duplicate_definitions": duplicates
        }
    }
    return result

@function_tool(strict_mode=False)
async def analyze_contract_consistency(
    context: RunContextWrapper[ContractAnalysisContext],
    contract_file_path: Optional[str] = None, 
    contract_text: Optional[str] = None, 
    multi_document_analysis: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive contract consistency analysis tool with multi-document support
    
    Args:
        contract_file_path: Path to contract file to process
        contract_text: Direct contract text to analyze  
        multi_document_analysis: Whether to perform multi-document analysis
    
    Returns:
        Dictionary containing comprehensive consistency analysis results
    """
    
    try:
        analysis_results = {}
        
        # Process document if file path provided
        if contract_file_path:
            doc_count = rag_service.process_and_store_document(contract_file_path)
            print(f"Processed {doc_count} document chunks from {contract_file_path}")
            
            # Get contract text for analysis
            rag_result = rag_service.chat("full contract text content")
            contract_text = rag_result["response"]
        
        if not contract_text:
            print("No contract text provided IN ANALYZE CONTRACT CONSISTENCY")
            return {"error": "No contract text provided"}
        
        # Perform multi-document analysis if requested
        if multi_document_analysis:
            try:
                # Get all documents from RAG
                all_docs_result = get_all_documents_from_rag(context)
                if all_docs_result.get("success"):
                    documents = all_docs_result.get("documents", [])
                    
                    # Multi-document consistency check
                    multi_doc_result = check_multi_document_consistency(context, documents)
                    analysis_results["multi_doc_analysis"] = multi_doc_result
                    
                    # Case alignment check across documents
                    case_alignment_result = check_case_alignment(context, contract_text)
                    analysis_results["case_analysis"] = case_alignment_result
                    
            except Exception as e:
                analysis_results["multi_doc_analysis"] = {
                    "success": False,
                    "error": f"Multi-document analysis failed: {str(e)}"
                }
        
        # Single document analysis
        try:
            # Analyze clause references
            clause_analysis = analyze_clause_references(context, contract_text)
            analysis_results["clause_analysis"] = clause_analysis
            
            if clause_analysis.get("success"):
                # Check reference validity
                section_refs = clause_analysis.get("references", {}).get("section_ref", [])
                existing_sections = clause_analysis.get("existing_sections", [])
                
                validity_result = check_reference_validity(context, section_refs, existing_sections)
                analysis_results["reference_validity"] = validity_result
                
                # Find circular references
                circular_refs = find_circular_references(context, contract_text)
                analysis_results["circular_references"] = circular_refs
                
                # Validate defined terms
                defined_terms = clause_analysis.get("defined_terms", [])
                term_validation = validate_defined_terms(context, defined_terms, contract_text)
                analysis_results["term_validation"] = term_validation
                
                # Check duplicate definitions
                duplicate_check = check_duplicate_definitions(context, contract_text)
                analysis_results["duplicate_definitions"] = duplicate_check

        except Exception as e:
            analysis_results["single_doc_analysis"] = {
                "success": False,
                "error": f"Single document analysis failed: {str(e)}"
            }
        
        # Generate comprehensive report
        try:
            report_result = generate_consistency_report(context, analysis_results)
            analysis_results["comprehensive_report"] = report_result
        except Exception as e:
            analysis_results["comprehensive_report"] = {
                "success": False,
                "error": f"Report generation failed: {str(e)}"
            }
        
        return {
            "final_output": analysis_results.get("comprehensive_report", {}).get("report", analysis_results),
            "success": True,
            "analysis_results": analysis_results,
            "contract_length": len(contract_text) if contract_text else 0,
            "processing_complete": True,
            "multi_document_enabled": multi_document_analysis,
            "summary": {
                "total_analyses_performed": len([k for k, v in analysis_results.items() if isinstance(v, dict) and v.get("success")]),
                "failed_analyses": len([k for k, v in analysis_results.items() if isinstance(v, dict) and not v.get("success")]),
                "has_multi_document_results": multi_document_analysis and "multi_doc_analysis" in analysis_results,
                "consistency_score": analysis_results.get("comprehensive_report", {}).get("report", {}).get("consistency_score", "Not calculated")
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "analysis_results": {}
        }

# Additional helper function for quick analysis
@function_tool(strict_mode=False) 
def quick_contract_consistency_check(context: RunContextWrapper[ContractAnalysisContext], contract_text: str) -> Dict[str, Any]:
    """
    Quick consistency check for contract text without multi-document analysis
    
    Args:
        contract_text: Contract text to analyze
        
    Returns:
        Dictionary with quick analysis results
    """
    try:
        results = {}
        
        # Basic clause analysis
        clause_result = analyze_clause_references(context, contract_text)
        results["clause_references"] = clause_result
        
        # Check for obvious issues
        if clause_result.get("success"):
            section_refs = clause_result.get("references", {}).get("section_ref", [])
            existing_sections = clause_result.get("existing_sections", [])
            
            # Quick validity check
            missing_refs = [ref for ref in section_refs if ref not in existing_sections]
            results["missing_references"] = missing_refs
            results["missing_count"] = len(missing_refs)
            
            # Quick duplicate check
            duplicate_result = check_duplicate_definitions(context, contract_text)
            results["duplicate_definitions"] = duplicate_result
            
            # Case consistency check
            case_result = check_case_alignment(context, contract_text)
            results["case_consistency"] = case_result
        
        # Calculate quick score
        issues_found = (
            results.get("missing_count", 0) +
            len(results.get("duplicate_definitions", {}).get("duplicate_definitions", {})) +
            (1 if results.get("case_consistency", {}).get("needs_alignment") else 0)
        )
        
        quick_score = max(0, 100 - (issues_found * 10))
        
        return {
            "success": True,
            "quick_analysis": results,
            "quick_score": quick_score,
            "issues_found": issues_found,
            "recommendations": [
                "Run full analysis for comprehensive results",
                "Address missing references" if results.get("missing_count", 0) > 0 else None,
                "Resolve duplicate definitions" if results.get("duplicate_definitions", {}).get("duplicate_definitions") else None,
                "Standardize case formatting" if results.get("case_consistency", {}).get("needs_alignment") else None
            ],
            "final_output": {
                "quick_score": quick_score,
                "issues_found": issues_found
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@function_tool(strict_mode=False)
@handle_errors
def enhanced_document_search(context: RunContextWrapper[ContractAnalysisContext], search_queries: List[str] = None) -> Dict[str, Any]:
    """Enhanced document search for complex inconsistency detection"""
    logger.info("Performing enhanced document search for inconsistencies")
    
    if not search_queries:
        # Default comprehensive search queries
        search_queries = [
            "liability limitation clause",
            "termination clause", 
            "payment terms",
            "confidentiality obligations",
            "dispute resolution",
            "governing law",
            "force majeure",
            "intellectual property",
            "indemnification",
            "warranty disclaimer"
        ]
    
    search_results = {}
    inconsistency_patterns = []
    
    for query in search_queries:
        try:
            result = rag_service.chat(query)
            search_results[query] = {
                "response": result["response"],
                "total_docs": result["metadata"]["total_docs"],
                "text_docs": result["metadata"]["text_docs"]
            }
            
            # Analyze response for inconsistency indicators
            response_text = result["response"]
            
            # Look for conflicting information in the response
            conflict_indicators = [
                r'(?:however|but|although|while|whereas).{1,100}(?:different|inconsistent|conflicting)',
                r'(?:document \w+|file \w+).{1,50}(?:states|says|indicates).{1,100}(?:document \w+|file \w+).{1,50}(?:states|says|indicates)',
                r'(?:version|variation|alternative).{1,50}(?:found|exists|present)'
            ]
            
            for pattern in conflict_indicators:
                matches = re.finditer(pattern, response_text, re.IGNORECASE)
                for match in matches:
                    inconsistency_patterns.append({
                        "query": query,
                        "pattern_type": "conflict_indicator",
                        "matched_text": match.group(),
                        "context": response_text[max(0, match.start()-100):match.end()+100]
                    })
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {str(e)}")
            search_results[query] = {"error": str(e)}
    
    return {
        "success": True,
        "search_results": search_results,
        "potential_inconsistencies": inconsistency_patterns,
        "total_queries_executed": len(search_queries),
        "queries_with_results": len([q for q, r in search_results.items() if "error" not in r]),
        "final_output": {
            "search_summary": f"Executed {len(search_queries)} searches, found {len(inconsistency_patterns)} potential inconsistencies",
            "inconsistency_patterns": inconsistency_patterns
        }
    }

@function_tool(strict_mode=False)
@handle_errors
def deep_case_alignment_analysis(context: RunContextWrapper[ContractAnalysisContext], documents: List[Dict[str, str]]) -> Dict[str, Any]:
    """Perform deep case alignment analysis across all documents with location tracking"""
    logger.info(f"Performing deep case alignment analysis on {len(documents)} documents")
    
    if not documents:
        logger.error("No documents provided for case alignment analysis")
        raise ValueError("No documents provided")
    
    case_issues_by_document = {}
    global_terminology_map = {}
    
    # Enhanced case analysis patterns
    legal_terms = [
        "agreement", "contract", "party", "parties", "section", "clause", "article",
        "exhibit", "schedule", "appendix", "attachment", "whereas", "therefore",
        "liability", "damages", "indemnification", "warranty", "representation",
        "covenant", "breach", "default", "termination", "governing", "jurisdiction"
    ]
    
    for doc in documents:
        doc_name = doc.get("name", "unknown")
        content = doc.get("content", "")
        
        doc_issues = {
            "document_name": doc_name,
            "case_inconsistencies": [],
            "formatting_issues": [],
            "style_violations": []
        }
        
        # Check each legal term for case consistency
        for term in legal_terms:
            # Find all variations of this term
            variations = {}
            patterns = [
                term.lower(),           # lowercase
                term.capitalize(),      # First letter capital
                term.upper(),          # UPPERCASE
                term.title()           # Title Case
            ]
            
            for pattern in patterns:
                matches = list(re.finditer(r'\b' + re.escape(pattern) + r'\b', content))
                if matches:
                    variations[pattern] = []
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        variations[pattern].append({
                            "line_number": line_num,
                            "position": match.start(),
                            "context": content[max(0, match.start()-30):match.end()+30]
                        })
            
            # If multiple variations exist, it's an inconsistency
            if len(variations) > 1:
                doc_issues["case_inconsistencies"].append({
                    "term": term,
                    "variations_found": list(variations.keys()),
                    "occurrences": variations,
                    "total_occurrences": sum(len(v) for v in variations.values())
                })
                
                # Add to global terminology map
                if term not in global_terminology_map:
                    global_terminology_map[term] = {}
                if doc_name not in global_terminology_map[term]:
                    global_terminology_map[term][doc_name] = []
                global_terminology_map[term][doc_name].extend(list(variations.keys()))
        
        # Check for formatting consistency issues
        header_patterns = [
            r'^[A-Z\s]+$',           # ALL CAPS headers
            r'^[A-Z][a-z\s]+$',      # Title case headers  
            r'^\d+\.\s+[A-Z]',       # Numbered headers
            r'^[A-Z]\.\s+[A-Z]'      # Lettered headers
        ]
        
        header_styles = {}
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if len(line) > 3 and len(line) < 100:  # Likely header length
                for pattern in header_patterns:
                    if re.match(pattern, line):
                        if pattern not in header_styles:
                            header_styles[pattern] = []
                        header_styles[pattern].append({
                            "line_number": i,
                            "text": line
                        })
        
        if len(header_styles) > 1:
            doc_issues["formatting_issues"].append({
                "issue_type": "inconsistent_header_styles",
                "styles_found": list(header_styles.keys()),
                "examples": header_styles
            })
        
        case_issues_by_document[doc_name] = doc_issues
    
    # Cross-document terminology analysis
    cross_document_inconsistencies = {}
    for term, doc_variations in global_terminology_map.items():
        if len(doc_variations) > 1:
            # Check if different documents use different cases for the same term
            all_variations = set()
            for doc_name, variations in doc_variations.items():
                all_variations.update(variations)
            
            if len(all_variations) > 1:
                cross_document_inconsistencies[term] = {
                    "total_variations": list(all_variations),
                    "documents_affected": list(doc_variations.keys()),
                    "details": doc_variations
                }
    
    total_issues = sum(
        len(doc_data["case_inconsistencies"]) + len(doc_data["formatting_issues"]) 
        for doc_data in case_issues_by_document.values()
    )
    
    return {
        "success": True,
        "total_documents_analyzed": len(documents),
        "total_case_issues_found": total_issues,
        "documents_with_issues": len([d for d in case_issues_by_document.values() 
                                    if d["case_inconsistencies"] or d["formatting_issues"]]),
        "case_issues_by_document": case_issues_by_document,
        "cross_document_inconsistencies": cross_document_inconsistencies,
        "recommendations": [
            f"Standardize {len(cross_document_inconsistencies)} terms across all documents",
            "Implement consistent header formatting style",
            "Create a style guide for legal terminology capitalization",
            f"Review {len([d for d in case_issues_by_document.values() if d['case_inconsistencies']])} documents with case inconsistencies"
        ],
        "final_output": {
            "summary": f"Found {total_issues} case alignment issues across {len(documents)} documents",
            "critical_terms_needing_standardization": list(cross_document_inconsistencies.keys()),
            "documents_requiring_review": [name for name, data in case_issues_by_document.items() 
                                         if data["case_inconsistencies"] or data["formatting_issues"]]
        }
    }