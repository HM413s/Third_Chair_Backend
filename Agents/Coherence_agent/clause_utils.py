from dotenv import load_dotenv
import re
from typing import Dict, List, Any, Set, Tuple, Optional, Union
from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum
import difflib
from datetime import datetime
import hashlib

load_dotenv()

class SeverityLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class ConsistencyIssue:
    issue_type: str
    severity: SeverityLevel
    description: str
    affected_documents: List[str]
    locations: List[Dict[str, Any]]
    suggested_fix: Optional[str] = None
    confidence_score: float = 0.0

@dataclass
class TermDefinition:
    term: str
    definition: str
    document: str
    section: str
    line_number: int
    context: str
    hash_signature: str

@dataclass
class CrossReference:
    source_doc: str
    target_doc: str
    reference_type: str
    source_location: str
    target_location: str
    is_valid: bool
    confidence: float

class AdvancedClauseAnalyzer:
    """
    Comprehensive analyzer for multi-document legal clause consistency,
    case alignment, cross-references, and structural integrity validation.
    """
    
    def __init__(self, enable_nlp_analysis: bool = True, strictness_level: str = "high"):
        self.strictness_level = strictness_level
        self.enable_nlp_analysis = enable_nlp_analysis
        
        # Enhanced pattern library for comprehensive document analysis
        self.advanced_patterns = {
            # Legal reference patterns with variations
            'section_references': {
                'standard': r'(?:Section|§)\s*(\d+(?:\.\d+)*(?:\([a-z]\))*)',
                'abbreviated': r'(?:Sec\.?|§)\s*(\d+(?:\.\d+)*(?:\([a-z]\))*)',
                'spelled_out': r'(?:Section|SECTION)\s+((?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|\d+)(?:\s+(?:Point|\.)?\s*(?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|\d+))*)',
                'subsection': r'(?:subsection|sub-section|Subsection|Sub-Section)\s*\(([a-z]|\d+|[ivxlcdm]+)\)',
                'paragraph': r'(?:paragraph|Paragraph|¶)\s*\(([a-z]|\d+|[ivxlcdm]+)\)',
                'subparagraph': r'(?:subparagraph|sub-paragraph|Subparagraph)\s*\(([a-z]|\d+|[ivxlcdm]+)\)'
            },
            
            # Article and clause patterns
            'article_references': {
                'standard': r'(?:Article|ARTICLE)\s+([IVX]+|\d+(?:\.\d+)*)',
                'abbreviated': r'(?:Art\.?)\s+([IVX]+|\d+(?:\.\d+)*)',
                'clause': r'(?:Clause|CLAUSE)\s+(\d+(?:\.\d+)*(?:\([a-z]\))*)'
            },
            
            # Defined terms with various formats
            'defined_terms': {
                'quoted': r'"([^"]{2,})"(?:\s+(?:means|shall mean|is defined as|refers to|has the meaning))',
                'capitalized': r'\b([A-Z][A-Z\s]{2,}[A-Z])\b(?=\s+(?:means|shall mean|is defined as))',
                'parenthetical': r'\(as defined (?:herein|in Section \d+(?:\.\d+)*|in the [^)]+)\)',
                'bold_terms': r'\*\*([^*]{2,})\*\*(?:\s+(?:means|shall mean))',
                'italicized': r'\*([^*]{2,})\*(?:\s+(?:means|shall mean))'
            },
            
            # Cross-document references
            'cross_document_refs': {
                'exhibit_schedule': r'(?:Exhibit|Schedule|Appendix|Attachment)\s+([A-Z]|\d+|[IVX]+)(?:\s+(?:hereto|attached|to this Agreement))?',
                'other_agreements': r'(?:the\s+)?([A-Z][a-z\s]+(?:Agreement|Contract|Indenture|Note|Bond|Instrument|Document))(?:\s+dated\s+[^,]+)?',
                'incorporation': r'(?:incorporated|included|attached)\s+(?:herein\s+)?by\s+reference',
                'amendments': r'(?:Amendment|Addendum|Modification|Supplement)\s+(?:No\.\s+)?(\d+|[A-Z]|[IVX]+)',
                'related_docs': r'(?:Master|Framework|Parent|Subsidiary|Ancillary|Related)\s+([A-Z][a-z\s]+(?:Agreement|Contract))'
            },
            
            # Legal terminology patterns
            'legal_terms': {
                'condition_precedent': r'(?:condition precedent|conditions precedent)\s+(?:to|for|of)',
                'force_majeure': r'(?:force majeure|Force Majeure)',
                'governing_law': r'(?:governed by|subject to|under)\s+(?:the laws of|laws of)\s+([^,\.]+)',
                'jurisdiction': r'(?:jurisdiction|courts)\s+of\s+([^,\.]+)',
                'severability': r'(?:severability|severable|unenforceable)',
                'entire_agreement': r'(?:entire agreement|entire understanding|complete agreement)',
                'counterparts': r'(?:executed in counterparts|counterpart execution)',
                'electronic_signature': r'(?:electronic signature|electronic execution|DocuSign|e-signature)'
            },
            
            # Date and time patterns
            'temporal_references': {
                'specific_dates': r'(?:on or before|by|no later than|as of)\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})',
                'relative_dates': r'(?:within\s+)?(\d+)\s+(?:days?|months?|years?)\s+(?:of|from|after|before)',
                'business_days': r'(\d+)\s+(?:business|working|calendar)\s+days?',
                'notice_periods': r'(?:notice|notification)\s+(?:of\s+)?(?:at least\s+)?(\d+)\s+days?'
            },
            
            # Financial and numerical patterns
            'financial_terms': {
                'currency_amounts': r'\$[\d,]+(?:\.\d{2})?(?:\s+(?:million|billion|thousand))?',
                'percentages': r'(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)',
                'interest_rates': r'(?:interest rate|rate)\s+of\s+(\d+(?:\.\d+)?)\s*(?:%|percent|per annum)',
                'thresholds': r'(?:threshold|limit|maximum|minimum)\s+of\s+\$?[\d,]+(?:\.\d{2})?'
            }
        }
        
        # Case variation patterns for consistency checking
        self.case_patterns = {
            'section_cases': [r'Section', r'SECTION', r'section', r'Sec\.', r'SEC\.', r'sec\.', r'§'],
            'article_cases': [r'Article', r'ARTICLE', r'article', r'Art\.', r'ART\.', r'art\.'],
            'clause_cases': [r'Clause', r'CLAUSE', r'clause'],
            'exhibit_cases': [r'Exhibit', r'EXHIBIT', r'exhibit', r'Exh\.', r'EXH\.', r'exh\.'],
            'schedule_cases': [r'Schedule', r'SCHEDULE', r'schedule', r'Sch\.', r'SCH\.', r'sch\.'],
            'appendix_cases': [r'Appendix', r'APPENDIX', r'appendix', r'App\.', r'APP\.', r'app\.'],
            'agreement_cases': [r'Agreement', r'AGREEMENT', r'agreement'],
            'contract_cases': [r'Contract', r'CONTRACT', r'contract'],
            'party_cases': [r'Party', r'PARTY', r'party', r'Parties', r'PARTIES', r'parties']
        }
        
        # Numbering format patterns
        self.numbering_patterns = {
            'arabic': r'\d+',
            'roman_upper': r'[IVX]+',
            'roman_lower': r'[ivx]+',
            'alpha_upper': r'[A-Z]',
            'alpha_lower': r'[a-z]',
            'mixed_decimal': r'\d+\.\d+',
            'parenthetical_alpha': r'\([a-z]\)',
            'parenthetical_numeric': r'\(\d+\)',
            'parenthetical_roman': r'\([ivx]+\)'
        }
        
        # Legal document structure patterns
        self.structure_patterns = {
            'preamble': r'(?:WHEREAS|NOW, THEREFORE|WITNESSETH)',
            'recitals': r'(?:RECITALS?|BACKGROUND|PREMISES)',
            'operative_clauses': r'(?:NOW, THEREFORE|AGREES?D? AS FOLLOWS?|COVENANTS?)',
            'signature_blocks': r'(?:IN WITNESS WHEREOF|EXECUTED|SIGNED)',
            'notarization': r'(?:NOTARY|ACKNOWLEDGED|SWORN|SUBSCRIBED)',
            'exhibits_schedules': r'(?:EXHIBITS?|SCHEDULES?|APPENDICES|ATTACHMENTS)'
        }

    def comprehensive_document_analysis(self, documents: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Perform comprehensive multi-document analysis for consistency issues
        """
        analysis_results = {
            'consistency_issues': [],
            'case_alignment_issues': [],
            'cross_reference_issues': [],
            'defined_term_conflicts': [],
            'structural_inconsistencies': [],
            'numbering_format_issues': [],
            'legal_term_variations': [],
            'temporal_consistency_issues': [],
            'financial_consistency_issues': [],
            'document_hierarchy_issues': [],
            'summary_statistics': {},
            'recommendations': []
        }
        
        # Extract comprehensive document data
        doc_data = self._extract_comprehensive_document_data(documents)
        
        # Perform various consistency checks
        analysis_results['consistency_issues'] = self._analyze_multi_document_consistency(doc_data)
        analysis_results['case_alignment_issues'] = self._analyze_case_alignment(doc_data)
        analysis_results['cross_reference_issues'] = self._validate_cross_references(doc_data)
        analysis_results['defined_term_conflicts'] = self._analyze_defined_term_consistency(doc_data)
        analysis_results['structural_inconsistencies'] = self._analyze_structural_consistency(doc_data)
        analysis_results['numbering_format_issues'] = self._analyze_numbering_consistency(doc_data)
        
        # Generate summary statistics
        analysis_results['summary_statistics'] = self._generate_summary_statistics(analysis_results)
        
        # Generate recommendations
        analysis_results['recommendations'] = self._generate_recommendations(analysis_results)
        
        return analysis_results

    def _extract_comprehensive_document_data(self, documents: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Extract comprehensive data from all documents for analysis
        """
        doc_data = {
            'documents': {},
            'all_references': defaultdict(list),
            'all_defined_terms': defaultdict(list),
            'all_case_variations': defaultdict(Counter),
            'document_structure': {},
            'cross_references': [],
            'numbering_schemes': defaultdict(set),
            'legal_terms': defaultdict(list),
            'temporal_references': defaultdict(list),
            'financial_terms': defaultdict(list)
        }
        
        for doc in documents:
            doc_name = doc.get('name', f'Document_{len(doc_data["documents"])}')
            doc_content = doc.get('content', '')
            doc_lines = doc_content.split('\n')
            
            # Store document metadata
            doc_data['documents'][doc_name] = {
                'content': doc_content,
                'lines': doc_lines,
                'length': len(doc_content),
                'line_count': len(doc_lines),
                'hash': hashlib.md5(doc_content.encode()).hexdigest()
            }
            
            # Extract various patterns
            self._extract_all_references(doc_name, doc_content, doc_data)
            self._extract_defined_terms(doc_name, doc_content, doc_lines, doc_data)
            self._extract_case_variations(doc_name, doc_content, doc_data)
            self._extract_document_structure(doc_name, doc_content, doc_data)
            self._extract_numbering_schemes(doc_name, doc_content, doc_data)
            self._extract_legal_terms(doc_name, doc_content, doc_data)
            self._extract_temporal_references(doc_name, doc_content, doc_data)
            self._extract_financial_terms(doc_name, doc_content, doc_data)
        
        return doc_data

    def _extract_all_references(self, doc_name: str, content: str, doc_data: Dict):
        """Extract all types of references from document"""
        for ref_category, patterns in self.advanced_patterns.items():
            if ref_category in ['section_references', 'article_references', 'cross_document_refs']:
                for pattern_name, pattern in patterns.items():
                    matches = list(re.finditer(pattern, content, re.IGNORECASE))
                    for match in matches:
                        doc_data['all_references'][f"{ref_category}_{pattern_name}"].append({
                            'document': doc_name,
                            'match': match.group(),
                            'start': match.start(),
                            'end': match.end(),
                            'line': content[:match.start()].count('\n') + 1,
                            'context': content[max(0, match.start()-50):match.end()+50]
                        })

    def _extract_defined_terms(self, doc_name: str, content: str, lines: List[str], doc_data: Dict):
        """Extract defined terms with comprehensive analysis"""
        for pattern_name, pattern in self.advanced_patterns['defined_terms'].items():
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                context_start = max(0, line_num - 3)
                context_end = min(len(lines), line_num + 3)
                context = '\n'.join(lines[context_start:context_end])
                
                # Extract the actual definition text
                definition_match = re.search(
                    rf'{re.escape(match.group(1))}\s+(?:means|shall mean|is defined as|refers to)\s+([^.]+\.)',
                    content[match.start():], re.IGNORECASE | re.DOTALL
                )
                
                definition_text = definition_match.group(1) if definition_match else ""
                
                term_def = TermDefinition(
                    term=match.group(1),
                    definition=definition_text.strip(),
                    document=doc_name,
                    section=self._find_containing_section(content, match.start()),
                    line_number=line_num,
                    context=context,
                    hash_signature=hashlib.md5(definition_text.strip().encode()).hexdigest()
                )
                
                doc_data['all_defined_terms'][match.group(1).lower()].append(term_def)

    def _extract_case_variations(self, doc_name: str, content: str, doc_data: Dict):
        """Extract case variations for consistency analysis"""
        for case_type, variations in self.case_patterns.items():
            for variation in variations:
                matches = re.findall(variation, content)
                if matches:
                    doc_data['all_case_variations'][case_type][variation] += len(matches)

    def _extract_document_structure(self, doc_name: str, content: str, doc_data: Dict):
        """Extract document structural elements"""
        structure = {
            'sections': [],
            'articles': [],
            'exhibits': [],
            'schedules': []
        }
        
        # Find all structural elements
        section_pattern = r'(?:^|\n)\s*(?:SECTION|Section)\s+(\d+(?:\.\d+)*)\s*[:\-\.]?\s*([^\n]+)'
        sections = re.finditer(section_pattern, content, re.MULTILINE)
        for section in sections:
            structure['sections'].append({
                'number': section.group(1),
                'title': section.group(2).strip(),
                'position': section.start(),
                'line': content[:section.start()].count('\n') + 1
            })
        
        doc_data['document_structure'][doc_name] = structure

    def _extract_numbering_schemes(self, doc_name: str, content: str, doc_data: Dict):
        """Extract numbering schemes used in the document"""
        for scheme_name, pattern in self.numbering_patterns.items():
            section_numbering = re.findall(rf'Section\s+({pattern})', content, re.IGNORECASE)
            if section_numbering:
                doc_data['numbering_schemes'][f'section_{scheme_name}'].update(section_numbering)

    def _extract_legal_terms(self, doc_name: str, content: str, doc_data: Dict):
        """Extract legal terms and phrases"""
        for term_type, pattern in self.advanced_patterns['legal_terms'].items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                doc_data['legal_terms'][term_type].append({
                    'document': doc_name,
                    'match': match.group(),
                    'position': match.start(),
                    'line': content[:match.start()].count('\n') + 1
                })

    def _extract_temporal_references(self, doc_name: str, content: str, doc_data: Dict):
        """Extract temporal references and deadlines"""
        for temp_type, pattern in self.advanced_patterns['temporal_references'].items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                doc_data['temporal_references'][temp_type].append({
                    'document': doc_name,
                    'match': match.group(),
                    'value': match.group(1) if match.groups() else match.group(),
                    'position': match.start(),
                    'context': content[max(0, match.start()-30):match.end()+30]
                })

    def _extract_financial_terms(self, doc_name: str, content: str, doc_data: Dict):
        """Extract financial terms and amounts"""
        for fin_type, pattern in self.advanced_patterns['financial_terms'].items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                doc_data['financial_terms'][fin_type].append({
                    'document': doc_name,
                    'match': match.group(),
                    'position': match.start(),
                    'context': content[max(0, match.start()-40):match.end()+40]
                })

    def _analyze_multi_document_consistency(self, doc_data: Dict) -> List[ConsistencyIssue]:
        """Analyze consistency across multiple documents"""
        issues = []
        
        # Check for inconsistent section numbering schemes
        section_schemes = defaultdict(set)
        for doc_name in doc_data['documents'].keys():
            structure = doc_data['document_structure'].get(doc_name, {})
            sections = structure.get('sections', [])
            for section in sections:
                number = section['number']
                if '.' in number:
                    scheme = 'decimal'
                elif number.isdigit():
                    scheme = 'sequential'
                else:
                    scheme = 'other'
                section_schemes[doc_name].add(scheme)
        
        # Identify documents with mixed numbering schemes
        for doc_name, schemes in section_schemes.items():
            if len(schemes) > 1:
                issues.append(ConsistencyIssue(
                    issue_type="mixed_numbering_scheme",
                    severity=SeverityLevel.MEDIUM,
                    description=f"Document {doc_name} uses mixed section numbering schemes: {', '.join(schemes)}",
                    affected_documents=[doc_name],
                    locations=[],
                    confidence_score=0.9
                ))
        
        # Check for inconsistent cross-references
        exhibit_refs = doc_data['all_references'].get('cross_document_refs_exhibit_schedule', [])
        exhibit_groups = defaultdict(list)
        for ref in exhibit_refs:
            exhibit_groups[ref['match'].lower()].append(ref)
        
        for exhibit, refs in exhibit_groups.items():
            if len(set(r['document'] for r in refs)) > 1:
                # Same exhibit referenced across multiple documents
                variations = set(r['match'] for r in refs)
                if len(variations) > 1:
                    issues.append(ConsistencyIssue(
                        issue_type="inconsistent_exhibit_reference",
                        severity=SeverityLevel.HIGH,
                        description=f"Exhibit referenced inconsistently across documents: {', '.join(variations)}",
                        affected_documents=list(set(r['document'] for r in refs)),
                        locations=[{'document': r['document'], 'line': r['line'], 'context': r['context']} for r in refs],
                        confidence_score=0.85
                    ))
        
        return issues

    def _analyze_case_alignment(self, doc_data: Dict) -> List[ConsistencyIssue]:
        """Analyze case alignment issues across documents"""
        issues = []
        
        for case_type, variations in doc_data['all_case_variations'].items():
            if len(variations) > 1:
                # Multiple case variations found
                most_common = variations.most_common(1)[0][0]
                total_occurrences = sum(variations.values())
                inconsistent_variations = [(var, count) for var, count in variations.items() if var != most_common]
                
                if inconsistent_variations:
                    severity = SeverityLevel.HIGH if total_occurrences > 10 else SeverityLevel.MEDIUM
                    
                    issues.append(ConsistencyIssue(
                        issue_type="case_inconsistency",
                        severity=severity,
                        description=f"Inconsistent case usage for {case_type.replace('_', ' ')}: {most_common} ({variations[most_common]} times) vs {', '.join([f'{var} ({count})' for var, count in inconsistent_variations])}",
                        affected_documents=list(doc_data['documents'].keys()),
                        locations=[],
                        suggested_fix=f"Standardize all instances to '{most_common}'",
                        confidence_score=0.95
                    ))
        
        return issues

    def _validate_cross_references(self, doc_data: Dict) -> List[ConsistencyIssue]:
        """Validate cross-references between documents"""
        issues = []
        
        # Check for broken internal references
        for doc_name, doc_info in doc_data['documents'].items():
            content = doc_info['content']
            
            # Find all section references
            section_refs = re.finditer(r'Section\s+(\d+(?:\.\d+)*)', content, re.IGNORECASE)
            
            # Get actual sections in the document
            actual_sections = set()
            structure = doc_data['document_structure'].get(doc_name, {})
            for section in structure.get('sections', []):
                actual_sections.add(section['number'])
            
            # Check if referenced sections exist
            for ref_match in section_refs:
                ref_section = ref_match.group(1)
                if ref_section not in actual_sections:
                    issues.append(ConsistencyIssue(
                        issue_type="broken_internal_reference",
                        severity=SeverityLevel.HIGH,
                        description=f"Reference to non-existent Section {ref_section} in {doc_name}",
                        affected_documents=[doc_name],
                        locations=[{
                            'document': doc_name,
                            'line': content[:ref_match.start()].count('\n') + 1,
                            'context': content[max(0, ref_match.start()-30):ref_match.end()+30]
                        }],
                        confidence_score=0.9
                    ))
        
        return issues

    def _analyze_defined_term_consistency(self, doc_data: Dict) -> List[ConsistencyIssue]:
        """Analyze defined term consistency across documents"""
        issues = []
        
        for term_lower, definitions in doc_data['all_defined_terms'].items():
            if len(definitions) > 1:
                # Check if definitions are consistent
                unique_definitions = set(d.hash_signature for d in definitions)
                
                if len(unique_definitions) > 1:
                    # Inconsistent definitions found
                    affected_docs = list(set(d.document for d in definitions))
                    
                    issues.append(ConsistencyIssue(
                        issue_type="inconsistent_defined_term",
                        severity=SeverityLevel.CRITICAL,
                        description=f"Term '{definitions[0].term}' has inconsistent definitions across documents",
                        affected_documents=affected_docs,
                        locations=[{
                            'document': d.document,
                            'section': d.section,
                            'line': d.line_number,
                            'definition': d.definition[:100] + "..." if len(d.definition) > 100 else d.definition
                        } for d in definitions],
                        confidence_score=0.95
                    ))
                
                # Check for case inconsistencies in term usage
                term_variations = set(d.term for d in definitions)
                if len(term_variations) > 1:
                    issues.append(ConsistencyIssue(
                        issue_type="defined_term_case_inconsistency",
                        severity=SeverityLevel.MEDIUM,
                        description=f"Defined term has case variations: {', '.join(term_variations)}",
                        affected_documents=list(set(d.document for d in definitions)),
                        locations=[],
                        suggested_fix=f"Standardize to most common variation",
                        confidence_score=0.8
                    ))
        
        return issues

    def _analyze_structural_consistency(self, doc_data: Dict) -> List[ConsistencyIssue]:
        """Analyze structural consistency across documents"""
        issues = []
        
        # Check for consistent section organization
        section_structures = {}
        for doc_name, structure in doc_data['document_structure'].items():
            sections = structure.get('sections', [])
            section_numbers = [s['number'] for s in sections]
            section_structures[doc_name] = section_numbers
        
        # Find documents with similar section structures that might be inconsistent
        doc_names = list(section_structures.keys())
        for i, doc1 in enumerate(doc_names):
            for doc2 in doc_names[i+1:]:
                similarity = self._calculate_structure_similarity(
                    section_structures[doc1], 
                    section_structures[doc2]
                )
                
                if 0.3 < similarity < 0.8:  # Partially similar but not identical
                    issues.append(ConsistencyIssue(
                        issue_type="partially_inconsistent_structure",
                        severity=SeverityLevel.MEDIUM,
                        description=f"Documents {doc1} and {doc2} have similar but inconsistent section structures",
                        affected_documents=[doc1, doc2],
                        locations=[],
                        confidence_score=similarity
                    ))
        
        return issues

    def _analyze_numbering_consistency(self, doc_data: Dict) -> List[ConsistencyIssue]:
        """Analyze numbering format consistency"""
        issues = []
        
        # Check for consistent numbering schemes across documents
        scheme_usage = defaultdict(set)
        for scheme, docs in doc_data['numbering_schemes'].items():
            if docs:
                scheme_type = scheme.split('_')[1]  # e.g., 'arabic', 'roman_upper'
                for doc_name in doc_data['documents'].keys():
                    # Check if this document uses this scheme
                    doc_content = doc_data['documents'][doc_name]['content']
                    if any(num in doc_content for num in docs):
                        scheme_usage[doc_name].add(scheme_type)
        
        # Find documents using multiple schemes
        for doc_name, schemes in scheme_usage.items():
            if len(schemes) > 1:
                issues.append(ConsistencyIssue(
                    issue_type="mixed_numbering_formats",
                    severity=SeverityLevel.MEDIUM,
                    description=f"Document {doc_name} uses multiple numbering formats: {', '.join(schemes)}",
                    affected_documents=[doc_name],
                    locations=[],
                    suggested_fix="Use consistent numbering format throughout document",
                    confidence_score=0.7
                ))
        
        return issues

    def _calculate_structure_similarity(self, structure1: List[str], structure2: List[str]) -> float:
        """Calculate similarity between two document structures"""
        if not structure1 or not structure2:
            return 0.0
        
        # Convert to sets for comparison
        set1 = set(structure1)
        set2 = set(structure2)
        
        # Calculate Jaccard similarity
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        jaccard_similarity = intersection / union
        
        # Also consider sequence similarity (order matters)
        common_subsequence_length = self._longest_common_subsequence_length(structure1, structure2)
        max_length = max(len(structure1), len(structure2))
        sequence_similarity = common_subsequence_length / max_length if max_length > 0 else 0.0
        
        # Weighted average of both similarities
        return (jaccard_similarity * 0.6) + (sequence_similarity * 0.4)

    def _longest_common_subsequence_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate the length of the longest common subsequence"""
        m, n = len(seq1), len(seq2)
        
        # Create a DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill the DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]

    def _generate_summary_statistics(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics from analysis results"""
        stats = {
            'total_issues': 0,
            'critical_issues': 0,
            'high_severity_issues': 0,
            'medium_severity_issues': 0,
            'low_severity_issues': 0,
            'issue_breakdown': {},
            'most_common_issues': [],
            'affected_documents': set(),
            'confidence_distribution': {
                'high_confidence': 0,  # > 0.8
                'medium_confidence': 0,  # 0.5 - 0.8
                'low_confidence': 0   # < 0.5
            }
        }
        
        # Count issues by category and severity
        for category, issues in analysis_results.items():
            if isinstance(issues, list) and issues:
                stats['issue_breakdown'][category] = len(issues)
                stats['total_issues'] += len(issues)
                
                for issue in issues:
                    if hasattr(issue, 'severity'):
                        severity = issue.severity.value if hasattr(issue.severity, 'value') else str(issue.severity)
                        if severity == 'critical':
                            stats['critical_issues'] += 1
                        elif severity == 'high':
                            stats['high_severity_issues'] += 1
                        elif severity == 'medium':
                            stats['medium_severity_issues'] += 1
                        elif severity == 'low':
                            stats['low_severity_issues'] += 1
                    
                    if hasattr(issue, 'affected_documents'):
                        stats['affected_documents'].update(issue.affected_documents)
                    
                    if hasattr(issue, 'confidence_score'):
                        confidence = issue.confidence_score
                        if confidence > 0.8:
                            stats['confidence_distribution']['high_confidence'] += 1
                        elif confidence > 0.5:
                            stats['confidence_distribution']['medium_confidence'] += 1
                        else:
                            stats['confidence_distribution']['low_confidence'] += 1
        
        # Convert set to list for JSON serialization
        stats['affected_documents'] = list(stats['affected_documents'])
        
        # Find most common issue types
        sorted_issues = sorted(stats['issue_breakdown'].items(), key=lambda x: x[1], reverse=True)
        stats['most_common_issues'] = sorted_issues[:5]
        
        return stats

    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis results"""
        recommendations = []
        
        # Check for critical issues
        critical_count = 0
        high_count = 0
        
        for category, issues in analysis_results.items():
            if isinstance(issues, list):
                for issue in issues:
                    if hasattr(issue, 'severity'):
                        severity = issue.severity.value if hasattr(issue.severity, 'value') else str(issue.severity)
                        if severity == 'critical':
                            critical_count += 1
                        elif severity == 'high':
                            high_count += 1
        
        # Generate priority-based recommendations
        if critical_count > 0:
            recommendations.append(
                f"URGENT: Address {critical_count} critical issue(s) immediately. "
                "These may affect document enforceability."
            )
        
        if high_count > 0:
            recommendations.append(
                f"HIGH PRIORITY: Resolve {high_count} high-severity issue(s). "
                "These issues may cause confusion or legal ambiguity."
            )
        
        # Issue-specific recommendations
        issue_counts = analysis_results.get('summary_statistics', {}).get('issue_breakdown', {})
        
        if issue_counts.get('case_alignment_issues', 0) > 0:
            recommendations.append(
                "Standardize capitalization and formatting of legal terms across all documents. "
                "Use a style guide to ensure consistency."
            )
        
        if issue_counts.get('defined_term_conflicts', 0) > 0:
            recommendations.append(
                "Review and reconcile conflicting term definitions. "
                "Consider creating a master definitions schedule."
            )
        
        if issue_counts.get('cross_reference_issues', 0) > 0:
            recommendations.append(
                "Validate all cross-references and ensure referenced sections exist. "
                "Update or remove broken references."
            )
        
        if issue_counts.get('structural_inconsistencies', 0) > 0:
            recommendations.append(
                "Standardize document structure and section numbering schemes. "
                "Use consistent formatting templates."
            )
        
        # General recommendations
        if len(recommendations) == 0:
            recommendations.append("Document analysis complete. No major issues detected.")
        else:
            recommendations.append(
                "Consider implementing a document review checklist to prevent future inconsistencies."
            )
        
        return recommendations