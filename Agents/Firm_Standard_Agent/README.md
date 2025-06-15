# 🎨 Firm Standard Agent

## Overview

The **Firm Standard Agent** is a specialized AI agent designed for comprehensive language formatting and style analysis of legal documents. It focuses on detecting and reporting inconsistencies in:

- **Punctuation** (Oxford commas, quotation marks, list formatting)
- **Capitalization** (legal terms, headers, sentence case)
- **Sentence Structure** (length, fragments, run-ons)
- **Word Choice** (legal terminology standardization)
- **Cross-document Style Compliance**

## Key Features

### 🔤 Punctuation Analysis
- **Oxford comma consistency** - Detects mixed usage of serial commas
- **Quotation mark standardization** - Identifies inconsistent quote styles
- **List formatting** - Checks for consistent bullet points and list separators
- **Sentence endings** - Validates proper punctuation usage

### 🔠 Capitalization Analysis  
- **Legal term consistency** - Ensures uniform capitalization of legal terminology
- **Header standardization** - Identifies mixed header capitalization styles
- **Sentence case validation** - Checks for proper sentence capitalization
- **Proper noun consistency** - Validates company names and legal entities

### 📝 Sentence Structure Analysis
- **Length optimization** - Identifies overly long sentences (>40 words)
- **Fragment detection** - Finds incomplete sentences
- **Run-on identification** - Detects sentences with multiple unconnected clauses
- **Readability assessment** - Provides average sentence length metrics

### 🔤 Word Choice Analysis
- **Legal terminology standardization** - Ensures consistent use of preferred legal terms
- **Cross-document terminology** - Identifies terminology conflicts across documents
- **Firm-specific standards** - Enforces company-specific word choice guidelines
- **Alternative term detection** - Finds and suggests replacements for non-standard terms

### 📊 Style Compliance Reporting
- **Comprehensive scoring** - Provides overall style compliance percentage
- **Issue categorization** - Breaks down problems by type and severity
- **Location tracking** - Provides exact line numbers and file references
- **Actionable recommendations** - Suggests specific improvements

## Architecture

```
Firm_Standard_Agent/
├── __init__.py                    # Module initialization
├── context.py                     # Pydantic context model for shared state
├── tools.py                       # All analysis tools and functions
├── Firm_Standard_Agent.py         # Main agent definition and automation
├── demo.py                        # Demonstration script
└── README.md                      # This documentation
```

## Quick Start

### One-Click Analysis

```python
from Agents.Firm_Standard_Agent.Firm_Standard_Agent import firm_standard_agent_auto

# Run complete automated style analysis
result = firm_standard_agent_auto()
```

### Demo Script

```bash
cd backend/Agents/Firm_Standard_Agent
python demo.py
```

## Analysis Workflow

The agent follows a **6-step automated workflow**:

1. **Document Retrieval** - `get_documents_for_style_analysis()`
2. **Punctuation Analysis** - `analyze_punctuation_consistency()`  
3. **Capitalization Check** - `analyze_capitalization_patterns()`
4. **Sentence Structure** - `analyze_sentence_structure_consistency()`
5. **Word Choice Review** - `analyze_word_choice_consistency()`
6. **Final Report** - `generate_style_compliance_report()`

## Tool Functions

### Core Analysis Tools

#### `get_documents_for_style_analysis()`
- Retrieves all documents from RAG storage
- Processes document metadata (word count, character count)
- Updates context with document information

#### `analyze_punctuation_consistency()`
- Checks Oxford comma usage patterns
- Validates quotation mark consistency  
- Analyzes list formatting styles
- Provides location-specific recommendations

#### `analyze_capitalization_patterns()`
- Scans legal term capitalization
- Validates header consistency
- Checks sentence case compliance
- Tracks variations with line numbers

#### `analyze_sentence_structure_consistency()`
- Measures sentence length distribution
- Identifies fragments and run-ons
- Calculates readability metrics
- Provides structural recommendations

#### `analyze_word_choice_consistency()`
- Enforces legal terminology standards
- Detects cross-document inconsistencies
- Tracks preferred term usage
- Maps alternative terminology

#### `generate_style_compliance_report()`
- Calculates overall compliance score
- Categorizes issues by type and severity
- Provides executive summary
- Generates actionable recommendations

## Configuration

### Environment Variables
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

### Legal Terminology Standards

The agent uses predefined legal terminology standards:

```python
legal_terminology_standards = {
    "contract_synonyms": {
        "preferred": "agreement",
        "alternatives": ["contract", "deal", "arrangement", "understanding"]
    },
    "party_references": {
        "preferred": "Party", 
        "alternatives": ["party", "parties", "contracting party", "signatory"]
    },
    "obligation_terms": {
        "preferred": "shall",
        "alternatives": ["will", "must", "should", "ought to"]
    },
    "time_references": {
        "preferred": "immediately",
        "alternatives": ["right away", "at once", "forthwith", "without delay"]
    }
}
```

## Output Format

### Compliance Score
- **95-100%**: Excellent compliance
- **85-94%**: Good compliance  
- **70-84%**: Fair compliance
- **<70%**: Needs improvement

### Issue Location Tracking
Each issue includes:
- **File name** - Exact document reference
- **Line number** - Precise location in document
- **Position** - Character position in file
- **Context** - Surrounding text snippet
- **Recommendation** - Specific fix suggestion

### Sample Output
```json
{
  "executive_summary": {
    "total_documents_analyzed": 22,
    "total_issues_found": 15,
    "compliance_score": 87.5,
    "compliance_level": "Good",
    "issue_breakdown": {
      "punctuation": 3,
      "capitalization": 7,
      "sentence_structure": 2,
      "word_choice": 3
    }
  },
  "detailed_findings": {
    "punctuation": [...],
    "capitalization": [...],
    "sentence_structure": [...],
    "word_choice": [...]
  },
  "recommendations": [
    "Establish consistent punctuation style guide",
    "Standardize legal term capitalization",
    "Break down overly long sentences",
    "Implement firm-wide terminology standards"
  ]
}
```

## Integration with RAG System

The agent integrates seamlessly with the existing RAG infrastructure:

- **Document Retrieval**: Uses `rag_service.get_all_documents()`
- **Storage Stats**: Accesses `rag_service.get_storage_stats()`
- **Context Sharing**: Maintains state through Pydantic models
- **Error Handling**: Comprehensive logging and fallback mechanisms

## Logging

Comprehensive logging to `firm_standard_analysis.log`:

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('firm_standard_analysis.log'),
        logging.StreamHandler()
    ]
)
```

## Error Handling

All tools include robust error handling:
- Function execution timing
- Detailed error logging
- Graceful failure recovery
- Context preservation on errors

## Use Cases

### Legal Document Review
- **Contract standardization** across multiple agreements
- **Style guide enforcement** for law firm documents
- **Quality assurance** before document finalization
- **Cross-document consistency** checking

### Compliance Monitoring
- **Firm-wide style standards** enforcement
- **Client-specific formatting** requirements
- **Regulatory compliance** for document formatting
- **Training material** for legal writing standards

## Dependencies

- `agents` - Core agent framework
- `RAG` - Document storage and retrieval
- `pydantic` - Data validation and context management
- `openai` - Gemini API integration
- `python-dotenv` - Environment variable management

## Future Enhancements

- **Custom style guides** - Client-specific formatting rules
- **Machine learning** - Adaptive style preference learning
- **Integration APIs** - Direct document editor integration
- **Real-time analysis** - Live document checking during editing

---

**Note**: This agent complements the Coherence Agent by focusing specifically on language formatting and style, while the Coherence Agent handles structural consistency and cross-references. 