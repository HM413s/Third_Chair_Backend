from typing import List, Dict, Any
from RAG.document_processor import DocumentProcessor
from RAG.vector_store import ZillizVectorStore
import numpy as np
from rank_bm25 import BM25Okapi
from datetime import datetime
import os

class RAGService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
            cls._instance.document_processor = DocumentProcessor()
            cls._instance.vector_store = ZillizVectorStore()
            cls._instance._fit_vectorizer()
        return cls._instance
    
    def __init__(self):
        # Initialize only if not already initialized
        if not hasattr(self, 'document_processor'):
            self.document_processor = DocumentProcessor()
            self.vector_store = ZillizVectorStore()
            self._fit_vectorizer()
    
    def _fit_vectorizer(self):
        """Fit the vectorizer with all documents in the collection"""
        try:
            # Get all documents from the collection
            all_docs = self.vector_store.get_all_documents()
            print(f"Found {len(all_docs)} documents for vectorizer fitting")
            
            if all_docs:
                # Extract text content
                texts = [doc["content"] for doc in all_docs if doc["type"] == "text"]
                print(f"Extracted {len(texts)} text documents for vectorizer fitting")
                
                if texts:
                    # Fit vectorizer with all documents
                    self.document_processor.vectorizer.fit(texts)
                    print("Vectorizer fitted successfully")
                    
                    # Update BM25
                    self.document_processor.documents = texts
                    tokenized_docs = [text.split() for text in texts]
                    self.document_processor.bm25 = BM25Okapi(tokenized_docs)
                    print("BM25 updated successfully")
        except Exception as e:
            print(f"Error fitting vectorizer: {str(e)}")
    
    def process_and_store_document(self, file_path: str, source_file: str = None):
        """Process a document and store it with enhanced location tracking"""
        try:
            # Use provided source_file or extract from file_path
            actual_source = source_file or os.path.basename(file_path)
            processed_docs = self.document_processor.process_file(file_path, source_file=actual_source)
            print(f"Processed {len(processed_docs)} documents from {actual_source}")
            
            if processed_docs:
                # Store documents with enhanced location metadata
                self.vector_store.add_documents(processed_docs, source_file=actual_source)
                print(f"Stored {len(processed_docs)} documents in vector store with location tracking")
                
                # Print location summary for debugging
                location_summary = self._summarize_document_locations(processed_docs)
                print(f"Location Summary for {actual_source}:")
                for key, value in location_summary.items():
                    print(f"  {key}: {value}")
                
                # Refit vectorizer with new documents
                self._fit_vectorizer()
            return len(processed_docs)
        except Exception as e:
            print(f"Error processing document: {str(e)}")
            return 0
    
    def _summarize_document_locations(self, processed_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a summary of document locations for debugging and validation"""
        summary = {
            "total_chunks": len(processed_docs),
            "document_types": {},
            "sections_found": set(),
            "line_range": {"min": float('inf'), "max": 0},
            "page_range": {"min": float('inf'), "max": 0},
            "content_analysis": {
                "chunks_with_references": 0,
                "chunks_with_definitions": 0,
                "list_items": 0
            }
        }
        
        for doc in processed_docs:
            doc_type = doc.get("type", "unknown")
            summary["document_types"][doc_type] = summary["document_types"].get(doc_type, 0) + 1
            
            metadata = doc.get("metadata", {})
            containing_section = metadata.get("containing_section", {})
            location_context = metadata.get("location_context", {})
            
            # Track sections
            if containing_section.get("type") and containing_section.get("number"):
                section_ref = f"{containing_section['type']} {containing_section['number']}"
                summary["sections_found"].add(section_ref)
            
            # Track line and page ranges
            line_num = metadata.get("line_number", 0)
            page_num = metadata.get("page_number", 0)
            
            if line_num > 0:
                summary["line_range"]["min"] = min(summary["line_range"]["min"], line_num)
                summary["line_range"]["max"] = max(summary["line_range"]["max"], line_num)
            
            if page_num > 0:
                summary["page_range"]["min"] = min(summary["page_range"]["min"], page_num)
                summary["page_range"]["max"] = max(summary["page_range"]["max"], page_num)
            
            # Content analysis
            if metadata.get("contains_reference"):
                summary["content_analysis"]["chunks_with_references"] += 1
            if metadata.get("contains_definition"):
                summary["content_analysis"]["chunks_with_definitions"] += 1
            if metadata.get("is_list_item"):
                summary["content_analysis"]["list_items"] += 1
        
        # Convert sets and handle infinite values
        summary["sections_found"] = list(summary["sections_found"])
        if summary["line_range"]["min"] == float('inf'):
            summary["line_range"]["min"] = 0
        if summary["page_range"]["min"] == float('inf'):
            summary["page_range"]["min"] = 0
        
        return summary

    def generate_response(self, query: str, relevant_docs: List[Dict[str, Any]]) -> str:
        """Generate a response by combining relevant document chunks with location context"""
        if not relevant_docs:
            return "I couldn't find any relevant information to answer your question. Let me provide a general response based on my knowledge."
        
        # Sort documents by relevance score
        relevant_docs.sort(key=lambda x: x.get('score', 0), reverse=True)
        print(f"Found {len(relevant_docs)} relevant documents")
        
        # Combine relevant text chunks with location information
        text_chunks = []
        for doc in relevant_docs:
            if doc["type"] == "text":
                location_info = doc.get("location", {})
                source_file = doc.get("source_file", "unknown")
                
                # Create location reference
                location_ref = ""
                if location_info.get("section_path"):
                    location_ref = f"[{source_file}: {location_info['section_path']}]"
                elif location_info.get("line_number"):
                    location_ref = f"[{source_file}: Line {location_info['line_number']}]"
                else:
                    location_ref = f"[{source_file}]"
                
                text_chunks.append({
                    "content": doc["content"],
                    "location_ref": location_ref,
                    "location": location_info
                })
        
        if not text_chunks:
            return "I found some images but no relevant text to answer your question. Let me provide a general response based on my knowledge."
        
        # Create a structured response
        response = "Based on the available information:\n\n"
        
        # Group chunks by source file
        chunks_by_source = {}
        for chunk in text_chunks:
            source = chunk['location_ref'].split(':')[0].strip('[]')
            if source not in chunks_by_source:
                chunks_by_source[source] = []
            chunks_by_source[source].append(chunk)
        
        # Format response by source
        for i, (source, chunks) in enumerate(chunks_by_source.items(), 1):
            response += f"{i}. From {source}:\n"
            for chunk in chunks:
                # Clean up the content
                content = chunk['content'].strip()
                # Remove duplicate information
                if content not in response:
                    response += f"   • {content}\n"
            response += "\n"
        
        # Add note about images if present
        image_count = sum(1 for doc in relevant_docs if doc["type"] == "image")
        if image_count > 0:
            response += f"\nI also found {image_count} relevant image(s) that might help illustrate this information.\n"
        
        # Add a section for LLM-generated response
        response += "\n## Additional Insights\n\n"
        response += f"Based on the information above and my general knowledge, here's my answer to your question about '{query}':\n\n"
        response += "I'll combine the retrieved information with my knowledge to provide a comprehensive answer.\n\n"
        
        # Process through Google's Gemini model for better formatting and to add LLM-generated content
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain.prompts import ChatPromptTemplate
            from langchain.schema import HumanMessage, SystemMessage
            
            # Initialize the Gemini model
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0.7,
                convert_system_message_to_human=True
            )
            
            # Create the prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant that formats responses in clear, well-structured markdown.
                Follow these formatting rules strictly:
                1. Use **bold** for names and section headers
                2. Use *italics* for emphasis
                3. Use bullet points (*) for lists
                4. Use proper markdown links: [text](url)
                5. Use proper indentation for nested lists
                6. Use horizontal rules (---) to separate major sections
                7. Use code blocks (```) for technical content
                8. Preserve all markdown symbols in the output
                9. Ensure proper spacing between sections
                10. Use proper heading levels (# for main, ## for sub)
                
                IMPORTANT: You must provide a comprehensive answer that combines both the retrieved information AND your own knowledge.
                Don't just summarize the retrieved information - add value by providing additional context, explanations, or insights.
                Make sure to directly address the user's query: {query}
                """),
                ("human", """Format the following information in clear markdown, preserving all markdown symbols, and add your own insights to create a comprehensive response:

                {input_text}

                Ensure the output is well-structured and easy to read, with proper markdown formatting for all elements.
                Remember to add your own knowledge and insights to provide a complete answer to the query: {query}""")
            ])
            
            # Get formatted response from Gemini
            chain = prompt | llm
            formatted_response = chain.invoke({"input_text": response, "query": query}).content
            
            return formatted_response
            
        except Exception as e:
            print(f"Error formatting with Gemini: {str(e)}")
            # Return the original formatted response if Gemini processing fails
        return response
    
    def chat(self, query: str) -> Dict[str, Any]:
        """Process a chat query and return relevant documents with enhanced location metadata"""
        try:
            print(f"\nProcessing query: {query}")
            query_embedding = self.document_processor.vectorizer.transform([query]).toarray()[0].tolist()
            # Ensure embedding is exactly 4000 dimensions
            if len(query_embedding) > 4000:
                query_embedding = query_embedding[:4000]
            elif len(query_embedding) < 4000:
                query_embedding = query_embedding + [0.0] * (4000 - len(query_embedding))
            print("Generated query embedding with correct dimensions")
            
            # Increase the limit to get more potentially relevant documents
            relevant_docs = self.vector_store.hybrid_search(
                query_embedding=query_embedding,
                text_query=query,
                limit=10  # Increased from 5 to 10
            )
            print(f"Found {len(relevant_docs)} relevant documents in hybrid search")
            
            # Even if no documents are found, we'll still generate a response using the LLM
            images = []
            text_docs = []
            location_metadata = []
            
            for doc in relevant_docs:
                # Pass through all metadata fields including enhanced location data
                meta = doc.get("metadata", {})
                location = doc.get("location", {})
                content_flags = doc.get("content_flags", {})
                
                if doc["type"] == "text":
                    enhanced_doc = {
                        **doc, 
                        "metadata": meta,
                        "location": location,
                        "content_flags": content_flags
                    }
                    text_docs.append(enhanced_doc)
                    
                    # Collect location metadata for agents
                    location_metadata.append({
                        "source_file": doc.get("source_file", "unknown"),
                        "location": location,
                        "content_flags": content_flags,
                        "chunk_info": {
                            "content_preview": doc["content"][:100] + "..." if len(doc["content"]) > 100 else doc["content"],
                            "word_count": content_flags.get("word_count", 0)
                        }
                    })
                    
                elif doc["type"] == "image" and doc.get("image_base64"):
                    images.append({
                        "base64": doc["image_base64"],
                        "content": doc["content"],
                        "metadata": meta,
                        "location": location
                    })
            
            print(f"Separated into {len(text_docs)} text documents and {len(images)} images")
            
            # Generate response even if no relevant documents are found
            response = self.generate_response(query, text_docs)
            
            return {
                "response": response,
                "images": images,
                "metadata": {
                    "total_docs": len(relevant_docs),
                    "text_docs": len(text_docs),
                    "image_docs": len(images),
                    "combined_response": True  # Flag indicating this is a combined RAG+LLM response
                },
                "chunks": text_docs,  # include all text chunks with full metadata
                "location_summary": {
                    "documents_by_source": self._group_by_source(location_metadata),
                    "sections_referenced": self._extract_sections_referenced(location_metadata),
                    "content_types_found": self._analyze_content_types(location_metadata)
                }
            }
        except Exception as e:
            print(f"DEBUG: Exception caught in RAGService.chat: {str(e)}")
            # Even if there's an error, try to generate a response using just the LLM
            try:
                fallback_response = "I couldn't retrieve specific information from the documents, but I can still try to answer your question based on my general knowledge.\n\n"
                fallback_response += self.generate_response(query, [])
                return {
                    "response": fallback_response,
                    "images": [],
                    "metadata": {
                        "total_docs": 0,
                        "text_docs": 0,
                        "image_docs": 0,
                        "fallback_mode": True,
                        "error": str(e)
                    },
                    "chunks": [],
                    "location_summary": {}
                }
            except Exception as fallback_error:
                return {
                            "response": f"Error processing your query: {str(e)}. Additionally, fallback response generation failed: {str(fallback_error)}",
                    "images": [],
                    "metadata": {
                        "total_docs": 0,
                        "text_docs": 0,
                                "image_docs": 0,
                                "error": str(e),
                                "fallback_error": str(fallback_error)
                    },
                    "chunks": [],
                    "location_summary": {}
                }
            

    def _group_by_source(self, location_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Group location metadata by source file for easier agent analysis"""
        grouped = {}
        for meta in location_metadata:
            source = meta["source_file"]
            if source not in grouped:
                grouped[source] = {
                    "chunk_count": 0,
                    "sections": set(),
                    "line_range": {"min": float('inf'), "max": 0},
                    "content_analysis": {
                        "references": 0,
                        "definitions": 0,
                        "list_items": 0
                    }
                }
            
            group = grouped[source]
            group["chunk_count"] += 1
            
            location = meta.get("location", {})
            content_flags = meta.get("content_flags", {})
            
            # Track sections
            if location.get("section_path"):
                group["sections"].add(location["section_path"])
            
            # Track line range
            line_num = location.get("line_number", 0)
            if line_num > 0:
                group["line_range"]["min"] = min(group["line_range"]["min"], line_num)
                group["line_range"]["max"] = max(group["line_range"]["max"], line_num)
            
            # Content analysis
            if content_flags.get("contains_reference"):
                group["content_analysis"]["references"] += 1
            if content_flags.get("contains_definition"):
                group["content_analysis"]["definitions"] += 1
            if content_flags.get("is_list_item"):
                group["content_analysis"]["list_items"] += 1
        
        # Convert sets and handle infinite values
        for group in grouped.values():
            group["sections"] = list(group["sections"])
            if group["line_range"]["min"] == float('inf'):
                group["line_range"]["min"] = 0
        
        return grouped

    def _extract_sections_referenced(self, location_metadata: List[Dict[str, Any]]) -> List[str]:
        """Extract unique sections referenced in the results"""
        sections = set()
        for meta in location_metadata:
            location = meta.get("location", {})
            if location.get("section_path"):
                sections.add(location["section_path"])
        return list(sections)

    def _analyze_content_types(self, location_metadata: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze content types found in the results"""
        analysis = {
            "chunks_with_references": 0,
            "chunks_with_definitions": 0,
            "list_items": 0,
            "total_chunks": len(location_metadata)
        }
        
        for meta in location_metadata:
            content_flags = meta.get("content_flags", {})
            if content_flags.get("contains_reference"):
                analysis["chunks_with_references"] += 1
            if content_flags.get("contains_definition"):
                analysis["chunks_with_definitions"] += 1
            if content_flags.get("is_list_item"):
                analysis["list_items"] += 1
        
        return analysis

    def search_by_location(self, location_criteria: Dict[str, Any], limit: int = 10) -> Dict[str, Any]:
        """Search documents by specific location criteria for agent targeting"""
        try:
            print(f"Searching by location criteria: {location_criteria}")
            
            documents = self.vector_store.search_by_location(location_criteria, limit)
            
            # Organize results for agent consumption
            text_docs = []
            images = []
            
            for doc in documents:
                if doc["type"] == "text":
                    text_docs.append(doc)
                elif doc["type"] == "image":
                    images.append(doc)
            
            location_summary = {
                "search_criteria": location_criteria,
                "results_by_source": self._group_by_source([{
                    "source_file": doc.get("source_file", "unknown"),
                    "location": doc.get("location", {}),
                    "content_flags": doc.get("content_flags", {})
                } for doc in documents]),
                "total_matches": len(documents)
            }
            
            return {
                "documents": documents,
                "text_docs": text_docs,
                "images": images,
                "location_summary": location_summary,
                "metadata": {
                    "total_docs": len(documents),
                    "text_docs": len(text_docs),
                    "image_docs": len(images),
                    "search_type": "location_based"
                }
            }
        except Exception as e:
            print(f"Error in location-based search: {str(e)}")
            return {
                "documents": [],
                "text_docs": [],
                "images": [],
                "location_summary": {},
                "metadata": {"error": str(e)}
            }

    def get_documents_by_section(self, section_type: str, section_number: str = None, source_file: str = None) -> Dict[str, Any]:
        """Get documents from specific sections for agent analysis"""
        criteria = {"section_type": section_type}
        
        if section_number:
            criteria["section_number"] = section_number
        if source_file:
            criteria["source_file"] = source_file
        
        return self.search_by_location(criteria)

    def get_documents_with_references(self, source_file: str = None) -> Dict[str, Any]:
        """Get documents that contain references for coherence analysis"""
        criteria = {"contains_reference": True}
        if source_file:
            criteria["source_file"] = source_file
        
        return self.search_by_location(criteria)

    def get_documents_with_definitions(self, source_file: str = None) -> Dict[str, Any]:
        """Get documents that contain definitions for consistency analysis"""
        criteria = {"contains_definition": True}
        if source_file:
            criteria["source_file"] = source_file
        
        return self.search_by_location(criteria)

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about stored documents with enhanced location information"""
        try:
            stats = self.vector_store.get_document_stats()
            
            # Add additional analysis for agents
            stats["agent_insights"] = {
                "location_coverage": self._analyze_location_coverage(stats),
                "content_distribution": self._analyze_content_distribution(stats),
                "document_completeness": self._analyze_document_completeness(stats)
            }
            
            return stats
        except Exception as e:
            print(f"Error getting stats: {str(e)}")
            return {
                "total_documents": 0,
                "document_types": {},
                "sources": {},
                "creation_dates": {},
                "location_stats": {},
                "agent_insights": {}
            }

    def _analyze_location_coverage(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze location coverage for agent insights"""
        location_stats = stats.get("location_stats", {})
        
        return {
            "sections_mapped": len(location_stats.get("sections_by_type", {})),
            "document_types_processed": len(location_stats.get("documents_by_document_type", {})),
            "files_with_line_tracking": len(location_stats.get("line_coverage", {})),
            "files_with_page_tracking": len(location_stats.get("page_coverage", {}))
        }

    def _analyze_content_distribution(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content distribution for agent decision making"""
        location_stats = stats.get("location_stats", {})
        content_analysis = location_stats.get("content_analysis", {})
        
        total_docs = stats.get("total_documents", 1)  # Avoid division by zero
        
        return {
            "reference_density": content_analysis.get("documents_with_references", 0) / total_docs,
            "definition_density": content_analysis.get("documents_with_definitions", 0) / total_docs,
            "list_item_ratio": content_analysis.get("list_items", 0) / total_docs,
            "structural_completeness": min(1.0, len(location_stats.get("sections_by_type", {})) / 5)  # Assume 5 section types is good coverage
        }

    def _analyze_document_completeness(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze document completeness for quality assessment"""
        location_stats = stats.get("location_stats", {})
        
        return {
            "files_processed": len(stats.get("sources", {})),
            "total_chunks": stats.get("total_documents", 0),
            "avg_chunks_per_file": stats.get("total_documents", 0) / max(1, len(stats.get("sources", {}))),
            "location_tracking_quality": {
                "line_coverage": len(location_stats.get("line_coverage", {})),
                "section_detection": len(location_stats.get("sections_by_type", {})),
                "content_analysis_coverage": sum([
                    location_stats.get("content_analysis", {}).get("documents_with_references", 0),
                    location_stats.get("content_analysis", {}).get("documents_with_definitions", 0),
                    location_stats.get("content_analysis", {}).get("list_items", 0)
                ])
            }
        }

    def delete_documents(self, source_file: str = None):
        """Delete documents from storage and refit vectorizer"""
        try:
            self.vector_store.delete_documents(source_file)
            # Refit vectorizer after deletion
            self._fit_vectorizer()
            return {"message": "Documents deleted successfully"}
        except Exception as e:
            print(f"Error deleting documents: {str(e)}")
            return {"error": str(e)}
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents from the vector store"""
        try:
            # Get all documents from vector store
            documents = self.vector_store.get_all_documents()
            
            if not documents:
                print("No documents found in vector store")
                return []
            
            # Process and enhance documents
            enhanced_docs = []
            for doc in documents:
                enhanced_doc = {
                    "content": doc["content"],
                    "source_file": doc.get("metadata", {}).get("source_file", "unknown"),
                    "type": doc.get("type", "text"),
                    "metadata": doc.get("metadata", {}),
                    "location": doc.get("location", {}),
                    "content_flags": doc.get("content_flags", {}),
                    "created_at": doc.get("created_at", "")
                }
                enhanced_docs.append(enhanced_doc)
            
            print(f"Retrieved {len(enhanced_docs)} documents from vector store")
            return enhanced_docs
            
        except Exception as e:
            print(f"Error getting all documents: {str(e)}")
            return []

    def compare_documents(self, comparison_data: List[Dict[str, Any]], analysis_query: str = "") -> Dict[str, Any]:
        """Compare multiple documents and return analysis results"""
        try:
            comparison_result = {
                "documents_compared": len(comparison_data),
                "total_chunks_analyzed": sum(doc["total_chunks"] for doc in comparison_data),
                "document_sources": [doc["source"] for doc in comparison_data],
                "analysis_timestamp": datetime.now().isoformat(),
                "comparison_analysis": {},
                "cross_document_findings": {},
                "document_summaries": []
            }
            
            # Create document summaries
            for doc_data in comparison_data:
                source = doc_data["source"]
                chunks = doc_data["documents"]
                
                # Analyze document characteristics
                sections_found = set()
                content_types = {"references": 0, "definitions": 0, "list_items": 0}
                line_range = {"min": float('inf'), "max": 0}
                
                for chunk in chunks:
                    location = chunk.get("location", {})
                    content_flags = chunk.get("content_flags", {})
                    
                    if location.get("section_path"):
                        sections_found.add(location["section_path"])
                    
                    line_num = location.get("line_number", 0)
                    if line_num > 0:
                        line_range["min"] = min(line_range["min"], line_num)
                        line_range["max"] = max(line_range["max"], line_num)
                    
                    if content_flags.get("contains_reference"):
                        content_types["references"] += 1
                    if content_flags.get("contains_definition"):
                        content_types["definitions"] += 1
                    if content_flags.get("is_list_item"):
                        content_types["list_items"] += 1
                
                if line_range["min"] == float('inf'):
                    line_range["min"] = 0
                
                comparison_result["document_summaries"].append({
                    "source": source,
                    "total_chunks": len(chunks),
                    "sections_identified": list(sections_found),
                    "content_analysis": content_types,
                    "line_coverage": line_range,
                    "avg_chunk_length": sum(len(chunk["content"]) for chunk in chunks) / len(chunks) if chunks else 0
                })
            
            # Cross-document analysis
            all_sections = set()
            all_content = []
            document_content_by_source = {}
            
            for doc_data in comparison_data:
                source = doc_data["source"]
                chunks = doc_data["documents"]
                document_content_by_source[source] = []
                
                for chunk in chunks:
                    location = chunk.get("location", {})
                    content_flags = chunk.get("content_flags", {})
                    
                    if location.get("section_path"):
                        all_sections.add(location["section_path"])
                    
                    chunk_info = {
                        "content": chunk["content"],
                        "source": source,
                        "location": location,
                        "flags": content_flags
                    }
                    all_content.append(chunk_info)
                    document_content_by_source[source].append(chunk_info)
            
            # Find common sections across documents
            common_sections = set()
            if len(comparison_data) > 1:
                doc_sections = []
                for doc_data in comparison_data:
                    doc_section_set = set()
                    for chunk in doc_data["documents"]:
                        location = chunk.get("location", {})
                        if location.get("section_path"):
                            doc_section_set.add(location["section_path"])
                    doc_sections.append(doc_section_set)
                
                # Find intersection of all document sections
                if doc_sections:
                    common_sections = doc_sections[0]
                    for section_set in doc_sections[1:]:
                        common_sections = common_sections.intersection(section_set)
            
            # Identify potential inconsistencies
            inconsistencies = self._find_cross_document_inconsistencies(document_content_by_source)
            
            comparison_result["cross_document_findings"] = {
                "common_sections": list(common_sections),
                "unique_sections_per_document": self._find_unique_sections(comparison_data),
                "inconsistencies_found": inconsistencies,
                "content_similarity_analysis": self._analyze_content_similarity(document_content_by_source)
            }
            
            # Analysis based on query if provided
            if analysis_query:
                query_results = self._analyze_query_across_documents(analysis_query, all_content)
                comparison_result["query_analysis"] = query_results
            
            return comparison_result
            
        except Exception as e:
            print(f"Error in document comparison: {str(e)}")
            return {
                "error": str(e),
                "documents_compared": 0,
                "total_chunks_analyzed": 0,
                "document_sources": []
            }

    def _find_cross_document_inconsistencies(self, document_content_by_source: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Find inconsistencies across documents"""
        inconsistencies = []
        
        # Check for term variations across documents
        legal_terms = ['agreement', 'contract', 'party', 'parties', 'client', 'section', 'clause', 'exhibit']
        
        for term in legal_terms:
            term_usage = {}
            for source, chunks in document_content_by_source.items():
                variations = set()
                for chunk in chunks:
                    content = chunk["content"].lower()
                    if term in content:
                        # Find variations of the term
                        import re
                        pattern = rf'\b{re.escape(term)}\w*\b'
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        for match in matches:
                            variations.add(match.lower())
                
                if variations:
                    term_usage[source] = list(variations)
            
            if len(term_usage) > 1:
                # Check if there are inconsistencies
                all_variations = set()
                for variations in term_usage.values():
                    all_variations.update(variations)
                
                if len(all_variations) > 1:
                    inconsistencies.append({
                        "type": "term_variation",
                        "term": term,
                        "variations_by_document": term_usage,
                        "recommendation": f"Standardize usage of '{term}' across all documents"
                    })
        
        return inconsistencies

    def _find_unique_sections(self, comparison_data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Find sections that are unique to each document"""
        unique_sections = {}
        
        # Get all sections for each document
        doc_sections = {}
        for doc_data in comparison_data:
            source = doc_data["source"]
            sections = set()
            for chunk in doc_data["documents"]:
                location = chunk.get("location", {})
                if location.get("section_path"):
                    sections.add(location["section_path"])
            doc_sections[source] = sections
        
        # Find unique sections for each document
        for source, sections in doc_sections.items():
            other_sections = set()
            for other_source, other_section_set in doc_sections.items():
                if other_source != source:
                    other_sections.update(other_section_set)
            
            unique = sections - other_sections
            unique_sections[source] = list(unique)
        
        return unique_sections

    def _analyze_content_similarity(self, document_content_by_source: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze content similarity between documents"""
        similarity_analysis = {
            "document_pairs": [],
            "overall_similarity_score": 0.0,
            "common_phrases": []
        }
        
        sources = list(document_content_by_source.keys())
        
        # Compare each pair of documents
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                source1, source2 = sources[i], sources[j]
                
                # Get content from both documents
                content1 = " ".join([chunk["content"] for chunk in document_content_by_source[source1]])
                content2 = " ".join([chunk["content"] for chunk in document_content_by_source[source2]])
                
                # Simple similarity calculation (can be enhanced with more sophisticated methods)
                words1 = set(content1.lower().split())
                words2 = set(content2.lower().split())
                
                common_words = words1.intersection(words2)
                total_words = words1.union(words2)
                
                similarity_score = len(common_words) / len(total_words) if total_words else 0
                
                similarity_analysis["document_pairs"].append({
                    "document1": source1,
                    "document2": source2,
                    "similarity_score": similarity_score,
                    "common_word_count": len(common_words),
                    "unique_to_doc1": len(words1 - words2),
                    "unique_to_doc2": len(words2 - words1)
                })
        
        # Calculate overall similarity
        if similarity_analysis["document_pairs"]:
            avg_similarity = sum(pair["similarity_score"] for pair in similarity_analysis["document_pairs"]) / len(similarity_analysis["document_pairs"])
            similarity_analysis["overall_similarity_score"] = avg_similarity
        
        return similarity_analysis

    def _analyze_query_across_documents(self, query: str, all_content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze a specific query across all documents"""
        query_results = {
            "query": query,
            "matches_by_document": {},
            "total_matches": 0,
            "relevant_sections": []
        }
        
        query_lower = query.lower()
        
        for chunk in all_content:
            if query_lower in chunk["content"].lower():
                source = chunk["source"]
                if source not in query_results["matches_by_document"]:
                    query_results["matches_by_document"][source] = []
                
                match_info = {
                    "content": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                    "location": chunk["location"],
                    "section_path": chunk["location"].get("section_path", ""),
                    "line_number": chunk["location"].get("line_number", 0)
                }
                
                query_results["matches_by_document"][source].append(match_info)
                query_results["total_matches"] += 1
                
                if chunk["location"].get("section_path"):
                    query_results["relevant_sections"].append(chunk["location"]["section_path"])
        
        # Remove duplicate sections
        query_results["relevant_sections"] = list(set(query_results["relevant_sections"]))
        
        return query_results 

    def get_documents_by_source(self, source_file: str) -> List[Dict[str, Any]]:
        """Get all documents from a specific source file that have been processed by RAG"""
        try:
            # Search by source file in vector store
            documents = self.vector_store.search_by_location({"source_file": source_file})
            
            if not documents:
                print(f"No documents found for source: {source_file}")
                return []
            
            # Process and enhance documents with RAG metadata
            enhanced_docs = []
            for doc in documents:
                enhanced_doc = {
                    "content": doc["content"],
                    "type": doc["type"],
                    "metadata": doc.get("metadata", {}),
                    "location": doc.get("location", {}),
                    "content_flags": doc.get("content_flags", {}),
                    "rag_metadata": {
                        "chunk_index": doc.get("metadata", {}).get("chunk_index", 0),
                        "source_file": source_file,
                        "created_at": doc.get("created_at", ""),
                        "document_type": doc.get("document_type", "unknown")
                    }
                }
                enhanced_docs.append(enhanced_doc)
            
            print(f"Retrieved {len(enhanced_docs)} documents from RAG for source: {source_file}")
            return enhanced_docs
            
        except Exception as e:
            print(f"Error getting documents by source: {str(e)}")
            return []