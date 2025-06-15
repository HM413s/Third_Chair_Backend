from typing import List, Dict, Any
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np
import json
import os
from datetime import datetime
from config import (
    ZILLIZ_HOST, ZILLIZ_PORT, ZILLIZ_USER, ZILLIZ_PASSWORD, ZILLIZ_URI,
    ZILLIZ_CLOUD_API_KEY, ZILLIZ_CLUSTER_NAME, ZILLIZ_CLOUD_ENDPOINT,
    STORAGE_DIR, SEARCH_LIMIT, SEARCH_EF, MAX_FILE_SIZE, ENABLE_LOCAL_BACKUP
)

class ZillizVectorStore:
    _instance = None
    _connection_pool = {}
    
    def __new__(cls, collection_name: str = None):
        if cls._instance is None:
            cls._instance = super(ZillizVectorStore, cls).__new__(cls)
            cls._instance._initialize(collection_name)
        return cls._instance
    
    def _initialize(self, collection_name: str = None):
        """Initialize the vector store connection"""
        self.collection_name = collection_name or os.getenv("ZILLIZ_COLLECTION_NAME", "TRADE_IDE")
        if self.collection_name not in self._connection_pool:
            print(f"📁 Initializing vector store with collection: {self.collection_name}")
            print(f"☁️ Cloud-only storage mode: Local backup {'DISABLED' if not ENABLE_LOCAL_BACKUP else 'ENABLED'}")
            self.connect()
            self._create_collection_if_not_exists()
            if ENABLE_LOCAL_BACKUP:
                self._ensure_storage_dir()
            self._connection_pool[self.collection_name] = self.collection
        else:
            self.collection = self._connection_pool[self.collection_name]
    
    def __init__(self, collection_name: str = None):
        if not hasattr(self, 'collection'):
            self._initialize(collection_name)
    
    def _ensure_storage_dir(self):
        """Ensure storage directory exists for metadata backup"""
        if STORAGE_DIR:
            os.makedirs(STORAGE_DIR, exist_ok=True)
            print(f"📁 Local storage directory created: {STORAGE_DIR}")
        else:
            print("☁️ Local storage disabled - using cloud-only mode")
    
    def connect(self):
        """Connect to Zilliz Cloud with API key authentication"""
        try:
            # Use Zilliz Cloud configuration with API key
            print(f"🔗 Connecting to Zilliz Cloud cluster: {ZILLIZ_CLUSTER_NAME}")
            print(f"🌐 Endpoint: {ZILLIZ_CLOUD_ENDPOINT}")
            print(f"🔑 Using API key authentication")
            
            # Disconnect any existing connections
            try:
                connections.disconnect("default")
            except:
                pass  # Ignore if no existing connection
            
            # Connect using API key authentication for Zilliz Cloud
            connections.connect(
                alias="default",
                uri=ZILLIZ_CLOUD_ENDPOINT,
                token=ZILLIZ_CLOUD_API_KEY,
                secure=True,
                timeout=60,  # Increase timeout for cloud connection
                db_name="default"
            )
            
            # Verify connection
            if not connections.has_connection("default"):
                raise Exception("Failed to establish connection to Zilliz Cloud")
                
            print(f"✅ Successfully connected to Zilliz Cloud cluster: {ZILLIZ_CLUSTER_NAME}")
            return True
            
        except Exception as e:
            print(f"❌ Error connecting to Zilliz Cloud: {str(e)}")
            print("\n🔧 Troubleshooting steps:")
            print("1. Check your internet connection")
            print("2. Verify your Zilliz Cloud API key is correct")
            print("3. Ensure your cluster name 'TRADE_IDE' exists and is running")
            print("4. Check if your IP is whitelisted in Zilliz Cloud")
            print("5. Confirm your Zilliz Cloud subscription is active")
            print("6. Try accessing the Zilliz Cloud console in your browser")
            print(f"7. Verify cluster endpoint: {ZILLIZ_CLOUD_ENDPOINT}")
            
            # Cloud-only mode - no fallback to local storage
            print("☁️ Cloud-only mode enabled - no fallback to local Milvus")
            raise ConnectionError(f"Could not connect to Zilliz Cloud cluster '{ZILLIZ_CLUSTER_NAME}': {str(e)}")
    
    def _create_collection_if_not_exists(self):
        """Create collection if it doesn't exist with enhanced schema for location tracking"""
        if not utility.has_collection(self.collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=10),
                
                # Split image_base64 into multiple fields to handle large images
                FieldSchema(name="image_base64_1", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="image_base64_2", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="image_base64_3", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="image_base64_4", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="image_base64_5", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="image_base64_parts", dtype=DataType.INT64),  # Number of parts used
                
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=4000),  # TF-IDF dimension
                FieldSchema(name="metadata", dtype=DataType.JSON),  # Store additional metadata
                FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=255),
                
                # Enhanced location tracking fields
                FieldSchema(name="line_number", dtype=DataType.INT64),
                FieldSchema(name="page_number", dtype=DataType.INT64),
                FieldSchema(name="start_char", dtype=DataType.INT64),
                FieldSchema(name="end_char", dtype=DataType.INT64),
                FieldSchema(name="section_type", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="section_number", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="section_title", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="document_type", dtype=DataType.VARCHAR, max_length=20),
                
                # Content analysis fields for better agent detection
                FieldSchema(name="contains_reference", dtype=DataType.BOOL),
                FieldSchema(name="contains_definition", dtype=DataType.BOOL),
                FieldSchema(name="is_list_item", dtype=DataType.BOOL),
                FieldSchema(name="word_count", dtype=DataType.INT64),
                
                # Location context for agents
                FieldSchema(name="absolute_position", dtype=DataType.VARCHAR, max_length=200),
                FieldSchema(name="section_path", dtype=DataType.VARCHAR, max_length=300),
            ]
            schema = CollectionSchema(fields=fields, description="RAG document collection with enhanced location tracking")
            self.collection = Collection(name=self.collection_name, schema=schema)
            
            # Create index for vector field with optimized parameters
            index_params = {
                "metric_type": "COSINE",  # Better for semantic similarity
                "index_type": "HNSW",     # Faster and more accurate than IVF_FLAT
                "params": {
                    "M": 16,             # Number of bi-directional links
                    "efConstruction": 200 # Higher value = better accuracy
                }
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            
            # Create indexes for location-based searches
            print("Creating location-based indexes for enhanced agent detection...")
            # Note: Zilliz/Milvus automatically indexes scalar fields, but we can create explicit indexes if needed
            
        else:
            self.collection = Collection(self.collection_name)
    
    def _split_image_base64(self, image_base64: str) -> tuple:
        """Split image base64 string into multiple parts"""
        if not image_base64:
            return [""] * 5, 0
        
        # Split into chunks of 65000 characters (leaving some margin)
        chunk_size = 65000
        parts = [image_base64[i:i + chunk_size] for i in range(0, len(image_base64), chunk_size)]
        
        # Pad with empty strings if less than 5 parts
        while len(parts) < 5:
            parts.append("")
            
        return parts[:5], len(parts)

    def _combine_image_base64(self, parts: list, num_parts: int) -> str:
        """Combine image base64 parts back into a single string"""
        if num_parts == 0:
            return ""
        return "".join(parts[:num_parts])

    def add_documents(self, documents: List[Dict[str, Any]], source_file: str = None):
        """Add documents to the vector store with enhanced location metadata"""
        entities = []
        current_time = datetime.now().isoformat()
        
        for doc in documents:
            image_parts, num_parts = self._split_image_base64(doc.get("image_base64", ""))
            
            # Extract location metadata from document
            doc_metadata = doc.get("metadata", {})
            location_context = doc_metadata.get("location_context", {})
            containing_section = doc_metadata.get("containing_section", {})
            
            # Merge chunk metadata with additional info for comprehensive location tracking
            enhanced_metadata = {
                **doc_metadata,
                "chunk_index": len(entities),
                "source_type": doc["type"],
                "created_at": current_time,
                "source_file": source_file or doc_metadata.get("source_file", "unknown"),
                "content_length": len(doc["content"]),
                "has_image": bool(doc.get("image_base64")),
                
                # Enhanced location tracking
                "enhanced_location": {
                    "absolute_position": location_context.get("absolute_position", ""),
                    "section_path": location_context.get("section_path", ""),
                    "section_title": location_context.get("section_title", ""),
                    "file_path": location_context.get("file_path", ""),
                    "document_type": location_context.get("document_type", "unknown"),
                    "surrounding_context": location_context.get("surrounding_context", "")[:500],  # Limit context size
                },
                
                # Document structure for agent navigation
                "document_structure": doc_metadata.get("document_structure_summary", {}),
                "nearby_headers": doc_metadata.get("nearby_headers", []),
                
                # Content flags for agent filtering
                "content_flags": {
                    "contains_reference": doc_metadata.get("contains_reference", False),
                    "contains_definition": doc_metadata.get("contains_definition", False),
                    "is_list_item": doc_metadata.get("is_list_item", False),
                    "word_count": doc_metadata.get("word_count", 0),
                    "char_count": doc_metadata.get("char_count", 0)
                }
            }
            
            entity = {
                "content": doc["content"],
                "type": doc["type"],
                "image_base64_1": image_parts[0],
                "image_base64_2": image_parts[1],
                "image_base64_3": image_parts[2],
                "image_base64_4": image_parts[3],
                "image_base64_5": image_parts[4],
                "image_base64_parts": num_parts,
                "embedding": doc["embedding"],
                "metadata": enhanced_metadata,
                "created_at": current_time,
                "source_file": source_file or doc_metadata.get("source_file", "unknown"),
                
                # Enhanced location fields for direct agent access
                "line_number": int(doc_metadata.get("line_number", 0) or 0),
                "page_number": int(doc_metadata.get("page_number", 0) or 0),
                "start_char": int(doc_metadata.get("start_char", 0) or 0),
                "end_char": int(doc_metadata.get("end_char", 0) or 0),
                "section_type": containing_section.get("type", "unknown") or "unknown",
                "section_number": containing_section.get("number", "") or "",
                "section_title": containing_section.get("title", "") or "",
                "document_type": location_context.get("document_type", "unknown") or "unknown",
                
                # Content analysis for agent filtering - ensure booleans
                "contains_reference": bool(doc_metadata.get("contains_reference", False)),
                "contains_definition": bool(doc_metadata.get("contains_definition", False)),
                "is_list_item": bool(doc_metadata.get("is_list_item", False)),
                "word_count": int(doc_metadata.get("word_count", 0) or 0),
                
                # Location strings for agent queries
                "absolute_position": location_context.get("absolute_position", "") or "",
                "section_path": location_context.get("section_path", "") or "",
            }
            entities.append(entity)
        
        self.collection.insert(entities)
        self.collection.flush()
        
        # Only create local backup if enabled
        if ENABLE_LOCAL_BACKUP:
            self._backup_metadata_with_location(entities)
            print(f"✅ Successfully stored {len(entities)} documents in Zilliz Cloud with local backup")
        else:
            print(f"☁️ Successfully stored {len(entities)} documents in Zilliz Cloud (cloud-only mode)")
        
        return len(entities)
    
    def _backup_metadata_with_location(self, entities: List[Dict[str, Any]]):
        """Backup document metadata with location information to disk"""
        if not STORAGE_DIR or not ENABLE_LOCAL_BACKUP:
            print("☁️ Local backup disabled - metadata stored only in Zilliz Cloud")
            return
            
        backup_file = os.path.join(STORAGE_DIR, f"metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Create a backup with location summary for easier debugging
        backup_data = {
            "timestamp": datetime.now().isoformat(),
            "total_documents": len(entities),
            "location_summary": {},
            "documents": entities
        }
        
        # Create location summary for quick reference
        for entity in entities:
            source_file = entity.get("source_file", "unknown")
            if source_file not in backup_data["location_summary"]:
                backup_data["location_summary"][source_file] = {
                    "document_count": 0,
                    "sections_found": set(),
                    "line_range": {"min": float('inf'), "max": 0},
                    "document_type": entity.get("document_type", "unknown")
                }
            
            summary = backup_data["location_summary"][source_file]
            summary["document_count"] += 1
            summary["sections_found"].add(f"{entity.get('section_type', '')} {entity.get('section_number', '')}".strip())
            summary["line_range"]["min"] = min(summary["line_range"]["min"], entity.get("line_number", 0))
            summary["line_range"]["max"] = max(summary["line_range"]["max"], entity.get("line_number", 0))
        
        # Convert sets to lists for JSON serialization
        for file_summary in backup_data["location_summary"].values():
            file_summary["sections_found"] = list(file_summary["sections_found"])
            if file_summary["line_range"]["min"] == float('inf'):
                file_summary["line_range"]["min"] = 0
        
        with open(backup_file, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        print(f"Metadata backup with location tracking saved to: {backup_file}")
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about stored documents with location information"""
        self.collection.load()
        stats = {
            "total_documents": self.collection.num_entities,
            "document_types": {},
            "sources": {},
            "creation_dates": {},
            "location_stats": {
                "sections_by_type": {},
                "documents_by_document_type": {},
                "content_analysis": {
                    "documents_with_references": 0,
                    "documents_with_definitions": 0,
                    "list_items": 0
                },
                "line_coverage": {},
                "page_coverage": {}
            }
        }
        
        # Query all documents to get statistics with location data
        results = self.collection.query(
            expr="id > 0",
            output_fields=[
                "type", "metadata", "created_at", "source_file", "line_number", "page_number",
                "section_type", "section_number", "document_type", "contains_reference",
                "contains_definition", "is_list_item", "word_count"
            ]
        )
        
        for doc in results:
            # Count document types
            doc_type = doc["type"]
            stats["document_types"][doc_type] = stats["document_types"].get(doc_type, 0) + 1
            
            # Count sources
            source = doc["source_file"]
            stats["sources"][source] = stats["sources"].get(source, 0) + 1
            
            # Count by creation date
            date = doc["created_at"].split("T")[0]
            stats["creation_dates"][date] = stats["creation_dates"].get(date, 0) + 1
            
            # Location statistics
            section_type = doc.get("section_type", "unknown")
            stats["location_stats"]["sections_by_type"][section_type] = stats["location_stats"]["sections_by_type"].get(section_type, 0) + 1
            
            document_type = doc.get("document_type", "unknown")
            stats["location_stats"]["documents_by_document_type"][document_type] = stats["location_stats"]["documents_by_document_type"].get(document_type, 0) + 1
            
            # Content analysis
            if doc.get("contains_reference"):
                stats["location_stats"]["content_analysis"]["documents_with_references"] += 1
            if doc.get("contains_definition"):
                stats["location_stats"]["content_analysis"]["documents_with_definitions"] += 1
            if doc.get("is_list_item"):
                stats["location_stats"]["content_analysis"]["list_items"] += 1
            
            # Line and page coverage
            source_file = doc["source_file"]
            if source_file not in stats["location_stats"]["line_coverage"]:
                stats["location_stats"]["line_coverage"][source_file] = {"min": float('inf'), "max": 0}
            if source_file not in stats["location_stats"]["page_coverage"]:
                stats["location_stats"]["page_coverage"][source_file] = {"min": float('inf'), "max": 0}
            
            line_num = doc.get("line_number", 0)
            page_num = doc.get("page_number", 0)
            
            if line_num > 0:
                stats["location_stats"]["line_coverage"][source_file]["min"] = min(stats["location_stats"]["line_coverage"][source_file]["min"], line_num)
                stats["location_stats"]["line_coverage"][source_file]["max"] = max(stats["location_stats"]["line_coverage"][source_file]["max"], line_num)
            
            if page_num > 0:
                stats["location_stats"]["page_coverage"][source_file]["min"] = min(stats["location_stats"]["page_coverage"][source_file]["min"], page_num)
                stats["location_stats"]["page_coverage"][source_file]["max"] = max(stats["location_stats"]["page_coverage"][source_file]["max"], page_num)
        
        # Clean up infinite values
        for coverage in [stats["location_stats"]["line_coverage"], stats["location_stats"]["page_coverage"]]:
            for file_coverage in coverage.values():
                if file_coverage["min"] == float('inf'):
                    file_coverage["min"] = 0
        
        return stats
    
    def search(self, query_embedding: List[float], limit: int = SEARCH_LIMIT) -> List[Dict[str, Any]]:
        """Search for similar documents using Zilliz's vector search with location data"""
        self.collection.load()
        
        # Optimized search parameters
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": SEARCH_EF}  # Higher value = better accuracy
        }
        
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=[
                "content", "type", "image_base64_1", "image_base64_2", 
                "image_base64_3", "image_base64_4", "image_base64_5", 
                "image_base64_parts", "metadata", "created_at", "source_file",
                "line_number", "page_number", "start_char", "end_char",
                "section_type", "section_number", "section_title", "document_type",
                "absolute_position", "section_path", "contains_reference",
                "contains_definition", "is_list_item", "word_count"
            ]
        )
        
        documents = []
        for hits in results:
            for hit in hits:
                # Combine image parts
                image_parts = [
                    hit.entity.get("image_base64_1", ""),
                    hit.entity.get("image_base64_2", ""),
                    hit.entity.get("image_base64_3", ""),
                    hit.entity.get("image_base64_4", ""),
                    hit.entity.get("image_base64_5", "")
                ]
                num_parts = hit.entity.get("image_base64_parts", 0)
                image_base64 = self._combine_image_base64(image_parts, num_parts)
                
                # Enhanced document with location information
                doc = {
                    "content": hit.entity.get("content"),
                    "type": hit.entity.get("type"),
                    "image_base64": image_base64,
                    "score": hit.distance,
                    "metadata": hit.entity.get("metadata", {}),
                    "created_at": hit.entity.get("created_at"),
                    "source_file": hit.entity.get("source_file"),
                    
                    # Enhanced location information for agents
                    "location": {
                        "line_number": hit.entity.get("line_number", 0),
                        "page_number": hit.entity.get("page_number", 0),
                        "start_char": hit.entity.get("start_char", 0),
                        "end_char": hit.entity.get("end_char", 0),
                        "absolute_position": hit.entity.get("absolute_position", ""),
                        "section_path": hit.entity.get("section_path", ""),
                        "section_type": hit.entity.get("section_type", ""),
                        "section_number": hit.entity.get("section_number", ""),
                        "section_title": hit.entity.get("section_title", ""),
                        "document_type": hit.entity.get("document_type", "unknown")
                    },
                    
                    # Content flags for agent filtering
                    "content_flags": {
                        "contains_reference": hit.entity.get("contains_reference", False),
                        "contains_definition": hit.entity.get("contains_definition", False),
                        "is_list_item": hit.entity.get("is_list_item", False),
                        "word_count": hit.entity.get("word_count", 0)
                    }
                }
                documents.append(doc)
        
        return documents

    def search_by_location(self, location_criteria: Dict[str, Any], limit: int = SEARCH_LIMIT) -> List[Dict[str, Any]]:
        """Search documents by specific location criteria for agent targeting"""
        self.collection.load()
        
        # Build expression for location-based search
        expressions = []
        
        if "source_file" in location_criteria:
            expressions.append(f'source_file == "{location_criteria["source_file"]}"')
        
        if "section_type" in location_criteria:
            expressions.append(f'section_type == "{location_criteria["section_type"]}"')
        
        if "section_number" in location_criteria:
            expressions.append(f'section_number == "{location_criteria["section_number"]}"')
        
        if "document_type" in location_criteria:
            expressions.append(f'document_type == "{location_criteria["document_type"]}"')
        
        if "line_range" in location_criteria:
            line_min, line_max = location_criteria["line_range"]
            expressions.append(f'line_number >= {line_min} and line_number <= {line_max}')
        
        if "page_number" in location_criteria:
            expressions.append(f'page_number == {location_criteria["page_number"]}')
        
        if "contains_reference" in location_criteria:
            expressions.append(f'contains_reference == {str(location_criteria["contains_reference"]).lower()}')
        
        if "contains_definition" in location_criteria:
            expressions.append(f'contains_definition == {str(location_criteria["contains_definition"]).lower()}')
        
        # Combine expressions
        if not expressions:
            expr = "id > 0"  # Get all documents
        else:
            expr = " and ".join(expressions)
        
        print(f"Location search expression: {expr}")
        
        results = self.collection.query(
            expr=expr,
            output_fields=[
                "content", "type", "image_base64_1", "image_base64_2", 
                "image_base64_3", "image_base64_4", "image_base64_5", 
                "image_base64_parts", "metadata", "created_at", "source_file",
                "line_number", "page_number", "start_char", "end_char",
                "section_type", "section_number", "section_title", "document_type",
                "absolute_position", "section_path", "contains_reference",
                "contains_definition", "is_list_item", "word_count"
            ],
            limit=limit
        )
        
        documents = []
        for doc in results:
            # Combine image parts if present
            if doc["type"] == "image":
                image_parts = [
                    doc.get("image_base64_1", ""),
                    doc.get("image_base64_2", ""),
                    doc.get("image_base64_3", ""),
                    doc.get("image_base64_4", ""),
                    doc.get("image_base64_5", "")
                ]
                num_parts = doc.get("image_base64_parts", 0)
                image_base64 = self._combine_image_base64(image_parts, num_parts)
                doc["image_base64"] = image_base64
            
            # Add enhanced location information
            doc["location"] = {
                "line_number": doc.get("line_number", 0),
                "page_number": doc.get("page_number", 0),
                "start_char": doc.get("start_char", 0),
                "end_char": doc.get("end_char", 0),
                "absolute_position": doc.get("absolute_position", ""),
                "section_path": doc.get("section_path", ""),
                "section_type": doc.get("section_type", ""),
                "section_number": doc.get("section_number", ""),
                "section_title": doc.get("section_title", ""),
                "document_type": doc.get("document_type", "unknown")
            }
            
            doc["content_flags"] = {
                "contains_reference": doc.get("contains_reference", False),
                "contains_definition": doc.get("contains_definition", False),
                "is_list_item": doc.get("is_list_item", False),
                "word_count": doc.get("word_count", 0)
            }
            
            documents.append(doc)
        
        print(f"Location search returned {len(documents)} documents")
        return documents
    
    def hybrid_search(self, query_embedding: List[float], text_query: str, limit: int = SEARCH_LIMIT) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector similarity and text matching with location data"""
        try:
            self.collection.load()
            print(f"Collection loaded for hybrid search")
            
            # Vector search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": SEARCH_EF}
            }
            
            # Create a more flexible text search expression
            # Split query into words and create OR conditions
            query_words = text_query.lower().split()
            text_conditions = [f'content like "%{word}%"' for word in query_words]
            text_expr = " or ".join(text_conditions)
            
            print(f"Using text search expression: {text_expr}")
            
            # Combine vector search with text filtering
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit * 2,  # Get more results for better filtering
                output_fields=[
                    "content", "type", "image_base64_1", "image_base64_2", 
                    "image_base64_3", "image_base64_4", "image_base64_5", 
                    "image_base64_parts", "metadata", "created_at", "source_file",
                    "line_number", "page_number", "start_char", "end_char",
                    "section_type", "section_number", "section_title", "document_type",
                    "absolute_position", "section_path", "contains_reference",
                    "contains_definition", "is_list_item", "word_count"
                ],
                expr=text_expr
            )
            
            print(f"Found {len(results[0])} results in hybrid search")
            
            documents = []
            for hits in results:
                for hit in hits:
                    # Combine image parts
                    image_parts = [
                        hit.entity.get("image_base64_1", ""),
                        hit.entity.get("image_base64_2", ""),
                        hit.entity.get("image_base64_3", ""),
                        hit.entity.get("image_base64_4", ""),
                        hit.entity.get("image_base64_5", "")
                    ]
                    num_parts = hit.entity.get("image_base64_parts", 0)
                    image_base64 = self._combine_image_base64(image_parts, num_parts)
                    
                    # Enhanced document with comprehensive location data
                    doc = {
                        "content": hit.entity.get("content"),
                        "type": hit.entity.get("type"),
                        "image_base64": image_base64,
                        "score": hit.distance,
                        "metadata": hit.entity.get("metadata", {}),
                        "created_at": hit.entity.get("created_at"),
                        "source_file": hit.entity.get("source_file"),
                        
                        # Comprehensive location information for agents
                        "location": {
                            "line_number": hit.entity.get("line_number", 0),
                            "page_number": hit.entity.get("page_number", 0),
                            "start_char": hit.entity.get("start_char", 0),
                            "end_char": hit.entity.get("end_char", 0),
                            "absolute_position": hit.entity.get("absolute_position", ""),
                            "section_path": hit.entity.get("section_path", ""),
                            "section_type": hit.entity.get("section_type", ""),
                            "section_number": hit.entity.get("section_number", ""),
                            "section_title": hit.entity.get("section_title", ""),
                            "document_type": hit.entity.get("document_type", "unknown")
                        },
                        
                        # Content analysis flags for agent decision making
                        "content_flags": {
                            "contains_reference": hit.entity.get("contains_reference", False),
                            "contains_definition": hit.entity.get("contains_definition", False),
                            "is_list_item": hit.entity.get("is_list_item", False),
                            "word_count": hit.entity.get("word_count", 0)
                        }
                    }
                    documents.append(doc)
            
            # Sort by score and take top results
            documents.sort(key=lambda x: x["score"], reverse=True)
            return documents[:limit]
            
        except Exception as e:
            print(f"Error in hybrid search: {str(e)}")
            return []
    
    def delete_documents(self, source_file: str = None):
        """Delete documents from the vector store"""
        if source_file:
            # Delete documents from a specific source
            expr = f'source_file == "{source_file}"'
        else:
            # Delete all documents
            expr = "id > 0"
        
        self.collection.delete(expr)
        self.collection.flush()
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents from the collection with comprehensive location data"""
        try:
            self.collection.load()
            results = self.collection.query(
                expr="id > 0",
                output_fields=[
                    "content", "type", "image_base64_1", "image_base64_2", 
                    "image_base64_3", "image_base64_4", "image_base64_5", 
                    "image_base64_parts", "metadata", "created_at", "source_file",
                    "line_number", "page_number", "start_char", "end_char",
                    "section_type", "section_number", "section_title", "document_type",
                    "absolute_position", "section_path", "contains_reference",
                    "contains_definition", "is_list_item", "word_count"
                ]
            )
            
            documents = []
            for doc in results:
                # Combine image parts if present
                if doc["type"] == "image":
                    image_parts = [
                        doc.get("image_base64_1", ""),
                        doc.get("image_base64_2", ""),
                        doc.get("image_base64_3", ""),
                        doc.get("image_base64_4", ""),
                        doc.get("image_base64_5", "")
                    ]
                    num_parts = doc.get("image_base64_parts", 0)
                    image_base64 = self._combine_image_base64(image_parts, num_parts)
                    doc["image_base64"] = image_base64
                
                # Add comprehensive location information for agents
                doc["location"] = {
                    "line_number": doc.get("line_number", 0),
                    "page_number": doc.get("page_number", 0),
                    "start_char": doc.get("start_char", 0),
                    "end_char": doc.get("end_char", 0),
                    "absolute_position": doc.get("absolute_position", ""),
                    "section_path": doc.get("section_path", ""),
                    "section_type": doc.get("section_type", ""),
                    "section_number": doc.get("section_number", ""),
                    "section_title": doc.get("section_title", ""),
                    "document_type": doc.get("document_type", "unknown")
                }
                
                doc["content_flags"] = {
                    "contains_reference": doc.get("contains_reference", False),
                    "contains_definition": doc.get("contains_definition", False),
                    "is_list_item": doc.get("is_list_item", False),
                    "word_count": doc.get("word_count", 0)
                }
                
                documents.append(doc)
            
            print(f"Retrieved {len(documents)} documents with enhanced location tracking")
            return documents
        except Exception as e:
            print(f"Error getting all documents: {str(e)}")
            return []