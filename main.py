from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from RAG.rag_service import RAGService
from router import users_router,authentication
from typing import Dict, Any, List
from pydantic import BaseModel
from schemas import ChatResponse, StatsResponse, UploadResponse, DeleteResponse, UserCreate, Login, Token, UserResponse
from oauth2 import get_current_user
from models import User
from database import engine,Base
import uuid
from datetime import datetime
from Agents.Coherence_agent.Project_coherence_agent import clause_consistency_agent_auto, enhanced_automated_consistency_analysis
from Agents.Coherence_agent.context import ContractAnalysisContext
from Agents.Firm_Standard_Agent.Firm_Standard_Agent import enhanced_automated_style_analysis
from Agents.Firm_Standard_Agent.context import FirmStandardContext
from websocket_manager import active_connections, send_websocket_message
from Agents.Firm_Standard_Agent.report_processor import process_style_report
from Agents.Coherence_agent.report_processor import process_coherence_report
import json
import sys

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Set up a handler with UTF-8 encoding
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

app = FastAPI(title="Third Chair Backend API",
             description="Backend API for Third Chair application",
             version="1.0.0")

# Create database tables
Base.metadata.create_all(bind=engine)

# Get allowed origins from environment variable or use default
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://third-chair.vercel.app"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)


# Create uploads directory if it doesn't exist
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Define response models
class MultiUploadResponse(BaseModel):
    message: str
    session_id: str
    documents_processed: List[Dict[str, Any]]
    total_documents: int

class ComparisonRequest(BaseModel):
    query: str
    session_id: str = None
    document_sources: List[str] = None

class ComparisonResponse(BaseModel):
    comparison_analysis: Dict[str, Any]
    documents_analyzed: List[str]
    session_id: str = None

# Initialize RAG service
app.include_router(authentication.router)
app.include_router(users_router.router)
rag_service = RAGService()

# In-memory session storage (in production, use Redis or database)
document_sessions = {}

@app.websocket("/ws/agent-logs")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # Keep the connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)

@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    current_user: UserResponse = Depends(get_current_user)
):
    """DEPRECATED: Single document upload - Use /upload-multiple instead for better multi-document analysis"""
    try:
        print("⚠️ WARNING: Single document upload is deprecated. Use /upload-multiple for better analysis.")
        
        # Save the file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the document
        num_docs = rag_service.process_and_store_document(file_path)
        
        # Clean up the uploaded file
        os.remove(file_path)
        
        # Create a session for this single document to enable multi-document features
        session_id = f"single_doc_{str(uuid.uuid4())[:8]}"
        document_sessions[session_id] = {
            "created_at": datetime.now().isoformat(),
            "user_id": current_user.id,
            "documents": [file.filename],
            "total_chunks": num_docs,
            "session_name": "Single Document Session"
        }
        
        return {
            "message": f"Document processed successfully. Session created: {session_id}. Recommend uploading multiple documents for comparison analysis.",
            "documents_processed": num_docs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-multiple", response_model=MultiUploadResponse)
async def upload_multiple_files(
    files: List[UploadFile] = File(...),
    session_name: str = None,
    current_user: UserResponse = Depends(get_current_user)
):
    """PRIMARY ENDPOINT: Upload and process multiple documents for comprehensive cross-document analysis"""
    try:
        if len(files) < 2:
            print("⚠️ WARNING: Only 1 file uploaded. Multi-document analysis requires 2+ documents for comparison.")
        
        # Create a unique session ID for this document set
        session_id = str(uuid.uuid4())
        if session_name:
            session_id = f"{session_name}_{session_id[:8]}"
        
        processed_docs = []
        total_chunks = 0
        
        print(f"📁 Processing {len(files)} documents for multi-document analysis...")
        
        for file in files:
            # Save the file
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Process the document with session tracking
            num_docs = rag_service.process_and_store_document(file_path)
            total_chunks += num_docs
            
            processed_docs.append({
                "filename": file.filename,
                "documents_processed": num_docs,
                "file_size": len(content),
                "file_type": os.path.splitext(file.filename)[1].lower()
            })
            
            # Clean up the uploaded file
            os.remove(file_path)
            
            print(f"  ✅ {file.filename}: {num_docs} chunks processed")
        
        # Store session information
        document_sessions[session_id] = {
            "created_at": datetime.now().isoformat(),
            "user_id": current_user.id,
            "documents": [doc["filename"] for doc in processed_docs],
            "total_chunks": total_chunks,
            "session_name": session_name
        }
        
        print(f"📊 Multi-document session created: {session_id}")
        print(f"📈 Ready for cross-document comparison analysis!")
        
        return {
            "message": f"Multi-document processing completed successfully! {len(files)} documents ready for comparison analysis.",
            "session_id": session_id,
            "documents_processed": processed_docs,
            "total_documents": len(files)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-multi-documents")
async def analyze_multi_documents(
    session_id: str = None,
    document_sources: List[str] = None,
    analysis_type: str = "comprehensive",
    current_user: UserResponse = Depends(get_current_user)
):
    """Advanced multi-document analysis using specialized agents"""
    try:
        # Import the multi-document agent
        from Agents.Firm_Standard_Agent.Firm_Standard_Agent import multi_document_comparison_agent
        
        # Determine which documents to analyze
        if session_id and session_id in document_sessions:
            session = document_sessions[session_id]
            if session["user_id"] != current_user.id:
                raise HTTPException(status_code=403, detail="Access denied")
            document_sources = session["documents"]
        elif document_sources:
            # Use specified document sources
            pass
        else:
            # Get all available documents
            stats = rag_service.get_storage_stats()
            document_sources = list(stats.get("sources", {}).keys())
        
        if len(document_sources) < 2:
            raise HTTPException(
                status_code=400, 
                detail=f"Multi-document analysis requires at least 2 documents. Found: {len(document_sources)}"
            )
        
        print(f"🤖 Starting multi-document analysis for {len(document_sources)} documents...")
        
        # Run the multi-document comparison agent
        result = multi_document_comparison_agent(document_sources)
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Analysis failed"))
        
        return {
            "success": True,
            "analysis_type": "multi_document_comparison",
            "session_id": session_id,
            "documents_analyzed": document_sources,
            "analysis_results": result,
            "summary": result.get("summary", {}),
            "recommendations": result.get("final_report", {}).get("recommendations", [])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare-documents", response_model=ComparisonResponse)
async def compare_documents(
    request: ComparisonRequest,
    current_user: UserResponse = Depends(get_current_user)
):
    """Compare multiple documents using agent analysis"""
    try:
        # Determine which documents to compare
        if request.session_id and request.session_id in document_sessions:
            # Use documents from session
            session = document_sessions[request.session_id]
            document_sources = session["documents"]
        elif request.document_sources:
            # Use specified document sources
            document_sources = request.document_sources
        else:
            # Get all available documents
            stats = rag_service.get_storage_stats()
            document_sources = list(stats.get("sources", {}).keys())
        
        if len(document_sources) < 2:
            raise HTTPException(
                status_code=400, 
                detail="At least 2 documents are required for comparison"
            )
        
        # Get documents for comparison
        comparison_data = []
        for source in document_sources:
            docs = rag_service.search_by_location({"source_file": source}, limit=50)
            comparison_data.append({
                "source": source,
                "documents": docs["text_docs"],
                "total_chunks": len(docs["text_docs"])
            })
        
        # Perform comparison analysis
        comparison_result = rag_service.compare_documents(comparison_data, request.query)
        
        return {
            "comparison_analysis": comparison_result,
            "documents_analyzed": document_sources,
            "session_id": request.session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions")
async def get_document_sessions(
    current_user: UserResponse = Depends(get_current_user)
):
    """Get all document sessions for the current user"""
    user_sessions = {
        session_id: session_data 
        for session_id, session_data in document_sessions.items()
        if session_data["user_id"] == current_user.id
    }
    return {
        "sessions": user_sessions,
        "total_sessions": len(user_sessions)
    }

@app.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: UserResponse = Depends(get_current_user)
):
    """Delete a document session and optionally its documents"""
    if session_id not in document_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = document_sessions[session_id]
    if session["user_id"] != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Delete documents from the session
    for doc_name in session["documents"]:
        rag_service.delete_documents(doc_name)
    
    # Delete session
    del document_sessions[session_id]
    
    return {"message": f"Session {session_id} deleted successfully"}

@app.post("/chat", response_model=ChatResponse)
async def chat(
    query: str,
    current_user: UserResponse = Depends(get_current_user)
):
    """Process a chat query and return response with relevant images"""
    try:
        response = rag_service.chat(query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=StatsResponse)
async def get_stats(
    current_user: UserResponse = Depends(get_current_user)
):
    """Get statistics about stored documents"""
    try:
        return rag_service.get_storage_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents", response_model=DeleteResponse)
async def delete_documents(
    source_file: str = None,
    current_user: UserResponse = Depends(get_current_user)
):
    """Delete documents from storage"""
    try:
        return rag_service.delete_documents(source_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-consistency")
async def analyze_consistency(files: List[UploadFile] = File(...)):
    """Analyze consistency across multiple uploaded documents"""
    try:
        # Create analysis context
        analysis_context = ContractAnalysisContext()
        
        # Process uploaded files and store in RAG
        for file in files:
            try:
                content = await file.read()
                
                # Save file temporarily
                temp_path = f"uploads/{file.filename}"
                with open(temp_path, "wb") as f:
                    f.write(content)
                
                # Process through RAG
                await send_websocket_message(
                    f"Processing document through RAG: {file.filename}",
                    "RAG Processor",
                    20
                )
                
                # Process document in RAG
                doc_count = rag_service.process_and_store_document(temp_path)
                
                if doc_count > 0:
                    # Get processed documents from RAG
                    rag_docs = rag_service.get_documents_by_source(file.filename)
                    
                    # Add document to context with RAG-processed content
                    analysis_context.add_document({
                        "name": file.filename,
                        "content": "\n".join([doc["content"] for doc in rag_docs]),
                        "type": file.content_type,
                        "size": len(content),
                        "rag_processed": True,
                        "chunks": doc_count,
                        "rag_documents": rag_docs
                    })
                    
                    await send_websocket_message(
                        f"Successfully processed {doc_count} chunks from {file.filename}",
                        "RAG Processor",
                        40
                    )
                else:
                    await send_websocket_message(
                        f"Warning: No chunks extracted from {file.filename}",
                        "RAG Processor",
                        30
                    )
                
                # Clean up temp file
                os.remove(temp_path)
                
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                await send_websocket_message(
                    f"Error processing file {file.filename}: {str(e)}",
                    "Error Handler",
                    0
                )
                continue
        
        if not analysis_context.documents:
            raise HTTPException(
                status_code=400,
                detail="No documents could be processed successfully"
            )
        
        # Run analysis using RAG-processed documents
        await send_websocket_message(
            "Starting consistency analysis with RAG-processed documents...",
            "Consistency Agent",
            50
        )
        
        result = await enhanced_automated_consistency_analysis(analysis_context.documents)
        
        if not result or "error" in result:
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Analysis failed")
            )
        
        # Process and format the results
        await send_websocket_message(
            "Processing analysis results...",
            "Result Processor",
            80
        )
        
        # Extract the final output from the result
        final_output = result.get("final_output", {})
        if isinstance(final_output, str):
            try:
                final_output = json.loads(final_output)
            except:
                final_output = {"summary": final_output}
        
        analysis_result = {
            "documents_analyzed": len(analysis_context.documents),
            "total_issues": final_output.get("total_issues", 0),
            "issues_by_severity": final_output.get("issues_by_severity", {}),
            "issues_by_category": final_output.get("issues_by_category", {}),
            "detailed_issues": final_output.get("detailed_issues", []),
            "recommendations": final_output.get("recommendations", []),
            "summary": final_output.get("summary", ""),
            "rag_stats": {
                "total_documents": len(analysis_context.documents),
                "total_chunks": sum(doc.get("chunks", 0) for doc in analysis_context.documents)
            }
        }
        
        await send_websocket_message(
            "Analysis complete!",
            "Consistency Agent",
            100
        )
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Error in analyze_consistency: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/analyze-firm-standards")
async def analyze_firm_standards(files: List[UploadFile] = File(...)):
    """Analyze documents for firm standard compliance"""
    try:
        # Create a temporary directory for the uploaded files
        temp_dir = os.path.join(UPLOAD_DIR, f"temp_{uuid.uuid4()}")
        os.makedirs(temp_dir, exist_ok=True)
        
        await send_websocket_message(
            "Starting firm standards analysis...",
            "Firm Standards Analysis",
            10
        )
        
        file_paths = []
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            file_paths.append(file_path)
        
        await send_websocket_message(
            f"Processing {len(files)} documents...",
            "Firm Standards Analysis",
            20
        )
        
        # Create analysis context
        context = FirmStandardContext()
        
        # Process documents through RAG service
        processed_docs = []
        for file_path in file_paths:
            try:
                # Process document through RAG service
                num_chunks = rag_service.process_and_store_document(file_path)
                if num_chunks > 0:
                    # Get the processed documents from RAG
                    rag_docs = rag_service.get_documents_by_source(os.path.basename(file_path))
                    if rag_docs:
                        processed_docs.extend(rag_docs)
                        await send_websocket_message(
                            f"Successfully processed {len(rag_docs)} chunks from {os.path.basename(file_path)}",
                            "RAG Processor",
                            30
                        )
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                await send_websocket_message(
                    f"Error processing file {os.path.basename(file_path)}: {str(e)}",
                    "Error Handler",
                    0
                )
        
        if not processed_docs:
            raise HTTPException(
                status_code=400,
                detail="No documents could be processed successfully"
            )
        
        # Update context with processed documents
        context.processed_documents = processed_docs
        context.total_documents_analyzed = len(processed_docs)
        
        await send_websocket_message(
            "Starting style analysis...",
            "Firm Standards Analysis",
            40
        )
        
        # Perform analysis using the async function directly
        result = await enhanced_automated_style_analysis()
        
        # Clean up temporary files
        for file_path in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.rmdir(temp_dir)
        
        await send_websocket_message(
            "Analysis complete!",
            "Firm Standards Analysis",
            100
        )
        
        # Extract and format the response
        if result is None:
            return {"error": "Analysis failed or was interrupted", "success": False}
            
        # Handle RunResult object
        if hasattr(result, 'final_output'):
            final_output = result.final_output
            if not final_output:
                # Try to extract from tool results
                tool_results = []
                if hasattr(result, 'new_items'):
                    for item in result.new_items:
                        if hasattr(item, 'content') and isinstance(item.content, dict):
                            if 'final_output' in item.content:
                                tool_results.append(item.content['final_output'])
                
                if tool_results:
                    final_output = tool_results[-1]
                else:
                    final_output = {"message": "Analysis completed but no detailed results available"}
            
            return {
                "success": True,
                "analysis_complete": True,
                "result": final_output
            }
        
        # Handle dict result
        elif isinstance(result, dict):
            if "final_output" in result:
                return {
                    "success": True,
                    "analysis_complete": True,
                    "result": result["final_output"]
                }
            else:
                return {
                    "success": True,
                    "analysis_complete": True,
                    "result": result
                }
        
        # Handle other result types
        else:
            return {
                "success": True,
                "analysis_complete": True,
                "result": str(result)
            }
        
    except Exception as e:
        logger.error(f"Error in analyze_firm_standards: {str(e)}")
        await send_websocket_message(
            f"Error in firm standards analysis: {str(e)}",
            "Error Handler",
            0
        )
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    


class StyleReportRequest(BaseModel):
    report_data: Dict[str, Any]

class CoherenceReportRequest(BaseModel):
    report_data: Dict[str, Any]

@app.post("/process-style-report")
async def process_report(request: StyleReportRequest):
    try:
        # Ensure the report_data is properly structured
        if not isinstance(request.report_data, dict):
            raise HTTPException(status_code=400, detail="Invalid report data format")
            
        # Extract the report from the result if it exists
        report_data = request.report_data.get("result", request.report_data)
        
        # Process the report
        result = process_style_report(report_data)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return result
    except Exception as e:
        logger.error(f"Error processing style report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-coherence-report")
async def process_coherence_report_endpoint(request: CoherenceReportRequest):
    try:
        # Ensure the report_data is properly structured
        if not isinstance(request.report_data, dict):
            raise HTTPException(status_code=400, detail="Invalid report data format")
            
        # Extract the report from the result if it exists
        report_data = request.report_data.get("result", request.report_data)
        
        # Process the report
        result = process_coherence_report(report_data)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return result
    except Exception as e:
        logger.error(f"Error processing coherence report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/')
async def root():
    return {
        "message": "Welcome to Third Chair Backend API",
        "status": "operational",
        "version": "1.0.0"
    }

# Add health check endpoint
@app.get('/health')
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)






