from fastapi import WebSocket
from typing import List

# Store active WebSocket connections
active_connections: List[WebSocket] = []

async def send_websocket_message(message: str, agent: str = None, progress: int = None):
    """Send message to all connected WebSocket clients"""
    for connection in active_connections:
        try:
            await connection.send_json({
                "message": message,
                "agent": agent,
                "progress": progress
            })
        except:
            active_connections.remove(connection) 