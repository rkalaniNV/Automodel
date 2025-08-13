#!/usr/bin/env python3
"""
RAG Web Server for Documentation Search

This script creates a web API that serves the RAG functionality
from rag_chatbot to the enhanced search demo web interface.
"""

import os
import json
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pinecone import Pinecone

# Load environment variables from .env file
load_dotenv()

# Import the RAG system from the existing script
from rag_chatbot import DocumentationRAG

app = Flask(__name__)
CORS(app, 
     origins=["http://localhost:5000", "http://127.0.0.1:5000", "http://localhost:8000", "http://127.0.0.1:8000"],
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"])  # Enhanced CORS configuration

# Initialize RAG system
rag = None

def initialize_rag():
    """Initialize the RAG system with error handling."""
    global rag
    try:
        rag = DocumentationRAG()
        print("✅ RAG system initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        print("\nMake sure you have both environment variables set:")
        print("- PINECONE_API_KEY")
        print("- OPENAI_API_KEY")
        return False

@app.route('/')
def index():
    """Serve the enhanced search demo page."""
    return send_from_directory('docs', 'enhanced-search-demo.html')

@app.route('/docs/<path:filename>')
def serve_docs(filename):
    """Serve static documentation files."""
    return send_from_directory('docs', filename)

@app.route('/_static/<path:filename>')
def serve_static(filename):
    """Serve static assets."""
    return send_from_directory('docs/_static', filename)

@app.route('/_build/<path:filename>')
def serve_build(filename):
    """Serve built documentation files."""
    return send_from_directory('docs/_build', filename)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if the RAG system is healthy."""
    return jsonify({
        'status': 'healthy' if rag else 'unhealthy',
        'rag_initialized': rag is not None
    })

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    """Handle RAG chat requests."""
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'OK'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    
    print(f"🔍 Chat endpoint called with method: {request.method}")
    print(f"🔍 Request headers: {dict(request.headers)}")
    print(f"🔍 Request content type: {request.content_type}")
    
    if not rag:
        return jsonify({
            'error': 'RAG system not initialized',
            'message': 'Please check your API keys and restart the server'
        }), 500
    
    try:
        data = request.get_json()
        print(f"🔍 Request data: {data}")
        
        if not data or 'message' not in data:
            return jsonify({'error': 'Missing message in request'}), 400
        
        question = data['message'].strip()
        if not question:
            return jsonify({'error': 'Empty message'}), 400
        
        print(f"🔍 Processing question: '{question}'")
        
        # Get RAG response
        result = rag.ask(question, show_sources=True)
        
        # Format response for web UI
        response = {
            'answer': result['answer'],
            'sources': []
        }
        
        # Format sources for web display
        if result.get('sources'):
            for doc in result['sources']:
                source = {
                    'title': doc.get('title', 'Unknown Document'),
                    'url': doc.get('url', '#'),
                    'score': doc.get('score', 0),
                    'summary': doc.get('summary', '')
                }
                response['sources'].append(source)
        
        print(f"✅ Generated response with {len(response['sources'])} sources")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error processing chat request: {e}")
        return jsonify({
            'error': 'Failed to process request',
            'message': str(e)
        }), 500

@app.route('/api/search', methods=['POST'])
def search():
    """Handle search requests (for potential future use)."""
    if not rag:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query in request'}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({'error': 'Empty query'}), 400
        
        # Use the search functionality from RAG system
        documents = rag.search_documents(query, top_k=data.get('top_k', 5))
        
        # Format for web display
        results = []
        for doc in documents:
            result = {
                'title': doc.get('title', 'Unknown'),
                'url': doc.get('url', '#'),
                'content': doc.get('content', ''),
                'summary': doc.get('summary', ''),
                'score': doc.get('score', 0)
            }
            results.append(result)
        
        return jsonify({'results': results})
        
    except Exception as e:
        print(f"❌ Error processing search request: {e}")
        return jsonify({
            'error': 'Failed to process search',
            'message': str(e)
        }), 500

# Add a catch-all route to help debug routing issues
@app.route('/api/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'])
def api_catchall(path):
    """Catch-all for API routes to help debug routing issues."""
    print(f"🔍 API catchall triggered for path: {path}")
    print(f"🔍 Method: {request.method}")
    print(f"🔍 Headers: {dict(request.headers)}")
    return jsonify({
        'error': f'Route not found: /api/{path}',
        'method': request.method,
        'available_routes': [
            'GET /api/health',
            'POST /api/chat',
            'POST /api/search'
        ]
    }), 404

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

def main():
    """Run the web server."""
    print("🚀 Starting RAG Web Server...")
    
    # Initialize RAG system
    if not initialize_rag():
        print("⚠️  RAG system failed to initialize, but server will still start")
        print("   Chat functionality will be disabled until API keys are configured")
    
    print("\n📱 Server starting on http://localhost:5000")
    print("🌐 Access the enhanced search demo at: http://localhost:5000")
    print("\n💡 API Endpoints:")
    print("   - GET  /                 - Enhanced search demo page")
    print("   - GET  /api/health       - Health check")
    print("   - POST /api/chat         - RAG chat (requires: {'message': 'your question'})")
    print("   - POST /api/search       - Document search (requires: {'query': 'search terms'})")
    print("\n🛑 Press Ctrl+C to stop the server")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # Avoid double initialization in debug mode
    )

if __name__ == '__main__':
    main() 