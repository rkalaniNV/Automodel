#!/usr/bin/env python3
"""
Example script to query your Pinecone documentation search index.

This demonstrates how to perform semantic search against your uploaded documentation
using Pinecone's hosted embeddings.
"""

import os
from pinecone import Pinecone

def search_docs(query: str, top_k: int = 5, namespace: str = "docs-content"):
    """
    Search your documentation using semantic search.
    
    Args:
        query: The search query (plain text)
        top_k: Number of results to return
        namespace: Pinecone namespace to search in
    """
    # Initialize Pinecone
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        print("❌ PINECONE_API_KEY environment variable not set")
        return
    
    try:
        pc = Pinecone(api_key=api_key)
        index = pc.Index("docs-site-demo-starter-kit")  # Your index name
        
        print(f"🔍 Search query: '{query}'")
        
        # Search using hosted embeddings
        # Pinecone automatically converts your text query to embeddings
        results = index.search(
            namespace=namespace,
            query={
                "inputs": {"text": query},
                "top_k": top_k
            }
        )
        
        # Parse the hosted embeddings response format
        hits = None
        if hasattr(results, 'result') and hasattr(results.result, 'hits'):
            hits = results.result.hits
        elif isinstance(results, dict) and 'result' in results and 'hits' in results['result']:
            hits = results['result']['hits']
        
        if not hits:
            print(f"⚠️  No results found for query: '{query}'")
            return
        
        print(f"📄 Found {len(hits)} results:\n")
        
        for i, hit in enumerate(hits, 1):
            # Parse hit structure: {'_id': '...', '_score': 0.xx, 'fields': {...}}
            doc_id = hit.get('_id', 'Unknown ID')
            score = hit.get('_score', 0)
            fields = hit.get('fields', {})
            
            print(f"{i}. Score: {score:.4f}")
            print(f"   ID: {doc_id}")
            print(f"   Title: {fields.get('title', 'N/A')}")
            print(f"   URL: {fields.get('url', 'N/A')}")
            print(f"   Format: {fields.get('format', 'N/A')}")
            
            # Show summary if available
            if 'summary' in fields and fields['summary']:
                summary = fields['summary'][:200] + "..." if len(fields['summary']) > 200 else fields['summary']
                print(f"   Summary: {summary}")
            
            # Show content preview if available
            if 'content' in fields and fields['content']:
                content = fields['content'][:150] + "..." if len(fields['content']) > 150 else fields['content']
                print(f"   Content: {content}")
            
            print()
            
    except Exception as e:
        print(f"❌ Error during search: {e}")
        print(f"   Query: '{query}'")
        print(f"   Namespace: '{namespace}'")

def test_connection():
    """Test the Pinecone connection and show index stats."""
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        print("❌ PINECONE_API_KEY environment variable not set")
        return False
    
    try:
        pc = Pinecone(api_key=api_key)
        index = pc.Index("docs-site-demo-starter-kit")
        
        # Get index stats
        stats = index.describe_index_stats()
        print("✅ Connection successful!")
        print(f"📊 Index stats:")
        print(f"   - Total vectors: {stats.get('total_vector_count', 0)}")
        print(f"   - Dimension: {stats.get('dimension', 'unknown')}")
        
        if 'namespaces' in stats and stats['namespaces']:
            print(f"   - Namespaces:")
            for ns_name, ns_info in stats['namespaces'].items():
                print(f"     - {ns_name}: {ns_info.get('vector_count', 0)} vectors")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def main():
    """Run example searches."""
    print("🚀 Pinecone Documentation Search Examples\n")
    
    # First test connection
    if not test_connection():
        return
    
    print("\n" + "="*50 + "\n")
    
    # Start with simple searches
    example_queries = [
        "Spark integration",
        "NeMo Curator",
        "Apache Spark"
    ]
    
    for query in example_queries:
        search_docs(query, top_k=3)
        print("-" * 50)

if __name__ == '__main__':
    main() 