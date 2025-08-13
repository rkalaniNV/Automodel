#!/usr/bin/env python3
"""
Simple test script to validate Pinecone setup for hosted embeddings.

This script tests your Pinecone connection and index configuration
for the hosted llama-text-embed-v2 embedding model.

Usage:
    python scripts/test_pinecone_setup.py
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, Optional

try:
    from pinecone import Pinecone
    DEPENDENCIES_OK = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Please install: pip install pinecone")
    DEPENDENCIES_OK = False


def test_pinecone_connection(index_name: str) -> bool:
    """Test Pinecone connection and index access."""
    print("🔗 Testing Pinecone connection...")
    
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        print("❌ PINECONE_API_KEY environment variable not set")
        return False
    
    try:
        pc = Pinecone(api_key=api_key)
        print("✅ Pinecone client initialized")
        
        # Test index connection
        index = pc.Index(index_name)
        print(f"✅ Connected to index: {index_name}")
        
        # Get index stats
        stats = index.describe_index_stats()
        print(f"📊 Index stats:")
        print(f"   - Total vectors: {stats.get('total_vector_count', 0)}")
        print(f"   - Dimension: {stats.get('dimension', 'unknown')}")
        
        # Show namespaces if any exist
        if 'namespaces' in stats and stats['namespaces']:
            print(f"   - Namespaces:")
            for ns_name, ns_info in stats['namespaces'].items():
                print(f"     - {ns_name}: {ns_info.get('vector_count', 0)} vectors")
        else:
            print(f"   - Namespaces: None")
        
        return True
        
    except Exception as e:
        print(f"❌ Pinecone connection failed: {e}")
        return False


def test_index_file(index_file_path: str) -> Optional[Dict[str, Any]]:
    """Test loading and parsing the index file."""
    print(f"📁 Testing index file: {index_file_path}")
    
    if not os.path.exists(index_file_path):
        print(f"❌ Index file not found: {index_file_path}")
        print("   Make sure your documentation is built first")
        return None
    
    try:
        with open(index_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("✅ Index file loaded successfully")
        
        # Analyze structure
        total_docs = 1 + len(data.get('children', []))  # Root + children
        print(f"📄 Found {total_docs} documents")
        
        # Show sample document structure
        sample_doc = data
        print("📋 Sample document structure:")
        for key in ['id', 'title', 'url', 'content', 'format', 'summary']:
            if key in sample_doc:
                value = sample_doc[key]
                if isinstance(value, str) and len(value) > 50:
                    value = value[:50] + "..."
                print(f"   - {key}: {value}")
        
        return data
        
    except Exception as e:
        print(f"❌ Failed to load index file: {e}")
        return None


def test_hosted_embeddings(index_name: str) -> bool:
    """Test that the index is configured for hosted embeddings."""
    print("🤖 Testing hosted embedding configuration...")
    
    # For hosted embeddings, we just need to verify the index exists and has the right dimensions
    # The actual embedding model is handled by Pinecone
    
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        print("❌ PINECONE_API_KEY not set")
        return False
    
    try:
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        
        expected_dim = 1024  # llama-text-embed-v2 dimensions
        actual_dim = stats.get('dimension')
        
        if actual_dim == expected_dim:
            print(f"✅ Index configured for hosted embeddings ({actual_dim} dimensions)")
            print("✅ llama-text-embed-v2 model will handle embedding generation automatically")
            return True
        else:
            print(f"⚠️  Unexpected dimensions: {actual_dim} (expected {expected_dim})")
            print("   Your index may not be configured for llama-text-embed-v2")
            return False
            
    except Exception as e:
        print(f"❌ Failed to verify hosted embedding config: {e}")
        return False


def run_simple_test(index_name: str, index_file: str) -> bool:
    """Run simplified tests for hosted embeddings."""
    print("🧪 Starting Pinecone hosted embeddings test\n")
    
    all_tests_passed = True
    
    # Test 1: Dependencies
    if not DEPENDENCIES_OK:
        print("❌ Dependency test failed")
        all_tests_passed = False
    else:
        print("✅ Pinecone package available")
    
    print()
    
    # Test 2: Pinecone connection
    pinecone_ok = test_pinecone_connection(index_name)
    all_tests_passed = all_tests_passed and pinecone_ok
    
    print()
    
    # Test 3: Hosted embedding configuration
    if pinecone_ok:
        embedding_ok = test_hosted_embeddings(index_name)
        all_tests_passed = all_tests_passed and embedding_ok
    
    print()
    
    # Test 4: Index file
    index_data = test_index_file(index_file)
    index_ok = index_data is not None
    all_tests_passed = all_tests_passed and index_ok
    
    print("\n" + "="*50)
    
    if all_tests_passed:
        print("🎉 All tests passed! You're ready to upload to Pinecone.")
        print("\nNext steps:")
        print("1. Run: python scripts/send_to_pinecone_simple.py --dry-run --namespace docs-content")
        print("2. If dry run looks good: python scripts/send_to_pinecone_simple.py --namespace docs-content")
        print("3. Or use make: make docs-pinecone-update PINECONE_ARGS='--namespace docs-content'")
    else:
        print("❌ Some tests failed. Please fix the issues above before uploading.")
        print("\nCommon fixes:")
        print("- Set PINECONE_API_KEY environment variable")
        print("- Build your documentation first: make docs-html")
        print("- Ensure your Pinecone index supports hosted embeddings")
    
    return all_tests_passed


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test Pinecone setup for hosted embeddings")
    parser.add_argument(
        '--index-name',
        default='docs-site-demo-starter-kit',
        help='Pinecone index name (default: docs-site-demo-starter-kit)'
    )
    parser.add_argument(
        '--index-file',
        default='docs/_build/html/index.json',
        help='Path to index file (default: docs/_build/html/index.json)'
    )
    
    args = parser.parse_args()
    
    success = run_simple_test(args.index_name, args.index_file)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main() 