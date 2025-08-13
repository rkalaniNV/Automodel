#!/usr/bin/env python3
"""
Simple script to send documentation index data to Pinecone using hosted embeddings.

This script reads the master index.json file and uploads documents to Pinecone
using the hosted llama-text-embed-v2 model for automatic embedding generation.

Usage:
    python scripts/send_to_pinecone_simple.py [--batch-size BATCH_SIZE] [--index-file INDEX_FILE]

Environment Variables:
    PINECONE_API_KEY: Your Pinecone API key
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

try:
    from pinecone import Pinecone
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install required packages:")
    print("pip install pinecone")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PineconeSimpleUploader:
    """Handles uploading documentation data to Pinecone with hosted embeddings."""
    
    def __init__(self, index_name: str, api_key: Optional[str] = None):
        """
        Initialize the Pinecone uploader.
        
        Args:
            index_name: Name of the Pinecone index
            api_key: Pinecone API key (if not provided, uses PINECONE_API_KEY env var)
        """
        self.index_name = index_name
        self.api_key = api_key or os.getenv('PINECONE_API_KEY')
        
        if not self.api_key:
            raise ValueError("Pinecone API key is required. Set PINECONE_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        
        # Connect to the index
        try:
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            logger.info("Using hosted llama-text-embed-v2 model for embeddings")
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone index '{self.index_name}': {e}")
            raise
    
    def prepare_document_for_pinecone(self, doc: Dict[str, Any], base_url: str = "") -> Dict[str, Any]:
        """
        Prepare a document from the index.json for Pinecone upload with hosted embeddings.
        
        Args:
            doc: Document data from index.json
            base_url: Base URL for constructing full URLs
            
        Returns:
            Dictionary formatted for Pinecone upsert_records
        """
        # Extract content for embedding
        content_for_embedding = []
        
        # Add title
        if doc.get('title'):
            content_for_embedding.append(doc['title'])
        
        # Add content
        if doc.get('content'):
            content_for_embedding.append(doc['content'])
        
        # Add summary if available
        if doc.get('summary'):
            content_for_embedding.append(doc['summary'])
        
        # Combine all content for embedding
        text_content = "\n\n".join(content_for_embedding)
        
        # For hosted embeddings, metadata goes directly at the record level
        record = {
            'id': doc.get('id', ''),
            'content': text_content,
            'title': doc.get('title', ''),
            'url': doc.get('url', ''),
            'format': doc.get('format', 'text'),
            'last_modified': doc.get('last_modified', ''),
        }
        
        # Add full URL if base_url provided
        if base_url:
            record['full_url'] = base_url + doc.get('url', '')
        
        # Add summary (truncated)
        if doc.get('summary'):
            record['summary'] = doc.get('summary')[:500]
        
        # Add heading information (truncated)
        if doc.get('headings'):
            headings_text = " | ".join([h.get('text', '') for h in doc['headings'] if h.get('text')])
            if headings_text:
                record['headings'] = headings_text[:500]
        
        # Add any additional metadata fields (with length limits)
        for key in ['description', 'tags', 'categories', 'author']:
            if doc.get(key):
                if isinstance(doc[key], list):
                    record[key] = ', '.join(str(item) for item in doc[key])[:200]
                else:
                    record[key] = str(doc[key])[:200]
        
        return record
    
    def upload_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100, base_url: str = "", namespace: str = "") -> None:
        """
        Upload documents to Pinecone in batches using hosted embeddings.
        
        Args:
            documents: List of documents from index.json
            batch_size: Number of documents to upload per batch
            base_url: Base URL for constructing full URLs
            namespace: Pinecone namespace (optional)
        """
        total_docs = len(documents)
        logger.info(f"Starting upload of {total_docs} documents to Pinecone")
        
        successful_uploads = 0
        failed_uploads = 0
        
        # Process in batches
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_docs + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
            
            # Prepare documents for Pinecone
            pinecone_records = []
            for doc in batch:
                prepared_doc = self.prepare_document_for_pinecone(doc, base_url)
                if prepared_doc and prepared_doc.get('content'):  # Ensure we have content
                    pinecone_records.append(prepared_doc)
                else:
                    failed_uploads += 1
                    logger.warning(f"Skipping document {doc.get('id', 'unknown')} - no content")
            
            if not pinecone_records:
                logger.warning(f"No valid documents in batch {batch_num}, skipping")
                continue
            
            # Upload batch to Pinecone using hosted embeddings
            try:
                if namespace:
                    self.index.upsert_records(namespace=namespace, records=pinecone_records)
                else:
                    self.index.upsert_records(records=pinecone_records)
                
                successful_uploads += len(pinecone_records)
                logger.info(f"Successfully uploaded batch {batch_num} ({len(pinecone_records)} documents)")
                
                # Small delay to avoid rate limits
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Failed to upload batch {batch_num}: {e}")
                failed_uploads += len(pinecone_records)
        
        logger.info(f"Upload completed. Success: {successful_uploads}, Failed: {failed_uploads}")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index."""
        try:
            stats = self.index.describe_index_stats()
            return stats
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}


def load_index_file(index_file_path: str) -> Dict[str, Any]:
    """
    Load the master index.json file.
    
    Args:
        index_file_path: Path to the index.json file
        
    Returns:
        Parsed JSON data
    """
    try:
        with open(index_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded index file: {index_file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load index file '{index_file_path}': {e}")
        raise


def collect_all_documents(index_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Collect all documents from the index data, including nested children.
    
    Args:
        index_data: The loaded index.json data
        
    Returns:
        List of all documents
    """
    documents = []
    
    # Add the root document
    root_doc = {
        'id': index_data.get('id', 'root'),
        'title': index_data.get('title', ''),
        'url': index_data.get('url', ''),
        'content': index_data.get('content', ''),
        'format': index_data.get('format', 'text'),
        'last_modified': index_data.get('last_modified', ''),
        'summary': index_data.get('summary', ''),
        'headings': index_data.get('headings', [])
    }
    
    # Add any additional metadata from root
    for key in ['description', 'tags', 'categories', 'author']:
        if key in index_data:
            root_doc[key] = index_data[key]
    
    documents.append(root_doc)
    
    # Add all children documents
    if 'children' in index_data:
        documents.extend(index_data['children'])
    
    logger.info(f"Collected {len(documents)} documents for upload")
    return documents


def main():
    """Main function to handle command line arguments and execute the upload."""
    parser = argparse.ArgumentParser(
        description="Upload documentation index to Pinecone using hosted embeddings"
    )
    parser.add_argument(
        '--index-file',
        default='docs/_build/html/index.json',
        help='Path to the index.json file (default: docs/_build/html/index.json)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Number of documents to upload per batch (default: 50)'
    )
    parser.add_argument(
        '--index-name',
        default='docs-site-demo-starter-kit',
        help='Pinecone index name (default: docs-site-demo-starter-kit)'
    )
    parser.add_argument(
        '--base-url',
        default='',
        help='Base URL for constructing full URLs (e.g., https://yoursite.com/)'
    )
    parser.add_argument(
        '--namespace',
        default='',
        help='Pinecone namespace (optional)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Process documents without uploading to Pinecone'
    )
    
    args = parser.parse_args()
    
    # Check if index file exists
    if not os.path.exists(args.index_file):
        logger.error(f"Index file not found: {args.index_file}")
        print(f"Please make sure the index file exists at: {args.index_file}")
        print("You may need to build your documentation first.")
        sys.exit(1)
    
    try:
        # Load the index file
        index_data = load_index_file(args.index_file)
        
        # Collect all documents
        documents = collect_all_documents(index_data)
        
        if args.dry_run:
            logger.info("DRY RUN: Would upload the following documents:")
            for doc in documents[:5]:  # Show first 5 as example
                print(f"  - {doc.get('id', 'unknown')}: {doc.get('title', 'No title')}")
            if len(documents) > 5:
                print(f"  ... and {len(documents) - 5} more documents")
            
            # Show sample prepared document
            if documents:
                uploader = PineconeSimpleUploader(args.index_name)
                sample_prepared = uploader.prepare_document_for_pinecone(documents[0], args.base_url)
                print(f"\nSample prepared document:")
                print(f"  - ID: {sample_prepared.get('id')}")
                print(f"  - Content preview: {sample_prepared.get('content', '')[:100]}...")
                print(f"  - Record keys: {list(sample_prepared.keys())}")
            return
        
        # Initialize Pinecone uploader
        uploader = PineconeSimpleUploader(args.index_name)
        
        # Show current index stats
        stats = uploader.get_index_stats()
        if stats:
            logger.info(f"Current index stats: {stats}")
        
        # Upload documents
        uploader.upload_documents(
            documents=documents,
            batch_size=args.batch_size,
            base_url=args.base_url,
            namespace=args.namespace
        )
        
        # Show final stats
        final_stats = uploader.get_index_stats()
        if final_stats:
            logger.info(f"Final index stats: {final_stats}")
        
        logger.info("Upload process completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Upload interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 