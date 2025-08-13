#!/usr/bin/env python3
"""
RAG Chatbot for Documentation Search

This script combines Pinecone semantic search with OpenAI's LLM to create
a chatbot that can answer questions based on your documentation.

Updated to use Pinecone's native llama-text-embed-v2 for embeddings,
while still using OpenAI for text generation.

ALTERNATIVE APPROACHES TO ELIMINATE OPENAI ENTIRELY:
1. Use Hugging Face Transformers with local models (e.g., Llama, Mistral)
2. Use Ollama for local LLM inference
3. Use other cloud LLM providers (Anthropic, Google, etc.)
4. Wait for Pinecone to add native text generation models
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from pinecone import Pinecone
import openai

# Load environment variables from .env file
load_dotenv()

class DocumentationRAG:
    """RAG system for documentation Q&A using Pinecone native inference."""
    
    def __init__(self, pinecone_index_name: str = "docs-site-demo-starter-kit"):
        """Initialize the RAG system."""
        # Initialize Pinecone (handles embedding via native llama-text-embed-v2)
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(pinecone_index_name)
        
        # Initialize OpenAI (only needed for text generation, not embeddings)
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
    
    def search_documents(self, query: str, top_k: int = 5, namespace: str = "docs-content") -> List[Dict[str, Any]]:
        """
        Search for relevant documents using Pinecone native inference.
        
        This now uses Pinecone's llama-text-embed-v2 model for embeddings,
        eliminating the need for OpenAI embeddings.
        """
        try:
            # Pinecone handles embedding the query automatically with native inference
            results = self.index.search(
                namespace=namespace,
                query={
                    "inputs": {"text": query},  # Text gets embedded by llama-text-embed-v2
                    "top_k": top_k
                }
            )
            
            # Extract and format results
            documents = []
            for hit in results.result.hits:
                fields = hit.get('fields', {})
                documents.append({
                    'score': hit.get('_score', 0),
                    'title': fields.get('title', 'Unknown'),
                    'url': fields.get('url', ''),
                    'content': fields.get('content', ''),
                    'summary': fields.get('summary', '')
                })
            
            return documents
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def generate_answer(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """
        Generate an answer using OpenAI based on retrieved documents.
        
        This is the one part that still requires OpenAI (or another LLM service)
        since llama-text-embed-v2 is an embedding model, not a completion model.
        """
        
        # Prepare context from retrieved documents
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"""
Document {i}: {doc['title']}
URL: {doc['url']}
Content: {doc['content'][:1000]}...
""")
        
        context = "\n".join(context_parts)
        
        # Create the prompt
        prompt = f"""You are a helpful documentation assistant. Answer the user's question based on the provided documentation context. 

If the answer is not clearly found in the documentation, say so and suggest what information might be available.

Context from documentation:
{context}

User Question: {query}

Answer based on the documentation:"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # or "gpt-3.5-turbo" for faster/cheaper
                messages=[
                    {"role": "system", "content": "You are a helpful documentation assistant that answers questions based only on the provided documentation context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def ask(self, question: str, show_sources: bool = True) -> Dict[str, Any]:
        """
        Ask a question and get a RAG-generated answer.
        
        Args:
            question: The question to ask
            show_sources: Whether to include source documents
            
        Returns:
            Dictionary with answer and source documents
        """
        print(f"🔍 Searching for: '{question}'")
        
        # Step 1: Retrieve relevant documents
        documents = self.search_documents(question, top_k=3)
        
        if not documents:
            return {
                "answer": "I couldn't find any relevant documentation to answer your question.",
                "sources": []
            }
        
        print(f"📄 Found {len(documents)} relevant documents")
        
        # Step 2: Generate answer using LLM
        print("🤖 Generating answer...")
        answer = self.generate_answer(question, documents)
        
        result = {
            "answer": answer,
            "sources": documents if show_sources else []
        }
        
        return result

def main():
    """Interactive RAG chatbot."""
    print("🚀 Documentation RAG Chatbot")
    print("Ask questions about your documentation!\n")
    
    try:
        rag = DocumentationRAG()
        print("✅ RAG system initialized successfully!\n")
        
        # Example questions
        example_questions = [
            "How do I integrate NeMo Curator with Apache Spark?",
            "What are the deployment options?",
            "How do I configure data processing workflows?",
            "What integrations are available?"
        ]
        
        print("📝 Example questions you can ask:")
        for i, q in enumerate(example_questions, 1):
            print(f"   {i}. {q}")
        print()
        
        while True:
            question = input("❓ Your question (or 'quit' to exit): ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            print()
            
            # Get RAG answer
            result = rag.ask(question)
            
            # Display answer
            print("💬 Answer:")
            print(result["answer"])
            print()
            
            # Display sources
            if result["sources"]:
                print("📚 Sources:")
                for i, doc in enumerate(result["sources"], 1):
                    print(f"   {i}. {doc['title']} (Score: {doc['score']:.3f})")
                    print(f"      URL: {doc['url']}")
                    if doc['summary']:
                        print(f"      Summary: {doc['summary'][:100]}...")
                print()
            
            print("-" * 50)
            print()
    
    except ValueError as e:
        print(f"❌ Setup error: {e}")
        print("\nMake sure you have both environment variables set:")
        print("- PINECONE_API_KEY")
        print("- OPENAI_API_KEY")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    main() 