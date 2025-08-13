import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables from .env file
load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "docs-site-demo-starter-kit"

index = pc.Index(index_name)

# Because your index is integrated with a hosted embedding model, you provide the query as text 
# and Pinecone converts the query to a dense vector automatically.
query = "Tell me about feature a."

results = index.search(
    namespace="docs-content",
    query={
        "inputs": {"text": query},
        "top_k": 3
    }
)

print(results)