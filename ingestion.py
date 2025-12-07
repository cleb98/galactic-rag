from datapizza.clients.openai import OpenAIClient
from datapizza.core.vectorstore import VectorConfig
from datapizza.embedders import ChunkEmbedder
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.modules.captioners import LLMCaptioner
from datapizza.modules.parsers.docling import DoclingParser
from datapizza.modules.splitters import NodeSplitter
from datapizza.pipeline import IngestionPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore
from dotenv import load_dotenv  
import os

load_dotenv()  # Load environment variables from .env file

vectorstore = QdrantVectorstore(
    api_key=os.getenv("QDRANT_API_KEY"),
    host=os.getenv("QDRANT_HOST"),
    port=int(os.getenv("QDRANT_PORT")),
    https=False,
)
vectorstore.create_collection(
    "test_collection",
    vector_config=[VectorConfig(name="text-embedding-3-small", dimensions=1536)]
)

embedder_client = OpenAIEmbedder(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small",
)

ingestion_pipeline = IngestionPipeline(
    modules=[
        DoclingParser(), # choose between Docling, Azure or TextParser to parse plain text

        #LLMCaptioner(
        #    client=OpenAIClient(api_key="YOUR_API_KEY"),
        #), # This is optional, add it if you want to caption the media

        NodeSplitter(max_char=1000),             # Split Nodes into Chunks
        ChunkEmbedder(client=embedder_client),   # Add embeddings to Chunks
    ],
    vector_store=vectorstore,
    collection_name="test_collection"
)

file_path = "Dataset/knowledge_base/menu/Anima Cosmica.pdf"
ingestion_pipeline.run(file_path, metadata={"source": "user_upload"})

res = vectorstore.search(
    query_vector = [0.0] * 1536,
    collection_name="test_collection",
    k=2,
)
print(res)