import os
from fastmcp import FastMCP
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext,
    load_index_from_storage
)
from pydantic import Field


documents_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
index_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index")

if not os.path.exists(index_dir):
    os.makedirs(index_dir)
    documents = SimpleDirectoryReader(documents_dir).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=index_dir)
else:
    storage_context = StorageContext.from_defaults(persist_dir=index_dir)
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()

mcp_server = FastMCP(name="RAG_server")


# Define the query tool
# Note: tool does not define a return type, so results may be unnecessarily verbose
@mcp_server.tool(description="Retrieve relevant information extracted from the Nike 2023 Annual Report.\
     Submit a query, and retrieve results with semantic similarity.")
def query(query: str = Field(..., description="The query to search the Nike 2023 Annual Report index for")):
    return query_engine.query(query)


if __name__ == "__main__":
    mcp_server.run(show_banner=False)