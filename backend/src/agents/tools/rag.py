from langchain_core.tools import tool
from langsmith import traceable

@traceable
@tool
def rag(query: str) -> str:
  return ""