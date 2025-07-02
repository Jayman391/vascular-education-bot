from langchain_core.tools import tool
from langsmith import traceable

@traceable
@tool
def evaluate(query: str) -> str:
  return ""