from langchain_core.tools import tool
from langsmith import traceable

@traceable
@tool
def kg(query: str) -> str:
  return ""