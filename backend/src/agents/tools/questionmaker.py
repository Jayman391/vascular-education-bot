from langchain_core.tools import tool
from langsmith import traceable

@traceable
@tool
def questionmaker(query: str) -> str:
  return ""