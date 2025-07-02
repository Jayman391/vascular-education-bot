from langgraph.store.memory import InMemoryStore

from langgraph.constants import END, START
from langgraph.graph import  StateGraph
from langgraph.prebuilt import ToolNode

from langchain_graph_retriever import GraphRetriever 
from langchain_chroma import Chroma

from openevals.llm import create_llm_as_judge
from openevals.prompts import CONCISENESS_PROMPT

from openai import Client
from langchain_openai import ChatOpenAI
from langsmith import traceable

from typing import TypedDict, Union, List, Tuple
from pydantic import BaseModel, Field


from dotenv import load_dotenv

from grader.agent import grader_workflow
from researcher.agent import researcher_workflow
from teacher.agent import teacher_workflow

langsmith_client = Client()

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o",
)


class SupervisorState(BaseModel):
    grades: List[Tuple[str, float]] = Field(
        default_factory=list,
        description="List of tuples containing question and its corresponding grade."
    )
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)

class SupervisorInput(TypedDict):
    question: str
  
class SupervisorOutput(TypedDict):
    output: str 

full_workflow = StateGraph(SupervisorState, input=SupervisorInput, output=SupervisorOutput)

full_workflow.add_node("start", START)
full_workflow.add_node("end", END)

full_workflow.add_node("grader", grader_workflow)
full_workflow.add_node("researcher", researcher_workflow)
full_workflow.add_node("teacher",teacher_workflow)
full_workflow.add_node("decider", "placeholder, decides between researching and making questions")

full_workflow.add_edge("start", "decider")
full_workflow.add_conditional_edges(
    "decider",
    {
        "research": "researcher",
        "question": "teacher",
    }
)
full_workflow.add_edge("researcher", "end")
full_workflow.add_edge("researcher", "grader")

full_workflow.add_edge("teacher", "grader")
full_workflow.add_edge("grader", "end")

full_workflow.compile()