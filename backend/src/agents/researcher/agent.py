from langgraph.constants import END, START
from langgraph.graph import  StateGraph
from langgraph.prebuilt import ToolNode
from tools.evaluator import evaluate
from tools.kg import kg 
from tools.websearch import websearch
from tools.rag import rag

from langmem import create_search_memory_tool
from langgraph.store.memory import InMemoryStore
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import MemorySaver

from typing import TypedDict, Union, List, Tuple
from pydantic import BaseModel, Field

from langchain.agents import create_react_agent


class GraderState(BaseModel):
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

model = "gpt-4o-mini"

class QuestionSynthesis(TypedDict):
    question: str

def synthesize_question() -> QuestionSynthesis:
    pass   

class AnswerGrade(TypedDict):
    grade : float | bool
    user_answer : str 
    tool_answer : str 

def grade_answer() -> AnswerGrade:
    pass

def is_kg_answer_relevant():
    return "continue" if False else "search_knowledge_graph"

def is_web_answer_relevant():
    return "continue" if False else "search_web"

def is_rag_answer_relevant():
    return "continue" if False else "search_rag"


synthesistool = ToolNode(evaluate)
kgtool = ToolNode(kg)
ragtool = ToolNode(rag)
websearchtool = ToolNode(websearch)
memorytool = ToolNode(create_search_memory_tool(namespace=('memories', 'researcher')))

synthesisagent = create_react_agent(
    model=model,
    tools=[synthesistool, memorytool],
    verbose=True,
    max_iterations=3,
    max_iterations_per_tool=1,
    handle_parsing_errors=False,
)
kgagent = create_react_agent(
    model=model,
    tools=[kgtool, memorytool],
    verbose=True,
    max_iterations=3,
    max_iterations_per_tool=1,
    handle_parsing_errors=False,
)
ragagent = create_react_agent(
    model=model,
    tools=[ragtool, memorytool],
    verbose=True,
    max_iterations=3,
    max_iterations_per_tool=1,
    handle_parsing_errors=False,
)

websearchagent = create_react_agent(
    model=model,
    tools=[websearchtool, memorytool],
    verbose=True,
    max_iterations=3,
    max_iterations_per_tool=1,
    handle_parsing_errors=False,
)


class ResearchInput(TypedDict):
    question: str
    
class ResearchOutput(TypedDict):
    rag_output: str
    kg_output: str
    web_output: str
    urls : List[str]

researcher_workflow = StateGraph(GraderState, input=ResearchInput, output=ResearchOutput)

researcher_workflow.add_node("start", START)
researcher_workflow.add_node("synthesize_question", synthesize_question)

researcher_workflow.add_node("search_knowledge_graph", kgagent)
researcher_workflow.add_node("search_rag", ragagent)
researcher_workflow.add_node("search_web", websearchagent)

researcher_workflow.add_node("synthesis_tool", synthesisagent)

researcher_workflow.add_node("end", END)

researcher_workflow.add_edge("start", "synthesize_question")
researcher_workflow.add_edge("synthesize_question", "search_knowledge_graph")
researcher_workflow.add_edge("synthesize_question", "search_rag")
researcher_workflow.add_edge("synthesize_question", "search_web")

researcher_workflow.add_edge("search_knowledge_graph", "synthesis_tool")
researcher_workflow.add_edge("search_web", "synthesis_tool")
researcher_workflow.add_edge("search_memory", "synthesis_tool")
researcher_workflow.add_edge("search_rag", "synthesis_tool")

researcher_workflow.add_edge("synthesis_tool", "synthesize_answer")

researcher_workflow.add_edge("synthesize_answer", "end")

researcher_workflow.compile(
    store=store,
    checkpointer=MemorySaver()
).with_config({'tags' : ['researcher']})

@entrypoint(
      store=store,
      checkpointer=MemorySaver()
)
async def call_researcher_workflow(workflow, query)-> ResearchOutput:
    return await workflow.ainvoke({"messages" : [{"role": "user", "content": query}]})

