from langgraph.constants import END, START
from langgraph.graph import  StateGraph
from langgraph.prebuilt import ToolNode
from tools.evaluator import evaluate
from tools.kg import kg 
from tools.websearch import websearch

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

class QASynthesis(TypedDict):
    question: str
    user_answer: str

def synthesize_qa() -> QASynthesis:
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

def is_memory_answer_relevant():
    return "continue" if False else "search_memory"


gradingtool = ToolNode(evaluate)
kgtool = ToolNode(kg)
websearchtool = ToolNode(websearch)
memorytool = ToolNode(create_search_memory_tool(namespace=('memories', 'grader')))

gradingagent = create_react_agent(
    model=model,
    tools=[gradingtool],
    verbose=True,
    max_iterations=3,
    max_iterations_per_tool=1,
    handle_parsing_errors=False,
)
kgagent = create_react_agent(
    model=model,
    tools=[kgtool],
    verbose=True,
    max_iterations=3,
    max_iterations_per_tool=1,
    handle_parsing_errors=False,
)
websearchagent = create_react_agent(
    model=model,
    tools=[websearchtool],
    verbose=True,
    max_iterations=3,
    max_iterations_per_tool=1,
    handle_parsing_errors=False,
)
memoryagent = create_react_agent(
    model=model,
    tools=[memorytool],
    verbose=True,
    max_iterations=3,
    max_iterations_per_tool=1,
    handle_parsing_errors=False,
)

class GraderInput(TypedDict):
    question: str
    user_answer: str

class GraderOutput(TypedDict):
    grade: Union[float, bool]

grader_workflow = StateGraph(GraderState, input=GraderInput, output=GraderOutput)

grader_workflow.add_node("start", START)
grader_workflow.add_node("synthesize_qa", synthesize_qa)

grader_workflow.add_node("search_knowledge_graph", kgagent)
grader_workflow.add_node("search_web", websearchagent)
grader_workflow.add_node("search_memory", memoryagent)
grader_workflow.add_node("grading_tool", gradingagent)

grader_workflow.add_node("grade_answer", grade_answer)
grader_workflow.add_node("end", END)

grader_workflow.add_edge("search_knowledge_graph", "grading_tool")
grader_workflow.add_edge("search_web", "grading_tool")
grader_workflow.add_edge("search_memory", "grading_tool")

grader_workflow.add_edge("grading_tool", "grade_answer")

grader_workflow.add_edge("grade_answer", "end")

grader_workflow.compile(
    store=store,
    checkpointer=MemorySaver()
).with_config({'tags' : ['grader']})

@entrypoint(
      store=store,
      checkpointer=MemorySaver()
)
async def call_grader_workflow(workflow, query)-> GraderOutput:
    return await workflow.ainvoke({"messages" : [{"role": "user", "content": query}]})

