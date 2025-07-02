from langgraph.constants import END, START
from langgraph.graph import  StateGraph
from langgraph.prebuilt import ToolNode
from tools.evaluator import evaluate
from tools.kg import kg 
from tools.websearch import websearch
from tools.rag import rag
from tools.questionmaker import questionmaker

from langmem import create_search_memory_tool
from langgraph.store.memory import InMemoryStore
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import MemorySaver

from typing import TypedDict, Union, List, Tuple
from pydantic import BaseModel, Field

from langchain.agents import create_react_agent


class TeacherState(BaseModel):
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
questiontool = ToolNode(questionmaker)

questionmakeragent = create_react_agent(
    model=model,
    tools=[questiontool, memorytool],
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


class TeacherInput(TypedDict):
    question: str
    user_answer: str

class TeacherOutput(TypedDict):
    grade: Union[float, bool]

teacher_workflow = StateGraph(TeacherState, input=TeacherInput, output=TeacherOutput)

teacher_workflow.add_node("start", START)
teacher_workflow.add_node("synthesize_question", synthesize_question)

teacher_workflow.add_node("search_knowledge_graph", kgagent)
teacher_workflow.add_node("search_rag", ragagent)
teacher_workflow.add_node("search_web", websearchagent)
teacher_workflow.add_node("question_maker", questionmakeragent)

teacher_workflow.add_node("end", END)

teacher_workflow.add_edge("start", "synthesize_question")
teacher_workflow.add_edge("synthesize_question", "search_knowledge_graph")
teacher_workflow.add_edge("synthesize_question", "search_rag")
teacher_workflow.add_edge("synthesize_question", "search_web")

teacher_workflow.add_edge("search_knowledge_graph", "question_maker")
teacher_workflow.add_edge("search_web", "question_maker")
teacher_workflow.add_edge("search_memory", "question_maker")
teacher_workflow.add_edge("search_rag", "question_maker")

teacher_workflow.add_edge("question_maker", "end")

teacher_workflow.compile(
    store=store,
    checkpointer=MemorySaver()
).with_config({'tags' : ['teacher']})

@entrypoint(
      store=store,
      checkpointer=MemorySaver()
)
async def call_teacher_workflow(workflow, query)-> TeacherOutput:
    return await workflow.ainvoke({"messages" : [{"role": "user", "content": query}]})

