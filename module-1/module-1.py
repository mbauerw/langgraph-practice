import os, getpass
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display
from langgraph.graph import START, StateGraph

load_dotenv()

def _set_env(var: str):
  if not os.environ.get(var):
    os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
_set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"

def multiply(a: int, b: int) -> int:
  """Multiply a and b.
  
    Args:
        a (int): The first number.
        b (int): The second number."""
  return a * b

def add(a: int, b: int) -> int:
  """Add a and b.
  
    Args:
        a (int): The first number.
        b (int): The second number."""
  return a + b

def divide(a: int, b: int) -> float:
  """Divide a and b.
  
    Args:
        a (int): The first number.
        b (int): The second number."""
  return a / b

tools = [add, multiply, divide]
llm = ChatOpenAI(model="gpt-4o")

llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

# sys-message
sys_msg = SystemMessage(content="You are a helpful assistant professor tasked with performing arithmetic on a set of inputs.")

def assistant(state: MessagesState):
  return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

builder = StateGraph(MessagesState)

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges(
  "assistant",
  tools_condition
)
builder.add_edge("tools", "assistant")
react_graph = builder.compile()

print(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))

messages = [HumanMessage(content="Add 3 and 4. Multiply the output by 2. Divide the output by 5")]
messages = react_graph.invoke({"messages": messages})

for m in messages['messages']:
    m.pretty_print()