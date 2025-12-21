import os, getpass
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver


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


sys_msg = SystemMessage("You are a helpful assistant who provides accurate and pertinant information when prompted")

# Node
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

builder = StateGraph(MessagesState)

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

# Memory
memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

messages = [HumanMessage(content="Add 3 and 4. Multiply output by 4. Divide output by 2")]

# Run
messages = react_graph_memory.invoke({"messages": messages},config)
for m in messages['messages']:
  m.pretty_print()

messages = [HumanMessage(content="Multiply that by 2.")]
print(messages)

messages = react_graph_memory.invoke({"messages": messages}, config)
for m in messages['messages']:
  m.pretty_print()

