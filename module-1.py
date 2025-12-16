import os, getpass
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


def _set_env(var: str):
  if not os.environ.get(var):
    os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
_set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"

def multiply(a: int, b: int) -> int:
  return a * b

def add(a: int, b: int) -> int:
  return a + b

def divide(a: int, b: int) -> int:
  return a / b

tools = [add, multiply, divide]
llm = ChatOpenAI(model="gpt-4o")

llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)