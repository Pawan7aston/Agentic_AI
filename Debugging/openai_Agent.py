from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# ====================================
# Define LangGraph State
# ====================================
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ====================================
# Initialize model
# ====================================
model = ChatOpenAI(temperature=0)

# ====================================
# Default Chat Agent Graph
# ====================================
def make_default_graph():
    graph = StateGraph(State)

    def call_model(state):
        return {"messages": [model.invoke(state["messages"])]}

    graph.add_node("agent", call_model)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)

    return graph.compile()

# ====================================
# Tool-Calling Agent Graph
# ====================================
def make_tool_graph():
    @tool
    def add(a: float, b: float):
        """Add two numbers."""
        return a + b

    model_with_tools = model.bind_tools([add])
    tool_node = ToolNode([add])

    graph = StateGraph(State)

    def call_model(state):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def should_continue(state: State):
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return END

    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges("agent", should_continue)

    return graph.compile()

# ====================================
# Export the graph for LangSmith
# ====================================
USE_TOOL_GRAPH = True
agent = make_tool_graph() if USE_TOOL_GRAPH else make_default_graph()
