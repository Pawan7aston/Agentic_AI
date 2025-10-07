import streamlit as st
from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
import os

# Load environment variables
load_dotenv()

# Set up environment keys
os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY", "")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

# Page config
st.set_page_config(page_title="Multi-Tool AI Agent", page_icon="🧠", layout="wide")

# Title and description
st.title("🧠 Multi-Tool AI Agent")
st.markdown("""
Ask anything! This agent uses **Wikipedia**, **ArXiv**, and **Tavily Search** dynamically — and shows you exactly where answers come from.
""")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input("Groq API Key", value=os.getenv("GROQ_API_KEY", ""), type="password")
    tavily_api_key = st.text_input("Tavily API Key", value=os.getenv("TAVILY_API_KEY", ""), type="password")

    model_choice = st.selectbox(
        "Choose Model",
        ["llama3-groq-70b-8192", "llama3-groq-8b-8192", "qwen/qwen3-32b"],
        index=2
    )

    clear_btn = st.button("Clear Chat")

if clear_btn:
    st.session_state.clear()
    st.rerun()

# Validate keys
if not groq_api_key or not tavily_api_key:
    st.warning("Please provide both Groq and Tavily API keys in the sidebar.")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key

# Initialize session state: store simple (role, content) tuples for UI
if "messages" not in st.session_state:
    st.session_state.messages = [
        ("system", (
            "You are a helpful assistant that answers questions using Wikipedia, ArXiv, or web search. "
            "Always mention your source: say 'According to Wikipedia', 'Research paper from ArXiv', "
            "or 'Based on recent news from Tavily' where appropriate. Be concise and accurate."
        ))
    ]

# Display chat history (only user and assistant)
for role, content in st.session_state.messages:
    if role in ["user", "assistant"]:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.write(content)

# Initialize tools
@st.cache_resource
def get_tools():
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper, name="arxiv")

    wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper, name="wikipedia")

    tavily_tool = TavilySearchResults(name="tavily_search")

    return [arxiv_tool, wiki_tool, tavily_tool]

tools = get_tools()

# Map tool names to friendly labels
TOOL_LABELS = {
    "wikipedia": "🌐 Wikipedia",
    "arxiv": "📜 ArXiv",
    "tavily_search": "🔍 Web Search (Tavily)"
}

# Convert session messages to LangChain messages for agent
def get_langchain_messages():
    lc_messages = []
    for role, content in st.session_state.messages:
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        elif role == "system":
            lc_messages.append(SystemMessage(content=content))
    return lc_messages

# Initialize LLM
llm = ChatGroq(model=model_choice, temperature=0.3)
llm_with_tools = llm.bind_tools(tools)

# Define state
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Define LLM node
def tool_calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
def create_agent():
    builder = StateGraph(State)
    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges("tool_calling_llm", tools_condition)
    builder.add_edge("tools", "tool_calling_llm")  # Return to LLM after tool use
    return builder.compile()

agent = create_agent()

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Append user message as tuple
    st.session_state.messages.append(("user", prompt))
    st.chat_message("user").write(prompt)

    # Prepare messages for agent
    lc_messages = get_langchain_messages()
    state = {"messages": lc_messages}

    # Placeholder for assistant response
    assistant_container = st.chat_message("assistant")
    message_placeholder = assistant_container.empty()
    full_response = ""

    try:
        # Stream events from the agent
        for event in agent.stream(state, stream_mode="values"):
            messages = event.get("messages", [])
            if not messages:
                continue

            message = messages[-1]

            # Handle tool calls: show which tool is being used
            if isinstance(message, AIMessage) and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call["name"].lower()
                    tool_args = tool_call["args"]
                    query_value = list(tool_args.values())[0] if tool_args else "..."
                    label = TOOL_LABELS.get(tool_name, f"🔧 {tool_name.title()}")

                    with assistant_container:
                        with st.expander(f"{label} → Query: *{query_value}*", expanded=True):
                            st.write("🔄 Retrieving information...")

            # Handle tool responses
            elif isinstance(message, ToolMessage):
                tool_call_id = message.tool_call_id
                tool_name = "unknown"
                # Find the matching tool call to get its name
                for prev_msg in reversed(lc_messages):
                    if isinstance(prev_msg, AIMessage) and hasattr(prev_msg, "tool_calls"):
                        for tc in prev_msg.tool_calls:
                            if tc["id"] == tool_call_id:
                                tool_name = tc["name"].lower()
                                break
                        if tool_name != "unknown":
                            break
                label = TOOL_LABELS.get(tool_name, "🔧 External Tool")

                with assistant_container:
                    with st.expander(f"{label} → Retrieved Data", expanded=False):
                        st.text(message.content[:600] + ("..." if len(message.content) > 600 else ""))

            # Handle final AI message
            elif isinstance(message, AIMessage) and message.content:
                full_response = message.content
                message_placeholder.markdown(full_response)

        # Save assistant response as simple tuple
        if full_response:
            st.session_state.messages.append(("assistant", full_response))
        else:
            fallback = "I couldn't retrieve an answer."
            st.session_state.messages.append(("assistant", fallback))
            message_placeholder.markdown(fallback)

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        st.session_state.messages.append(("assistant", error_msg))
        message_placeholder.markdown(error_msg)