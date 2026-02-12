import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langchain.messages import RemoveMessage, SystemMessage, AIMessage
from langgraph.graph.message import BaseMessage, add_messages
from langgraph.prebuilt import ToolNode
from typing import Annotated, Literal, Sequence, TypedDict
from pathlib import Path


load_dotenv()
api_key=os.environ.get("GOOGLE_API_KEY")
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)
    
from tools.retrieve_and_reply import fast_search_engine

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    summary: str

class RAGAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key)
        
        self.tools = [fast_search_engine]
        self.tool_node = ToolNode(self.tools)
        self.memory = InMemorySaver()
        self.graph = self._build_graph()

    def _message_content_to_text(self, content) -> str:
        """Normalize message content to plain text for summarization."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "".join(parts)
        if isinstance(content, dict):
            return str(content.get("text", ""))
        return str(content)
    
    # ---  Node: mem manager ---
    def manage_memory_func(self, state: AgentState) -> AgentState:
        """Function to manage short term memory by storing and retrieving conversation history."""
        summary = state.get("summary", "")
        messages = state.get("messages", [])
        
        chat_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
        if len(chat_messages) > 10:
            oldest_messages = chat_messages[:5]
            prompt = ChatPromptTemplate.from_template(
                "Summarize the following messages briefly.\n"
                "The current summary: {current_summary}\n"
                "The new conversations: {new_lines}\n"
            )
            
            chain = prompt | self.llm
            resp = chain.invoke({
                "current_summary": summary,
                "new_lines": "\n".join(
                    self._message_content_to_text(msg.content) for msg in oldest_messages
                ),
            })
            
            new_summary = self._message_content_to_text(getattr(resp, "content", resp))

            delete_operations = [RemoveMessage(id=m.id) for m in oldest_messages]
            
            return {
                "summary": new_summary,
                "messages": delete_operations
            }
            
        return {}
    
    # ---  Node: LLM Oracle ---
    def call_model(self, state: AgentState, config: RunnableConfig):
        """
        The decision-making node. Injects the summary and decides 
        whether to answer or call a tool.
        """
        summary = state.get("summary", "")
        messages = state["messages"]
        
        # System instructions with priority logic
       
        instructions = (
            "You are a professional financial assistant.\n"
            "1. Always check the conversation history and summary first.\n"
            "2. If information is missing, use 'fast_search_engine' to find it.\n"
            "3. IMPORTANT: If the tool returns 'ERROR_CODE: DATA_NOT_FOUND', it means the data "
            "does not exist in the financial reports. DO NOT attempt to search again "
            "with different keywords. Instead, politely inform the user that the information "
            "could not be found in the current records."
        )
        
        if summary:
            instructions += f"--- [Conversation Summary] ---\n{summary}\n"

        # Bind tools to the model so it knows how to call them
        model_with_tools = self.llm.bind_tools(self.tools)
        
        # Prepend the system instructions to the message list
        full_context = [SystemMessage(content=instructions)] + messages
        response = model_with_tools.invoke(full_context, config)
        
        return {"messages": [response]}

    # --- Conditional Routing ---
    def should_continue(self, state: AgentState) -> Literal["tools", "__end__"]:
        """
        Determines if the LLM output requires a tool call or can end the loop.
        """
        last_message = state["messages"][-1]
        # If the LLM generated 'tool_calls', route to the 'tools' node
        if last_message.tool_calls:
            return "tools"
        return "__end__"

    # --- Graph Construction ---
    def _build_graph(self):
        """
        Wires the nodes together into a single, cohesive state machine.
        """
        workflow = StateGraph(AgentState)

        # Add Nodes
        workflow.add_node("mem_manager", self.manage_memory_func)
        workflow.add_node("oracle", self.call_model)
        workflow.add_node("tools", self.tool_node)

        # Set Edges
        workflow.add_edge(START, "mem_manager")
        workflow.add_edge("mem_manager", "oracle")
        
        # The ReAct Loop: Oracle -> Tools -> Oracle
        workflow.add_conditional_edges(
            "oracle",
            self.should_continue,
        )
        workflow.add_edge("tools", "oracle")

        return workflow.compile(checkpointer=self.memory)
    

my_agent = RAGAgent()

    
