import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain.messages import SystemMessage, AIMessage
from app.rag_agent import RAGAgent


class DummyLLM:
    def invoke(self, _input):
        return AIMessage(content="summary")
    __call__ = invoke


def _dummy_msg(tool_calls):
    class DummyMsg:
        def __init__(self, tool_calls):
            self.tool_calls = tool_calls
    return DummyMsg(tool_calls)


def test_should_continue_end():
    agent = RAGAgent.__new__(RAGAgent)
    state = {"messages": [_dummy_msg([])]}
    assert agent.should_continue(state) == "__end__"


def test_should_continue_tools():
    agent = RAGAgent.__new__(RAGAgent)
    state = {"messages": [_dummy_msg([{
        "name": "fast_search_engine",
        "args": {"query": "x"},
        "id": "call_1",
        "type": "tool_call",
    }])]}
    assert agent.should_continue(state) == "tools"


def test_manage_memory_noop_under_limit():
    agent = RAGAgent.__new__(RAGAgent)
    state = {
        "summary": "",
        "messages": [SystemMessage(content="sys"), AIMessage(content="hi")],
    }
    assert agent.manage_memory_func(state) == {}


def test_manage_memory_triggers_summarize():
    agent = RAGAgent.__new__(RAGAgent)
    agent.llm = DummyLLM()
    messages = [AIMessage(content=f"m{i}", id=f"m{i}") for i in range(11)]
    state = {"summary": "", "messages": messages}

    out = agent.manage_memory_func(state)

    assert out["summary"] == "summary"
    assert len(out["messages"]) == 5
    removed_ids = [m.id for m in out["messages"]]
    assert removed_ids == [f"m{i}" for i in range(5)]


def _run_tests():
    test_should_continue_end()
    test_should_continue_tools()
    test_manage_memory_noop_under_limit()
    test_manage_memory_triggers_summarize()
    print("All tests passed.")


if __name__ == "__main__":
    _run_tests()
