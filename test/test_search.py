import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.retrieve_and_reply import fast_search_engine

def test_financial_tools():
    print("financial tools test started")
    test_query = "What is the R&D Expense of 2025Q3?"
    
    print("\n--- test Fast Tool ---")
    res_fast = fast_search_engine.invoke({"query": test_query})
    print(f"Fast Tool result: {res_fast}")
    assert "2025" in res_fast 

    print("\n All test passed!")

if __name__ == "__main__":
    test_financial_tools()

