from fastapi import FastAPI
from rag_agent import my_agent
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    query: str
    session_id: str

@app.post("/chat")
def chat(request: ChatRequest):
    """Endpoint to handle chat requests."""
    
    config = {"configurable": {"thread_id": request.session_id}}
    
    result = my_agent.graph.invoke(
        {"messages": [("user", request.query)]}, 
        config
    )
    
    last_message = result["messages"][-1]
    raw_content = last_message.content

    if isinstance(raw_content, list):
        final_text = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in raw_content])
    else:
        final_text = str(raw_content)
    
    return {"answer": final_text}


if __name__ == "__main__":
    import uvicorn
    print("🚀 Finance RAG Agent started，listening http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)