from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import openai
import os
from pydantic import BaseModel

app = FastAPI()

# Set your OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.Client(api_key=OPENAI_API_KEY)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                      {"role": "user", "content": request.message}]
        )
        return JSONResponse(content={"response": response.choices[0].message.content})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
