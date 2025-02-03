from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import pipeline
from pydantic import BaseModel

app = FastAPI()

# Load Hugging Face model for text generation
generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct")

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = generator(request.message, max_length=200, do_sample=True)
        return JSONResponse(content={"response": response[0]["generated_text"]})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
