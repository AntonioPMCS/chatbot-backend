from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import requests
import os
from pydantic import BaseModel

app = FastAPI()

# Set Hugging Face API details
HF_API_KEY = os.getenv("HF_API_KEY")  # Set this in Railway's environment variables
HF_MODEL = "mistralai/Mistral-7B-Instruct"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        payload = {"inputs": request.message, "parameters": {"max_length": 200, "do_sample": True}}
        response = requests.post(HF_API_URL, headers=HEADERS, json=payload)
        response_json = response.json()
        
        if "error" in response_json:
            return JSONResponse(content={"error": response_json["error"]}, status_code=500)
        
        return JSONResponse(content={"response": response_json[0]["generated_text"]})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
