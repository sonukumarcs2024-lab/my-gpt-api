from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 200

@app.post("/chat")
def chat(request: ChatRequest):

    inputs = tokenizer(request.prompt, return_tensors="pt")

    outputs = model.generate(
        inputs["input_ids"],
        max_length=request.max_tokens
    )

    response = tokenizer.decode(outputs[0])

    return {"response": response}
