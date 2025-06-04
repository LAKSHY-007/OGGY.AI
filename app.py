
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

app = FastAPI()
model_path = "./final_model"  
device = 0 if torch.cuda.is_available() else -1
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
chatbot = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device
)
class Query(BaseModel):
    text: str
@app.post("/chat")
async def chat(query: Query):
    response = chatbot(
        query.text,
        max_length=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    return {"response": response[0]["generated_text"]}
