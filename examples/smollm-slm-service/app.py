from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = FastAPI(title="VeloxML Small Language Model (SLM) Service")

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

print(f"Loading SLM: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="auto"
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print("SLM loaded successfully!")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 60
    temperature: float = 0.7

@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_ID, "type": "SLM"}

@app.post("/predict")
def predict(req: GenerateRequest):
    messages = [{"role": "user", "content": req.prompt}]
    formatted_prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(
        formatted_prompt,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        do_sample=True,
        pad_token_id=pipe.tokenizer.eos_token_id
    )
    raw = outputs[0]["generated_text"]
    generated = raw[len(formatted_prompt):] if raw.startswith(formatted_prompt) else raw
    return {
        "model": MODEL_ID,
        "prompt": req.prompt,
        "response": generated.strip()
    }
