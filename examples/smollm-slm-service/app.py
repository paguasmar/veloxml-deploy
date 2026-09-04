from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()
pipe = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-135M-Instruct")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: dict):
    messages = [{"role": "user", "content": data["prompt"]}]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"response": pipe(prompt, max_new_tokens=60, return_full_text=False)[0]["generated_text"].strip()}
