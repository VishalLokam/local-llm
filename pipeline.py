## pipeline
from transformers import pipeline

pipe = pipeline("text-generation", model="WeiboAI/VibeThinker-1.5B", device=0)
messages = [
    {"role": "user", "content": "Who are you?"},
]
context = pipe(messages)
answer = context[0].get("generated_text")

print(answer[1].get("content"))
