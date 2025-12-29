from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model = AutoModelForCausalLM.from_pretrained(
    "WeiboAI/VibeThinker-1.5B", dtype="auto", device="auto"
)

tokenizer = AutoTokenizer("WeiboAI/VibeThinker-1.5B")
