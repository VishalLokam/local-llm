from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="WeiboAI/VibeThinker-1.5B"
)
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="WeiboAI/VibeThinker-1.5B"
)

input = tokenizer("Who are you?", return_tensors="pt")

print(input)
