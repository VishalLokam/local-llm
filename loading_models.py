from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="WeiboAI/VibeThinker-1.5B",
    dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="WeiboAI/VibeThinker-1.5B"
)

model_input = tokenizer("Who are you?", return_tensors="pt").to(model.device)

generate_ids = model.generate(**model_input)

model_output = tokenizer.batch_decode(generate_ids, skip_special_token=True)[0]

print(model_output)
