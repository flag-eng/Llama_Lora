import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel, PeftConfig

device = 'cuda'
mode_path = '/root/autodl-tmp/llama_8b_git/Meta-Llama-3-8B-Instruct'
lora_path = './lora_llama_adapter'  # lora权重路径

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto", torch_dtype=torch.bfloat16).to(device)

# 加载lora权重
config = PeftConfig.from_pretrained(lora_path)
model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)
print(tokenizer.eos_token_id)
print(tokenizer.eos_token)
while True:
    user_input = input("请输入你的问题（输入 'quit' 退出）：")
    if user_input.lower() == "quit":
        break
    messages = [
        {"role": "system", "content": "现在你要扮演皇帝身边的女人--甄嬛"},
        {"role": "user", "content": user_input}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    attention_mask = torch.ones(model_inputs.input_ids.shape, dtype=torch.long, device=device)
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        attention_mask=attention_mask,
        do_sample=True,
        top_p=0.9,
        temperature=0.5,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(response)
