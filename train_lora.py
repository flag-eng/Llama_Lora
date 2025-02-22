import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import TrainingArguments, Trainer


def format_instruction_prompt(instruction, input_text="", output_text=""):
    """格式化指令微调数据"""
    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    现在你要扮演皇帝身边的女人 - -甄嬛<|eot_id|><|start_header_id|>user<|end_header_id|>
    {instruction} {input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    return template.format(instruction=instruction, input=input_text)


# 处理数据集并应用 Prompt 格式
def preprocess_function(example, tokenizer):
    MAX_LENGTH = 512  # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n现在你要扮演皇帝身边的女人 - -甄嬛<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def process_data(tokenizer):
    dataset = load_dataset("json", data_files="./data/huanhuan.json", split='train')  # 确保 JSON 结构正确
    tokenized_data = dataset.map(preprocess_function, fn_kwargs={'tokenizer': tokenizer})

    return tokenized_data


def creat_model():
    # 选择 LLaMA 8B
    # model_name = "meta-llama/Llama-2-7b-hf"
    model_dir = "/root/autodl-tmp/llama_8b_git/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # 指定填充标记(pad_token)使用结束标记(eos_token)。pad_token是tokenizer中用于补足输入序列长度的填充标记,默认是 [PAD]。
    # eos_token是tokenizer中用于表示序列结束的标记,默认是 [SEP]
    # padding_side 设置为“right”以修复 fp16 的问题
    # train的时候需要padding在右边，并在句末加入eos，否则模型永远学不会什么时候停下
    # test的时候需要padding在左边，否则模型生成的结果可能全为eos
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="auto")

    # 配置 LoRA 适配器
    lora_config = LoraConfig(
        r=8,  # 低秩适配参数
        lora_alpha=32,
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False  # 训练模式
    )

    # 应用 LoRA 适配器
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # use_cache是对解码速度的优化，在解码器解码时，存储每一步输出的hidden-state用于下一步的输入
    # 因为后面会开启gradient checkpoint，中间激活值不会存储，因此use_cahe=False
    # 设置张量并行
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, tokenizer


def train(model, tokenizer, dataset):
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./lora_llama",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=6,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=100,
        save_on_each_node=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # 解决use_reentrant警告
        gradient_checkpointing=True,
        report_to=["tensorboard"]  # 将指标记录到Tensorboard
    )
    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    # 开始训练
    trainer.train()

    # 保存 LoRA 适配器
    model.save_pretrained("./lora_llama_adapter")


def main():
    model, tokenizer = creat_model()

    dataset = process_data(tokenizer)

    train(model, tokenizer, dataset)


if __name__ == '__main__':
    main()
