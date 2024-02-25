import argparse
import numpy as np
import torch
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
)
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', type=str,
        default='ift_dataset',
        help='Path to the FT dataset',
    )
    parser.add_argument(
        '--checkpoint', type=str,
        default='mistralai/Mistral-7B-v0.1',
        help='Transformers pretrained model id',
    )
    parser.add_argument(
        '--ft_checkpoint', type=str,
        default='trained_model',
        help='Path to the fine-tuned model',
    )
    return parser.parse_args()


def finetune(
    pretrained_checkpoint: str,
    dataset_path: str,
    max_samples: int  = None,
    epochs_to_train: int = 10,
    test_subset: float = 0.1,
    output_dir: str = 'output',
    final_checkpoint: str = 'ft_ckpt',
):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_checkpoint,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True
        ),
        device_map='auto'
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias='none',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = load_from_disk(dataset_path)
    if max_samples:
        target_ids = np.random.randint(0, high=len(dataset), size=max_samples)
        dataset = dataset.select(target_ids)
    dataset = dataset.map(
        lambda example: tokenizer(example['prompt'], max_length=256),
        batched=True,
    )
    dataset = dataset.train_test_split(test_subset, 1. - test_subset)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        logging_steps=100,
        save_steps=1000,
        learning_rate=2e-4,
        optim='paged_adamw_8bit',
        fp16=True,
        num_train_epochs=epochs_to_train,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test']
    )
    trainer.train()
    model.save_pretrained(final_checkpoint)


if __name__ == '__main__':
    args = parse_args()
    finetune(
        pretrained_checkpoint=args.checkpoint,
        dataset_path=args.dataset,
        final_checkpoint=args.ft_checkpoint,
    )
