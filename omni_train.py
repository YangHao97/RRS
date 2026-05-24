from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, AutoProcessor, Qwen2AudioForConditionalGeneration, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from transformers.generation import GenerationConfig
import random
import librosa
import numpy as np
import json
from tqdm import tqdm
import re
import os
import csv
from peft import LoraConfig, TaskType, get_peft_model
import torch
import torch.nn as nn
import logging
os.environ["WANDB_DISABLED"] = "true"
logging.basicConfig(filename='omni_top51.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
logging.info('Start of training')

processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto", device_map="cuda")
model.disable_talker()

harmful_dataset = load_dataset("json", data_files='./data/harmful.json', split="train")
benign_dataset = load_dataset("json", data_files='./data/benign.json', split="train")
train_dataset = concatenate_datasets([harmful_dataset, benign_dataset])

def batch_inputs(audios_list, prompts_list):
    conversations = [[{"role": "system","content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]}, {"role": "user", "content": [{"type": "audio", "audio": audio},{"type": "text", "text": prompt}]}] for audio, prompt in zip(audios_list, prompts_list)]

    USE_AUDIO_IN_VIDEO = True
    text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)

    audios, images, videos = process_mm_info(conversations, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    model_inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    model_inputs = model_inputs.to(model.dtype)

    return model_inputs



def data_collate_fn(batch):
    def collect_audio_text():
        HB_list = []
        for example in batch:
            if "B_" in example["path"]:
                HB_list.append("B")
            else:
                HB_list.append("H")

        # load delta
        avg_harmful_rp = torch.tensor(np.genfromtxt("./omni_train_data/delta_weight_top51_rp_wp.csv", delimiter=","), dtype=model.dtype)

        audios = ["./BeaverTails_audio/" + example["path"] + ".wav" for example in batch]

        tgt_rp = []
        rp_path = [example["path"] + ".csv" for example in batch]
        # set the optimization target
        # harmful + delta
        # benign - delta
        # input representations are from "WITHOUT refusal prompt" version
        for rp in rp_path:
            if "B_" in rp:
                tgt_rp.append((torch.tensor(np.genfromtxt("./omni_audio_benign_rp/" + rp, delimiter=","), dtype=model.dtype) - avg_harmful_rp).unsqueeze(dim=0))
            else:
                tgt_rp.append((torch.tensor(np.genfromtxt("./omni_audio_harmful_rp/" + rp, delimiter=","), dtype=model.dtype) + avg_harmful_rp).unsqueeze(dim=0))

        tgt_rp = torch.cat(tgt_rp, dim=0)

        # prompt pool
        prompt_list = ["Answer the question in the audio.", 
                       "Follow the instruction in the audio.", 
                       "Generate a response following the audio.", 
                       "Provide output based on the audio content.", 
                       "Please generate response for the question in the audio.",
                       "Give a response according to the audio content.",
                       "Produce a reply following the instructions from the audio.",
                       "Generate the appropriate response from the audio provided.",
                       "Write down detailed steps as requested by the audio.",
                       "Please provide a guideline for solving the question in the audio."]
        random.shuffle(prompt_list)
        prompts = [prompt_list[idx%10] for idx, audio in enumerate(audios)]
        
        inputs_audio_text = {}
        inputs_audio_text["audio_text"] = batch_inputs(audios, prompts)
        inputs_audio_text["HB_list"] = HB_list
        inputs_audio_text["tgt_rp"] = tgt_rp
        return inputs_audio_text

    return collect_audio_text()
    

class Reset_Trainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
        loss = 0

        # Note: Please revise the code of Model, and only return the last layer's hidden state.
        outputs = model.module.model.thinker(**inputs["audio_text"])
        HB_list = inputs["HB_list"]
        tgt_rp = inputs["tgt_rp"].to(outputs.dtype)

        output_rp = torch.cat([outputs[idx][-1].unsqueeze(0) for idx, position in enumerate(HB_list)], dim=0)

        # calculate loss
        criterion_loss =nn.MSELoss()
        loss = criterion_loss(output_rp, tgt_rp)

        # penalty loss
        diff_loss = 0
        for module_name, module in model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                for adapter_key in module.lora_A.keys():
                    A_linear = module.lora_A[adapter_key]
                    B_linear = module.lora_B[adapter_key]
                    A = A_linear.weight
                    B = B_linear.weight
                    alpha = getattr(module, "scaling", 1.0)[adapter_key]
                    delta = alpha * (B @ A)
                    diff_loss += (delta ** 2).sum()
        logging.info(f'train loss: {round((loss + diff_loss).item(), 2)}, diff loss: {round((diff_loss).item(), 6)}, cuda: {str(loss.device)}')
        loss = loss + diff_loss
        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    # set LoRA
    target_modules = []
    for name, param in model.named_parameters():
        if "thinker.model" in name:
            if "q_proj" in name or "k_proj" in name or "v_proj" in name or "o_proj" in name or "gate_proj" in name or "up_proj" in name or "down_proj" in name:
                if ".".join(name.split(".")[0:-1]) not in target_modules:
                    target_modules.append(".".join(name.split(".")[0:-1]))


    lora_config = LoraConfig(
        r=16,
        lora_alpha=4,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="./omni_top51",
        num_train_epochs=10,
        do_eval=False,
        eval_strategy="no",
        save_strategy="epoch",
        logging_steps=866,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        deepspeed="./ds_config.json",
        save_total_limit=30,
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
        adam_beta1=0.9,
        adam_beta2=0.98,
        weight_decay=0.001,
        learning_rate=5e-5,
        gradient_accumulation_steps=2,
    )


    reset_trainer = Reset_Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collate_fn,
        compute_metrics=None
    )

    reset_trainer.train()