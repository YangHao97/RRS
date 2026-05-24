from io import BytesIO
from urllib.request import urlopen
from datasets import load_dataset, concatenate_datasets
import librosa
from tqdm import tqdm
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from transformers.generation import GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import torch.nn as nn
import torch
import numpy as np
from qwen_omni_utils import process_mm_info
torch.manual_seed(1234)
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto", device_map="cuda").eval()
model.disable_talker()
model.generation_config.temperature = 1.0
model.generation_config.max_new_tokens = 2048
model.generation_config.top_p = 1.0
model.generation_config.do_sample=True


################## calculate weight delta #######################
# load json data
harmful_dataset = load_dataset("json", data_files='./data/harmful.json', split="train")
benign_dataset = load_dataset("json", data_files='./data/benign.json', split="train")

# calculate average harmful represetation
# Note: take the representation of "WITH refusal prompt" version
sum_harmful = np.zeros((3584,))
for harmful in harmful_dataset:
    sum_harmful = sum_harmful + np.genfromtxt("./omni_audio_harmful_rp/with_prompt/" + harmful["path"] + ".csv", delimiter=",")
sum_harmful = sum_harmful/len(harmful_dataset)
print(sum_harmful)
print(len(harmful_dataset))

# calculate average benign represetation
sum_benign = np.zeros((3584,))
for benign in benign_dataset:
    sum_benign = sum_benign + np.genfromtxt("./omni_audio_benign_rp/" + benign["path"] + ".csv", delimiter=",")
sum_benign = sum_benign/len(benign_dataset)
print(sum_benign)
print(len(benign_dataset))

# calculate delta
diff = torch.tensor((sum_harmful - sum_benign), dtype=model.dtype).to("cuda")
print(diff)

# load the parameter of "I" in Qwen2.5-Omni
I_param = 0
for name, param in model.named_parameters():
    if name == "thinker.lm_head.weight":
        I_param = param[40].clone()

# calculate the contribution of delta
param_harmful = (I_param * diff).to(torch.float16).cpu().detach().numpy()
sorted_indices = np.argsort(param_harmful)


################# keep all weights > 0 (~ top 51%) #################
# contribute to a larger I40
for idx, value in enumerate(param_harmful):
    if value < 0:
        diff[idx] = 0

import csv
csvf = open("./omni_train_data/delta_weight_top51_rp_wp.csv", 'w', newline="")
writer = csv.writer(csvf)
writer.writerow(v.item() for v in diff)
####################################################################


################# keep top 12.5/25/50 ##############################
# 12.5%:3136 25%:2688 50%:1792 
# for i in sorted_indices[0:1792]:
#     diff[i] = 0

# import csv
# csvf = open("./omni_train_data/delta_weight_top50_rp_wp.csv", 'w', newline="")
# writer = csv.writer(csvf)
# writer.writerow(v.item() for v in diff)
####################################################################
