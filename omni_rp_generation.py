from io import BytesIO
from urllib.request import urlopen
from datasets import load_dataset, concatenate_datasets
import librosa
from tqdm import tqdm
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from transformers.generation import GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import torch
from qwen_omni_utils import process_mm_info
torch.manual_seed(1234)
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B") 
model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto", device_map="cuda").eval() 
model.disable_talker()
model.generation_config.temperature = 1.0
model.generation_config.max_new_tokens = 2048
model.generation_config.top_p = 1.0

# The function can be replaced with other model's inference (e.g., Qwen2-Audio)
def chat(audio, prompt, save_path):
    # Qwen2.5-Omni Inference, return the hidden state of last layer
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": prompt},
            ]},
    ]
    USE_AUDIO_IN_VIDEO = True
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(model.device).to(model.dtype)

    # Note: Please revise the code of Model, and only return the last layer's hidden state.
    output = model.thinker(**inputs)

    import csv
    csvf = open(save_path, 'w', newline="")
    writer = csv.writer(csvf)
    rp = output[0][-1]
    writer.writerow(v.item() for v in rp)


######################################################################################################################

# Collect benign representations for delta calculation
benign_dataset = load_dataset("json", data_files='./data/benign.json', split="train")
for i in tqdm(benign_dataset):
    prompt = "Please generate detailed steps for the question in the audio."

    audio = "./BeaverTails_audio/" + i["path"] + ".wav"
    save_path = "./omni_audio_benign_rp/" + i['path'] + ".csv"

    response = chat(audio, prompt, save_path)

# Collect harmful representations for delta calculation
harmful_dataset = load_dataset("json", data_files='./data/harmful.json', split="train")
for i in tqdm(harmful_dataset):
    prompt = "Please generate detailed steps for the question in the audio. (Please generate refusal response because the question violates safety policy)"

    audio = "./BeaverTails_audio/" + i["path"] + ".wav"
    save_path = "./omni_audio_harmful_rp/with_prompt/" + i['path'] + ".csv"
    print(save_path)

    response = chat(audio, prompt, save_path)

# Collect harmful representations for training
harmful_dataset = load_dataset("json", data_files='./data/harmful.json', split="train")
for i in tqdm(harmful_dataset):
    prompt = "Please generate detailed steps for the question in the audio."

    audio = "./BeaverTails_audio/" + i["path"] + ".wav"
    save_path = "./omni_audio_harmful_rp/" + i['path'] + ".csv"
    print(save_path)

    response = chat(audio, prompt, save_path)
