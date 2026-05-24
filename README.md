# RRS for Safety Tuning in Large Audio Language Models </h1>
*The code is for EMNLP 2025 paper: [Reshaping Representation Space to Balance the Safety and Over-rejection in Large Audio Language Models](https://aclanthology.org/2025.emnlp-main.510.pdf)*


## Data Preparation
Generate representations for training and delta calculation
```
python omni_rp_generation.py
```

Calculate effective delta contributing to a larger logit value of token "I"
```
python omni_delta_calculation.py
```
Note: original code of target model should be revised based on instruction.

## Fine-tuning
The code in this repo is a minimal implementation for fine-tuning Qwen2.5-Omni, please follow its official instruction to deploy. Our code is based on:
```
pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview
```

```
python omni_train.py
```
## Dataset
The dataset and ckpt in our paper can be find here: [dataset](https://drive.google.com/file/d/18QhgOfIkc_ubDO7WJ-VEBz0x-EuuUug_/view?usp=drive_web) and [ckpt](https://drive.google.com/file/d/15fa4FlAKd-KgbWVen-8qr3CWFNbdTjaC/view?usp=sharing)

Audio dataset of this work is generated from [BeaverTails](https://papers.nips.cc/paper_files/paper/2023/file/4dbb61cb68671edc4ca3712d70083b9f-Paper-Datasets_and_Benchmarks.pdf).

## Citation
```
@inproceedings{yang2025reshaping,
  title={Reshaping Representation Space to Balance the Safety and Over-rejection in Large Audio Language Models},
  author={Yang, Hao and Qu, Lizhen and Shareghi, Ehsan and Haffari, Gholamreza},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={10078--10090},
  year={2025}
}
```
