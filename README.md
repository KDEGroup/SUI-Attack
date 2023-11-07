# Single-User Injection for Invisible Shilling Attack against Recommender Systems
A PyTorch implementation of paper:

[Single-User Injection for Invisible Shilling Attack against Recommender Systems](https://arxiv.org/abs/2308.10467), Chengzhi Huang, Hui Li , CIKM '2023

## Requirements
```
dgl==0.4.3
numpy>=1.15
pandas>=0.19
scipy>=0.18
torch>=1.3
higher
```
## Floder
+ `models` contains influence module, recommender and baseline attacker
+ `config` contains its super parameters


## Data
The dataset used in experiments or other scripts can be download from [Google Drive](https://drive.google.com/drive/folders/1a9-DeQ-v0IDJG6C69QBmuKjMyi4sYVZL)

## How to run
```python
python main.py
```
other examples can be shown in jupyter notebook.

The parameters args.do_train and args.do_eval control whether the program is training the model or evaluating the model. More details can be found in the paper.