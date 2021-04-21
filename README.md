# Deft Def Extraction
Deft Corpus Definition Extraction, SemEval2020 Task 6 

## Prepare environment
```
conda create -n deft python=3.6
conda activate deft
conda install pytorch -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
pip install -r requirements.txt 
```

## Sub Task 1

Sentence Classification, classify the sentences into 1 (contain a definition) or 0 (does not contain a definition)

RoBERTa-base, running script `run_task1.sh`

### Results

#### Results on dev
|                       | eval_accuracy | eval_f1 | eval_loss | eval_precision | eval_recall |
| --------------------- | ---- | ---- | ---- | ---- | ---- |
| RoBERTa-base | 0.8899 | 0.8367 | 0.2898 | 0.8092 | 0.8662 |
| RoBERTa-base-rsh | 0.8876 | 0.8262 | 0.2906 | 0.8321 | 0.8204 |
| qa-RoBERTa-base | 0.8830 | 0.8247 | 0.2987 | 0.8054 | 0.8451 |
| qa-suffix-RoBERTa-base | 0.8819 | 0.8209 | 0.3024 | 0.811 | 0.831 |


#### Results on test
|                       | eval_accuracy | eval_f1 | eval_loss | eval_precision | eval_recall |
| --------------------- | ---- | ---- | ---- | ---- | ---- |
| RoBERTa-base |  |  |  |  |  |
| RoBERTa-base-rsh | 0.8545 | 0.7748 | 0.3556 | 0.779 | 0.7706 |
| qa-RoBERTa-base | 0.8498 | 0.7749 | 0.3482 | 0.7551 | 0.7957 |
| qa-suffix-RoBERTa-base | 0.8685 | 0.7957 | 0.338 | 0.8029 | 0.7885 |