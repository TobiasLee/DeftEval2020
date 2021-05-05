# Deft Def Extraction
Deft Corpus Definition Extraction, SemEval2020 Task 6 

## Prepare environment
```
conda create -n deft python=3.6
conda activate deft
conda install pytorch -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
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
| qa-prefix-RoBERTa-base | 0.8830 | 0.8247 | 0.2987 | 0.8054 | 0.8451 |
| qa-prefix-w/o definition-RoBERTa-base | 0.8876 | 0.8304 | 0.2864 | 0.8163 | 0.8451 |
| qa-prefix-bullshit-RoBERTa-base | 0.8865 | 0.8272 | 0.2905 | 0.8201 | 0.8345 |
| qa-suffix-RoBERTa-base | 0.8819 | 0.8209 | 0.3024 | 0.811 | 0.831 |
| qa-suffix-w/o definition-RoBERTa-base | 0.8842 | 0.8243 | 0.2985 | 0.8144 | 0.8345 |
| qa-suffix-bullshit-RoBERTa-base | 0.8807 | 0.8219 | 0.2849 | 0.8 | 0.8451 |
| RoBERTa-large-rsh | 0.8888 | 0.8342 | 0.3001 | 0.8106 | 0.8592 |
| qa-suffix-RoBERTa-large | 0.8956 | 0.8401 | 0.2677 | 0.8386 | 0.8415 |


#### Results on test
|                       | eval_accuracy | eval_f1 | eval_loss | eval_precision | eval_recall |
| --------------------- | ---- | ---- | ---- | ---- | ---- |
| RoBERTa-base |  |  |  |  |  |
| RoBERTa-base-rsh | 0.8545 | 0.7748 | 0.3556 | 0.779 | 0.7706 |
| qa-prefix-RoBERTa-base | 0.8498 | 0.7749 | 0.3482 | 0.7551 | 0.7957 |
| qa-prefix-w/o definition-RoBERTa-base | 0.8591 | 0.7804 | 0.354 | 0.7904 | 0.7706 |
| qa-prefix-bullshit-RoBERTa-base | 0.8626 | 0.7839 | 0.3458 | 0.8015 | 0.767 |
| qa-suffix-RoBERTa-base | 0.8685 | 0.7957 | 0.338 | 0.8029 | 0.7885 |
| qa-suffix-w/o definition-RoBERTa-base | 0.865 | 0.7868 | 0.3637 | 0.8075 | 0.767 |
| qa-suffix-bullshit-RoBERTa-base | 0.8626 | 0.787 | 0.367 | 0.7927 | 0.7814 |
| RoBERTa-large-rsh | 0.8661 | 0.8007 | 0.3613 | 0.7752 | 0.828 |
| qa-suffix-RoBERTa-large | 0.8719 | 0.8022 | 0.3472 | 0.8051 | 0.7993 |