# 1.RNN-for-Joint-NLU

Pytorch implementation of "Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling" (https://arxiv.org/pdf/1609.01454.pdf)

<img src="https://github.com/DSKSD/RNN-for-Joint-NLU/raw/master/images/jointnlu0.png"/>

Intent prediction and slot filling are performed in two branches based on Encoder-Decoder model.

## 2.dataset (Atis)
You can get data from <a href="https://github.com/yvchen/JointSLU/tree/master/data ">here</a>
atis.test.w-intent.iob
atis-2.dev.w-intent.iob
atis-2.train.w-intent.iob
sample.iob

## 3.Requirements

* `Pytorch 0.2`

## 4.Train
`python3 train.py --data_path 'your data path e.g. ./data/atis-2.train.w-intent.iob'`
超参数：
 result_max_length =50
 embedding_size =32
 hidden_size =32
 num_layers =1
 step_size =10
 batch_size =8
 learning_rate =0.01


##5.Test
获取模型
训练集测试
输出预测
统计数据


## 6.Result

<img src="https://github.com/DSKSD/RNN-for-Joint-NLU/raw/master/images/jointnlu1.png"/>
<img src="https://github.com/DSKSD/RNN-for-Joint-NLU/raw/master/images/jointnlu2.png"/>
<img src="https://github.com/DSKSD/RNN-for-Joint-NLU/raw/master/images/jointnlu3.png"/>