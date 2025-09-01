---
title: 跟李沐精读论文合集
date: 2025-08-20 10:00:00 +0800
categories: [deep learning, limu]
tags: [deep learning]     # TAG names should always be lowercase
description: 跟李沐精读论文合集
---

## GPT，GPT-2，GPT-3 论文精读

- [链接](https://www.bilibili.com/video/BV1AF411b7xQ)

- ![时间线](assets/post_img/2025-08-20-LiMu_read_paper/2025-08-20-LiMu_read_paper_01.png)

### GPT-1

- [论文链接](https://www.semanticscholar.org/paper/Improving-Language-Understanding-by-Generative-Radford-Narasimhan/cd18800a0fe0b668a1cc19f2ec95b5003d0a5035)

#### 背景介绍

- 预训练 + 微调在 CV 领域很成熟，因为有 ImageNet 这样子的大数据集；但当时在 NLP 领域没有这么大的数据集；且一张图片所含信息要比一个句子多得多，所以要用有监督学习整的话，需要千万级别的数据量，导致深度学习在 NLP 领域遇到了困难；

- 为什么 OpenAI 不用 RNN？因为他们认为 Transformer 比 RNN 更能捕捉结构化的信息；

#### 框架介绍

- 第一个目标函数（预训练 - 无监督学习）：给定模型和前 K 个词（窗口大小），预测第 k + 1 个词的概率；换句话说，最大化模型生成和我给定文本一样的输出；

    - 使用 Transformer 的 Decoder，有掩码的存在； 

    - 预测未来的状态比推理过去的状态难得多；这或许也是为什么 GPT 比 BERT 效果差；

- 第二个目标函数（微调 - 有监督学习）：给定预训练模型最后一层输出的序列，预测其标签；
    
    - 一个标准的分类目标函数；

- 第三个目标函数：引入一个超参数 λ，调整前两个目标函数的占比，同时优化；

- 可以借助特殊词元构造不同的分类任务；

- 被后面的 BERT 用更大的数据集和更大的模型击败；

### GPT-2

- [论文链接](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

#### 背景介绍

- 反击 BERT，一作和大老板没有变，必须坚持搞解码器；

- GPT-2，1.5B 参数，引入 zero-shot 概念，为卖点；

#### 方法

- zero-shot 就不得不要求，在微调下游任务时，不能引入模型没有见过的符号；

- 之前有 Common Crawl，但是信噪比很低；

- 不妨采取 Reddit，里面都是用户筛选过的优质文本；

- 结论：在很多任务上 SOTA 了，但在一些任务上还不够好，但是注意到性能随着模型变大而变好，那更大的模型表现如何？接下来看 GPT-3；