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

- [Improving Language Understanding by Generative Pre-Training](https://www.semanticscholar.org/paper/Improving-Language-Understanding-by-Generative-Radford-Narasimhan/cd18800a0fe0b668a1cc19f2ec95b5003d0a5035)

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

- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

#### 背景介绍

- 反击 BERT，一作和大老板没有变，必须坚持搞解码器；

- GPT-2，1.5B 参数，引入 zero-shot 概念，为卖点；

#### 方法

- zero-shot 就不得不要求，在微调下游任务时，不能引入模型没有见过的符号；

- 之前有 Common Crawl，但是信噪比很低；

- 不妨采取 Reddit，里面都是用户筛选过的优质文本；

- 结论：在很多任务上 SOTA 了，但在一些任务上还不够好，但是注意到性能随着模型变大而变好，那更大的模型表现如何？接下来看 GPT-3；

### GPT-3

- **建议看原论文**

- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

#### 背景介绍

- 相比于复杂地更新梯度与参数，不如直接 few-shot 来干

#### 框架介绍与训练相关

- 175B 的 GPT-3 非常之大；计算复杂度与宽度成平方关系，与层数成线性关系；

- 同时在训练批次上，175B 的模型一个批次里的数据量达到了 3.2M，这极其巨大；

    - 小模型不能用这么大的，因为更容易过拟合

    - 为何大模型用大批次更不容易过拟合？大批次中信噪比更低。

    - 模型越大，训练学习率越小，从 6.0 * 10-4 降到了 0.6 * 10-4

- 对于预训练数据：首先下载清洗过的 common crawl 数据；

    1. 与 GPT-2 用的预训练数据集做比较，将 GPT-2 的数据集当作正例，common crawl 当作负例，我们就有了一个二分类分类器，然后就只保留 common crawl 里的正例数据；

    2. 使用 lsh 算法进行去重；

    3. 加入了之前的一些高质量数据集； 

    4. 对不同数据集，设置不同的采样率，以保证一个批次里，高质量的数据仍然占多数。

#### 评估相关

- 很多建议自己看

#### 模型局限性

- 长文本生成困难，结构上来说 gpt-3 只能向前看，没有涉及到别的领域，样本有效性较低，训练贵，可解释性差，歧视偏见现象严重