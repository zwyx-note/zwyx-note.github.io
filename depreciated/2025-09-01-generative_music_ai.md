---
title: Generative Music AI Course
date: 2025-04-09 17:00:00 +0800
categories: [AI + MUSIC, sound_of_ai]
tags: [ai music, sound_of_ai]     # TAG names should always be lowercase
description: sound_of_ai
---

- [课程链接](https://www.youtube.com/@ValerioVelardoTheSoundofAI/playlists)

## 1. Course Overview

- 水 + 广告时间

## 2. What's Generative Music

- 不存在一个 objective success metric，不存在一个 well-defined problem description

- 音乐生成是一个 ill-defined problem

## 3. History of Generative Music

- 5 eras of GM

### 1700 - 1956 Pre-computer era

- 人工算法，随机化，重组合

- Mozart Dice Game 1787：莫扎特准备好 176 个小节，掷色子，随机组合；

- Mode de valeurs et d'intensités (Messiaen, 1949)：对音乐的力度、演奏法、音高、时值这些要素进行了参数化处理，根据规则选择；

### 1957 - 2009 Academic era

- 分散但渐进的实验；主要关注古典音乐，数据集比较完善，symbolic 方向，难以产出完整乐谱，不关注音频质量；

- Illiac Suite (Hiller & Isaacson, 1957)：弦乐四重奏，四个乐章；

- Experiments in Musical Intelligence (Cope, 1981)：重组方法 - 组建语料库，提取特征，重组；

### 2010 - 2016 First startup wave

- 关注产品；整首乐曲的产出；高质量音乐数据集；机器学习办法；

- Melodrive (2016)：实时电子游戏音乐生成；

### 2017 - 2022 Big tech experiments

- 深度学习；大规模数据集和算力；没有商业化目标；

- AWS DeepComposer (Amazon, 2019)；

- Jukebox (OpenAI, 2020)：转折点；原始音频生成；深度学习；直接使用波形（或波形的其它形式）来训练；整个乐章生成；存在 lead vocal；

### 2023 - now Music AI type

- 商业化进行中；大规模科技；大规模数据集；startup 出现；数量多质量高类型丰富；

- MusicLM (Google, 2023)：text-to-music；

- MusicGen (Meta, 2023)：text2music；

## 4. Use Cases

- Classify GM systems - 目标；用户；自动化程度；生成手段；音乐表示形式；

- Text2Music；Singing voice cloning；Sound synthesis；

##  5. Ethical Implications

- 以音乐家为核心，征得同意？给予补偿？......

- 什么程度的音乐生成算是有版权的？

- 各国各地区法律制定有区别，倾向与判定有区别；

## 6. Symbolic Vs Audio Generation

- **A good music representation solves 50% of GM**

### Symbolic Music Generation

- MIDI，MusicXML，Piano-roll，tablature（吉他用谱 TAB），ABC notation，Kern；

- discipline connection：乐理、作曲、计算机音乐学；

- MuseNet (OpenAI, 2019) 经典之作；

- pros and cons：

    - 简单、准确、容易压缩、信息多、易捕捉长依赖关系、易训练；

    - 过度简化、音乐限制（混响、压缩、EQ 设计不了）、产出不是音频必然会限制（无法模仿合成器音色细节）；

- 适用：主要关注音乐的结构和作曲；适合用于有标记的西方音乐（古典、爵士等）

- 不适用：关注表演效果和产出效果，EDM 乐曲或持续低音/持续单音（drone）；

### Audio Music Generation

- wavform 波形图，spectrogram 频谱图，Mel-spectrogram 梅尔频谱图，audio embeddings

- discipline connection：数字信号处理、MIR、声音设计、music cognition 音乐认知

- Jukebox OpenAI，MusicLM Google，MusicGen Meta，RAVE Ircam

- pros and cons：

    - 表演细节丰富，音频输出；

    - 难，模型大，长期依赖关系捕捉难

## 7. Generative techniques

- 历史：选一个技术，解决一个问题

1. 基于规则的：从语料库中学习规则，人工编码规则很难；

    - CHORAL (Ebcioglu, 1990)：模仿巴赫风格，300+ 条规则；（音乐并非完全遵循规则，所以有很多灰色地带）

2. 优化办法：迭代式地优化一个拟合函数，其实是在模仿目标风格

    - 常见的有：genetic algorithm 遗传算法，particle swarm optimization 粒子群优化算法，simulated annealing 模拟退火算法；

    - GenJam (Biles, 1994)：jazz solos，基于交互式的遗传算法；由真人给出反馈（自注：有点像 RLHF...）

3. 复杂系统：算法简单，无需音乐知识，可以创造原始音乐素材

    - 常见的有：fractals 分形，cellular automata 元胞自动机

    - Conway's Game of Life：一个二维表格，每个表格在同一时刻有且仅有一个状态，不是 0 就是 1。01 的变化遵循特定规则；

    - CAMUS (Miranda, 1993)：二维元胞自动机，Determine pitch sequence (Conway's Game of Life)，Determine instrument (Griffeath's Crystalline Growths) - 每个细胞格被映射到指定音高；

4. 统计方法：从 corpus 中学习，模仿目标风格，捕捉长依赖关系有难度；

    - 常见的有：Markov chains 和 hidden markov models

    - Continuator (Pachet, 2002)：交互式，支持钢琴即兴演出，模仿风格，马尔科夫链。

5. 深度学习办法：神经网络，从大量数据中学习，音频生成/符号生成；可以学习长依赖关系；

    - 常见的有：RNN - DeepBach 2016，VAE - Jukebox 2020，Diffusion - Riffusion 2022，Transformers - MusicGen 2023

    - DeepBach：模仿巴赫风格，symbolic，

## 8. Limitations and Future Vision

- text2music：长依赖关系、音频质量、语义映射、创意控制少

- problems with DL models：音乐是 highly dimensional 的而神经网络学不到所有维度，模型没有乐理知识，需要大量数据，音乐整体一致性欠缺，模型是黑盒而人为难控制。

    - Yann LeCun：[A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf)

- 缓解 DL 问题的办法：hybrid systems，融合 DL 和符号音乐生成模型，我们需要让模型理解乐理以让模型更好地生成音乐

- 音乐表示：音频太复杂，符号太简单，但也没有一种完美的办法；

- 需要注意，用户界面（接口）也需要，以用户体验为核心，用户评估也很重要。

- 大公司正在垄断，模型要开源、研究员与音乐家多交流；

## 9. Generative Grammar for Music Generation

- grammar 是 rule-based：形式化来说 G = (N, T, P, S)

- N 是非终结符，T 是终结符，P 是产生式规则，S 是起始符号

- 最基本的规则只会生成确定的结果

```
T = {I, like, apples}
N = {S, PN, V, N}
S = S
P = {
    S -> PN V N
    PN -> I
    V -> like
    N -> apples
}
```

```
1. Start with S
2. Apply S -> PN V N
3. Apply PN -> I; I V N
4. Apply V -> like; I like N
5. Apply N -> apples; I like apples
```

- 我们加入概率学知识，就有了 probabilistic grammar

```
T = {I, you, like, love, apples}
N = {S, PN, V, N}
S = S
P = {
    S -> PN V N
    PN -> (0.5) I | (0.5) you
    V -> (0.7) like | (0.3) love
    N -> apples
}
```

```
1. Start with S
2. Apply S -> PN V N
3. Apply PN -> I | you (choose you); you V N
4. Apply V -> like | love (choose love); you love N
5. Apply N -> apples; you love apples
```

- 现在，将以上 grammar 迁移到音乐创作中

```
T = {C, D, E, F, G, A, B, Whole, Half, Quarter}
N = {Melody, Phrase, Pitch, Duration}
S = Melody
P = {
    Melody -> Phrase Phrase
    Phrase -> Pitch Duration | Pitch Pitch Duration
    Pitch -> C | D | E | F | G | A | B
    Duration -> Whole | Half | Quarter
}
```

```
1. Start with Melody
2. Apply Melody -> Phrase Phrase
3. Apply Phrase -> Pitch Duration | Pitch Pitch Duration; Pitch Duration Phrase
4. Apply Pitch -> C | D | E | F | G | A | B; C Duration Phrase
5. Apply Duration -> Whole | Half | Quarter; C Quarter Phrase
6. Apply Phrase -> Pitch Duration | Pitch Pitch Duration; C Quarter Pitch Pitch Duration
7. Apply Pitch -> C | D | E | F | G | A | B; C Quarter E Pitch Duration
8. Apply Pitch -> C | D | E | F | G | A | B; C Quarter E D Duration
9. Apply Duration -> Whole | Half | Quarter; C Quarter E D Whole
```

- 如何确定这些规则？

    - 人工提取，依赖乐理知识

    - 从数据集中学习

- Lindenmayer system - 一种 grammar

    - 每次迭代时，一次性应用所有的产生式规则

    - 可以生成分形，类似于细菌与植物的生长

- L-system 例子

- 给定规则：

```
A (alphabet) = {A, B}
S (axiom) = A
P = {
    A -> AB
    B -> A
}
```

- 每一次迭代，每个字母都同时应用规则

```
n=0: A
n=1: AB
n=2: ABA
n=3: ABAAB
n=4: ABAABABA
n=5: ABAABABAABAAB
```