---
title: 音频音乐与计算机的交融-音频音乐技术
date: 2025-07-26 10:00:00 +0800
categories: [AI + MUSIC, Basic knowledge]
tags: [ai music]     # TAG names should always be lowercase
description: MOOC 李伟
---

- [课程链接](https://www.icourse163.org/learn/FUDAN-1462119161)

## 1 音乐科技、音乐人工智能与计算机听觉概述

### 1.1 音频音乐技术概述

- JNMR, ICMC, CMJ, ISMIR, CSMCW, CSMT

- 概述历史，可以参考[该知乎回答](https://www.zhihu.com/question/314142299/answer/612302579)

### 1.2 理解数字音乐-音乐信息检索技术综述

- [CSMT 组委会提供：期刊会议参考](https://mp.weixin.qq.com/s/m1dh9MPs7n0nPCUsgnfB8A )

### 1.3 理解数字声音-基于一般音频/环境声的计算机听觉综述

- 计算机听觉 CA computer audition

    - 音频事件检测 audio event detection

    - 音频场景识别 audio scenes identification

## 2 声学基础

### 2.1 心理声学

- 研究客观声音和主观听觉之间的关系

- 频率的感知

    - 客观频率 - 主观音高：以 500hz 为分界点，低于 500hz 线性，高于 500hz 非线性 -> Mel 频率（感知加倍减半实验）

- 强度的感知

    - 声压: Pa(帕)即 $ 1\mathrm{N}/\mathrm{m}^2 $ ，一个标准大气压是 $ 10^5\mathrm{Pa} $ 
    
    - 声压级: dB(分贝)计算公式为 $ 20\lg(\frac{P}{P_0}) $ ，其中基准声压 $ \mathrm{P}_0 $ 为 20 微帕

    - 绝对听阈：刚刚能察觉声音存在的声压级大小，随频率而变；人类对 500hz - 7000hz 最敏感；

    - 响度：40dB 以下，响度加倍，声压级需要提高约 20%；40dB 以上，响度要加倍，声压级需要提高 10dB。

$$
\begin{equation}
    声压级 = 20\lg(\frac{P}{P_0})
    \label{eq:db}
\end{equation}
$$

- 掩蔽效应：强声掩蔽弱声的现象：

    - 频率相近时容易掩蔽；
    
    - 不对称：低频掩蔽高频容易，高频掩蔽低频难；

### 2.2 音乐声学

- 偏乐理

    - 国际音高标准 小字一组 a1 = 440hz

    - ADSR attack Decay Sustain Release

## 3 音乐心理学

### 3.1 音乐要素、句法和情绪意义的加工

- 声音要素、音乐构成要素、音乐句法、音乐情绪与意义

- 声音要素：音高、音长、音强、音色

- 音乐构成要素：旋律、和声、节奏

- 音乐句法：和声句法、节奏句法

- 音乐意义：隐含意图

- 音乐情绪：通过声音要素和音乐内在结构传达感情；

    - 对声学线索传达的音乐情绪加工具有跨文化的普遍性；

    - 如果听着不熟悉音乐的调式体系，无法对由音乐内在结构规则传递的情绪进行加工

### 3.2 音乐记忆以及音乐在教育和临床的应用

- 虽然人只能在一个瞬间听到音乐的一个片段，但音乐会作为整体被人脑加工

- 教育、临床领域

## 4 音频特征及音频信号处理

### 4.1 音频特征及音频信号处理

- 音频特征的分类

    - 直接输出 vs 统计值

    - 瞬态 vs 全局

    - 抽象程度的高低（波形 —> 乐谱元素 -> 音乐流派或情绪）

    - 提取过程中的差异

- 常见音频特征：

    - 能量特征：均方根能量 Root-Mean-Square Energy

    - 时域特征：起音时间 Attack Time, 过零率 Zero-Crossing Rate, 自相关 Autocorrelation（测量基频）

    - 频域特征：频谱质心 Spectral Centroid（能量的集中点，描述明亮度），频谱平坦度 Spectral Flatness（信号与噪声之间相似度的参数，越平坦，越可能是噪音），频谱通量 Spectral Flux（信号相邻帧之间的变化程度，计算音符起始点的特征）

    - 乐音特征：基音频率 Fundamental Frequency，失谐度 Inharmonicity

    - 感知特征：响度 Loudness，尖锐度 Sharpness

- 特征提取工具：librosa、Essentia、Madmom、Vamp Plugins

- 音频信号处理：模拟信号处理（卷积、FFT 等）、数字信号处理

    - 采样点：44Khz -> 每一秒内有 44K 个采样点

    - STFT 短时离散傅里叶变换，可将时域数字信号变换为时频谱（横轴时间，纵轴频率）

### 4.2 音频传输与压缩

- 音频编码器：

    - 带宽：窄带、宽带、超宽带、全带；

    - 编码速率：反映了信号的压缩程度，单位 比特/秒 b/s，表示编码器每秒所使用到的比特数。分低/中/高速率。

    - 声道数：单声道/立体声/多声道编码；

    - 质量评价：主观评价，客观评价

- 信号冗余性：时域/频域/空间/听觉感知冗余。

    - 压缩就是为了去除冗余；

- 音频编码分类：

    - 波形编码：波形更一致，质量更高，但压缩率低；

    - 参数编码：建立产生模型，提取特征参数；编码速率低，质量比波形编码低；

    - 混合编码：混合两种；

### 4.3 虚拟现实音频

- VR 音频，反正有这么项技术； 

## 5 音高估计、主旋律提取与自动音乐记谱、音乐节奏分析

### 5.1 音高估计，主旋律提取，自动音乐记谱

- 音高是主观的，但基频是客观的

- 方法分类：

    - 信号表示域：时域法、频域法；

    - 待估计参数：参数法、非参数法；

- 自相关函数法：

    - 对于无限长离散时间信号 $ x(n) $，其自相关函数定义为 \eqref{eq:inf_corr}；

    - 若待处理的信号不是无限长周期信号，则须对 \eqref{eq:inf_corr} 进行改进。对于长度为 $ N $ 的有限长离散时间信号 $ x(n) $，其自相关函数定义为 \label{eq:finite_corr}；

    - 缺点：在所有周期整数倍，自相关函数均取峰值；对于复杂信号，很难确定基频；

$$
\begin{equation}
    r_x(m) = \sum_{n = -\infty}^{\infty} x(n)x(n + m)
    \label{eq:inf_corr}
\end{equation}
$$  

$$
\begin{equation}
    r_x(m) = \sum_{n = 0}^{N - 1 - m} x(n)x(n + m)
    \label{eq:finite_corr}
\end{equation}
$$  

- YIN 音高估计法：

    - 构建幅度差平方和替代自相关函数中的乘积求和，定义为式 \eqref{eq:diff_square_sum}；
    
    - 由幅度差平方和求累积平均归一化差分函数，定义为式 \eqref{eq:normalized_diff}；

    - 优势：寻找零值（或某个接近 0 的值），比寻找某个未知最大值，要容易的多；

$$
\begin{equation}
    d_x(m)=\sum_{n = 1}^{N} [x(n) - x(n + m)]^2
    \label{eq:diff_square_sum}
\end{equation}
$$  

$$
\begin{equation}
    d'_x(m)=
    \begin{cases} 
    1, & m = 0 \\
    d_x(m) \big/ \left[ \frac{1}{m} \sum_{j = 1}^{m} d_x(m) \right], & \text{其他}
    \end{cases}
    \label{eq:normalized_diff}
\end{equation}
$$

- 主旋律提取：基于显著度/源分离/机器学习的方法： 

    - 重点：音高轮廓特征法：

    - 正弦提取 -> 显著度计算 -> 音高轮廓生成 -> 音高轮廓选择；

- 自动音乐记谱：

    - 多音高估计（多基频分析）；

    - 谐波自适应隐藏成分分析法：把音乐信号分解为谐波信号和噪声信号的和

### 5.2 音乐节奏分析

