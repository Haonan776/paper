# 论文阅读记录

**2025.09.29——SAM基础模型**
1. **Segment Anything**[[paper]](https://arxiv.org/pdf/2304.02643)

**2025.10.03---遥感领域的开放词汇分割**
1. [NeurIPS'25]**OpenWorldSAM： Extending SAM2 for Universal Image Segmentation with Language Prompts**[[paper]](https://arxiv.org/abs/2507.05427)>扩展SAM2以实现基于语言提示的通用图像分割
>笔记：1）图像和文本输入到encoder中，再输入解码器中
      >2）引入位置平局打破机制，执行多实例分割
      >3）软提示，文本查询和图像特征提升定位精度
2. [NeurIPS'25]**InstructSAM: A Training-Free Framework for Instruction-Oriented Remote Sensing Object Recognition**[[paper]](https://arxiv.org/abs/2505.15818)>面向指令的遥感目标识别的免训练框架
>笔记：1）使用LVLM来解释用户指令并预测目标类别和计数
      >2）使用SAM2来生成掩码
      >3）使用CLIP来计算生成的目标类别和mask proposal之间的语义相似性
      > 4）目标检测和分割表述为一个mask-label匹配问题

>疑问：1）如果 LVLM 在指令理解阶段出现错误，比如误报目标数量或识别错类别，InstructSAM 框架如何处理这些错误，以及这会对最终结果产生多大影响？ 会不会考虑人工反馈或者交互式来修正这些问题
      >2）BIP换成图算法是否可行

**2025.10.12---遥感领域的开放词汇分割**
1. [2025 arXiv] **Exploring Efffcient Open-Vocabulary Segmentation in the Remote Sensing**[[paper]]()
