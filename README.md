# paper
记录一下关于研究方向及感兴趣的文献阅读情况
**2025.0930——SAM基础模型及基于SAM的2个up版本**
1、**Segment Anything**[[paper]](https://arxiv.org/pdf/2304.02643)
2、[NeurIPS'25]**OpenWorldSAM： Extending SAM2 for Universal Image Segmentation with Language Prompts**[[paper]](https://arxiv.org/abs/2507.05427)>扩展SAM2以实现基于语言提示的通用图像分割
>笔记：1）图像和文本输入到encoder中，再输入解码器中 2）引入位置平局打破机制，执行多实例分割 3）软提示，文本查询和图像特征提升定位精度
3、[NeurIPS'25]**InstructSAM: A Training-Free Framework for Instruction-Oriented Remote Sensing Object Recognition**[[paper]](https://arxiv.org/abs/2505.15818)>面向指令的遥感目标识别的免训练框架
>笔记：1）使用LVLM来解释用户指令并预测目标类别和计数 2）使用SAM2来生成掩码 3）使用CLIP来计算生成的目标类别和mask proposal之间的语义相似性 4）目标检测和分割表述为一个mask-label匹配问题
>疑问：1）如果 LVLM 在指令理解阶段出现错误，比如误报目标数量或识别错类别，InstructSAM 框架如何处理这些错误，以及这会对最终结果产生多大影响？ 会不会考虑人工反馈或者交互式来修正这些问题

**老师提供暂存：** 
1. [2025CVPR] **POT: Prototypical Optimal Transport for Weakly Supervised Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_POT_Prototypical_Optimal_Transport_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2025_paper.pdf)
> 弱监督病理图像的语义分割
2. [2025CVPR] **Multi-Label Prototype Visual Spatial Search for Weakly Supervised Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Duan_Multi-Label_Prototype_Visual_Spatial_Search_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2025_paper.pdf)
> 弱监督病理图像的语义分割
3. [2024 arXiv]**Toward Modality Gap: Vision Prototype Learning for Weakly-supervised Semantic Segmentation with CLIP** [[paper]](https://arxiv.org/pdf/2412.19650)
4. [2025CVPR]**Prompt Categories Cluster for Weakly Supervised Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2025W/eLVM/papers/Wu_Prompt_Categories_Cluster_for_Weakly_Supervised_Semantic_Segmentation_CVPRW_2025_paper.pdf)


**多标签图像分类：** 
1. [Preprint]SPARC: Score Prompting and Adaptive Fusion for Zero-Shot Multi-Label Recognition in Vision-Language Models[[paper]](https://arxiv.org/pdf/2502.16911?)[[code]](https://github.com/kjmillerCURIS/SPARC)
2. [ICCV 2023]PatchCT: Aligning Patch Set and Label Set with Conditional Transport for Multi-Label Image Classification[[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_PatchCT_Aligning_Patch_Set_and_Label_Set_with_Conditional_Transport_ICCV_2023_paper.pdf)
3. [CVPR 2025]Classifier-guided CLIP Distillation for Unsupervised Multi-label Classification[[paper]](https://arxiv.org/pdf/2503.16873)[[code]](https://github.com/k0u-id/CCD)
4. [CVPR 2025]Recover and Match: Open-Vocabulary Multi-Label Recognition through Knowledge-Constrained Optimal Transport[[paper]](https://arxiv.org/pdf/2503.15337)[[code]](https://github.com/EricTan7/RAM)
5. [CVPR 2025]Correlative and Discriminative Label Grouping for Multi-Label Visual Prompt Tuning[[paper]](https://arxiv.org/pdf/2504.09990)


**Training-Free：**
目前的traning-free方法的核心其实就是算相似度矩阵
1. [2024 CVPR] **Clip-diy: Clip dense inference yields open-vocabulary semantic segmentation for-free** [[paper]](https://arxiv.org/pdf/2309.14289)
2. [2024 CVPR] **Training-Free Open-Vocabulary Segmentation with Offline Diffusion-Augmented Prototype Generation** [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10655445&tag=1) [[code]](https://github.com/aimagelab/freeda) 
3. [2024 ECCV] **Proxyclip: Proxy attention improves clip for open-vocabulary segmentation** [[paper]](https://arxiv.org/pdf/2408.04883) [[code]](https://github.com/mc-lan/ProxyCLIP?tab=readme-ov-file)
4. [2024 ECCV] **ClearCLIP: Decomposing CLIP Representations for Dense Vision-Language Inference** [[paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06346.pdf) [[code]](https://github.com/mc-lan/ClearCLIP)
5. [2024 ECCV] **Pay Attention to Your Neighbours: Training-Free Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2404.08181) [[code]](https://github.com/sinahmr/NACLIP)
6. [2024 ECCV] **SCLIP: Rethinking Self-Attention for Dense Vision-Language Inference** [[paper]](https://arxiv.org/pdf/2312.01597) [[code]](https://github.com/wangf3014/SCLIP)
7. [2024 ECCV] **Explore the Potential of CLIP for Training-Free Open Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2407.08268) [[code]](https://github.com/leaves162/CLIPtrase)
8. [2025 arXiv] **Self-Calibrated CLIP for Training-Free Open-Vocabulary Segmentation** [[paper]](https://arxiv.org/pdf/2411.15869) [[code]](https://github.com/SuleBai/SC-CLIP?tab=readme-ov-file)
9. [2025 CVPR] **LPOSS: Label Propagation Over Patches and Pixels for Open-vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2503.19777) [[code]](https://github.com/vladan-stojnic/LPOSS)
10. [2025 CVPR] **ResCLIP: Residual Attention for Training-free Dense Vision-language Inference** [[paper]](https://arxiv.org/pdf/2411.15851) [[code]](https://github.com/yvhangyang/ResCLIP?tab=readme-ov-file)
11. [2025 CVPR] **Distilling Spectral Graph for Object-Context Aware Open-Vocabulary Semantic Segmentation** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Kim_Distilling_Spectral_Graph_for_Object-Context_Aware_Open-Vocabulary_Semantic_Segmentation_CVPR_2025_paper.pdf) [[code]](https://github.com/MICV-yonsei/CASS)
12. [2025 CVPR] **Cheb-GR: Rethinking k-nearest neighbor search in Re-ranking for Person Re-identification** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_Cheb-GR_Rethinking_K-nearest_Neighbor_Search_in_Re-ranking_for_Person_Re-identification_CVPR_2025_paper.pdf) [[code]](https://github.com/Jinxi-Yang-WHU/Fast-GCR.git) 
> 笔记：本文提到的很多re-ranking的技术就是对直接计算的相似度矩阵进行更新，前面公式搞了一大堆，最后就是一个特征传播。
14. [2025 CVPR] **Realistic Test-Time Adaptation of Vision-Language Models** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zanella_Realistic_Test-Time_Adaptation_of_Vision-Language_Models_CVPR_2025_paper.pdf) [[code]](https://github.com/MaxZanella/StatA) [[original]](https://arxiv.org/pdf/2406.01837)
15. [2025 ICLR] **Efficient and Context-Aware Label Propagation for Zero-/Few-Shot Training-Free Adaptation of Vision-Language Model** [[paper]](https://arxiv.org/pdf/2412.18303) [[code]](https://github.com/Yushu-Li/ECALP?tab=readme-ov-file)
16. [2025 arXiv] **Test-Time Adaptation of Vision-Language Models for Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2505.21844v1) [[code]](https://github.com/dosowiechi/MLMP?tab=readme-ov-file)
