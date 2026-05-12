# Reproducibility

```bash
pip install -r requirements.txt
python solution.py
```
Or you can clone repo to Google Colab and run it with base colab enviroment


# Solution
In this work I researched Halucinations detection task, using Qwen2.5-0.5B hidden states and built lightweighted classifier 

## Experiments

### Layer and token ablation
As was discovered in previous papers (Orgad et al., 2025), the attributes of truthfulness are actually encoded in the internal representations of the model, and this information may be distributed unevenly across layers. Most infromation is encoded in the middle and last lyers. So I tried different setups, but the most effective ones is averging by 21-24 layers. 
At the same time, explicitly adding a final layer on top of the already averaged late layers did not yield a consistent improvement using the whole transformer's hidden states. Similarly, using the whole transformer's hidden states instead of averaging did not provide stable gains.
Also I tried using different last token setups for features such as only 1, last 8, 16, etc. And it was founded that the most effective setup is adding long tokens window, that's why I used (1, 8, 16, 32, 64, 128, 256, 512) as a final setup.

### Different data aggregation
There  were tested different types of pooling: mean pooling, max pooling, min pooling, std pooling, and range pooling. The best results in the final version were achieved using range pooling. This computes the difference between the maximum and minimum activations for tokens within the window, thereby reflecting the variance/instability of latent sentences at the end of the response. Also I used PCA with 80 components + StandardScaler.

### Model
I've tested a lot of models(Logistic Regression, Lasso/Ridge, 2,3 layer MLP, k-NN and SVM) and I figured out that the best one is SVM with linear kernel. It is really lightweighted classifier, and it showed better performance than MLP, because MLP can potentially learn a more complex boundary, but is more likely to overfit on a small dataset. 

### Activation tensor as layers × tokens structure
An additional motivation was that hidden states can be viewed not as individual independent points, but as a structured layer-by-token object. In the ACT-ViT paper, the authors propose treating the activation tensor as an image along the layers and tokens axes and demonstrate that such a structure may be more useful than isolated layer-token probes (Bar-Shalom et al., 2025).
My solution does not use a full-fledged vision model on top of the activation tensor, but uses a similar idea: features are built simultaneously along the token axis via range pooling and along the layer axis via a trajectory block.

### Spectral features
Inspired by INSIDE (Chen et al., 2024), I tried concatenating PCA-reduced hidden features with spectral features computed from Gram/covariance matrices of hidden activations. The tested spectral features included top eigenvalues, sum of eigenvalues, logdet, effective rank, participation ratio, condition number and spectral entropy.

This did not improve the final score. In my experiments, spectral features were less useful than direct hidden range pooling and layer trajectory features. A likely reason is that spectral statistics are too compressed: they describe the global shape of the activation cloud, but lose information about which tokens, layers and hidden directions carry the hallucination signal.

### Trajectories and LSD++ Trajectories
And the final hypothesis is that hallucination may manifest itself not only in a specific hidden state, but also in how the answer representation changes across layers. Idea was really similar to (Hameed Mir et al., 2025).
In my experiments the idea was checked in LSD++ trajectory-features. For different diapasones I calculated cues of change in answer representations between layers: norms, differences, cosine similarity between adjacent layers, slope, drift, and stability statistics.
Adding cross-layer trajectory features improves the hidden-state SVM baseline. The best result is obtained when trajectory features are computed over layers 10–18, suggesting that middle-to-late layer dynamics carry hallucination-relevant information beyond the final hidden representation. Full-layer trajectories from 1–24 increase recall but appear less stable, likely because early layers introduce low-level representational noise.


### Prompt-Answer Trajectories
I checked sepately another my hypothesis, that it is important not only the hidden state, but also how changes geometric distance between prompt and answer through transformer layers. For each layer, pooled representations of the prompt and answer were compared: cosine similarity, L1/L2/L∞ distances, dot product, prompt/answer/delta norms, norm ratio, and other scalar features. The trajectory of these values ​​was then calculated across layers. Result was quite good but did not improve solution pipeline significantly


## Final Method

The final solution is the best-accuracy configuration selected from the local
experiments:

- Hidden stack: mean transformer layers 21-24, then range pooling over tail
  windows `1, 8, 16, 32, 64, 128, 256, 512`.
- Trajectory block: `E_range_10_18`, layer trajectory range 10-18, 34 scalar
  trajectory features.
- Probe: train-only `StandardScaler` + PCA with 80 components on hidden
  features, separate `StandardScaler` on trajectory features, concatenation,
  linear SVM with `C=0.1`.

## Experiments

Experiment scripts are kept in `experiments/`.  Older cluster launcher scripts
and generated caches/results were removed from the final repository state.


## References

- Hadas Orgad, Michael Toker, Zorik Gekhman, Roi Reichart, Idan Szpektor, Hadas Kotek, and Yonatan Belinkov. 2025. *LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations*. ICLR 2025. https://belinkov.com/assets/pdf/iclr2025-know.pdf

- Chao Chen, Kai Liu, Ze Chen, Yi Gu, Yue Wu, Mingyuan Tao, Zhihang Fu, and Jieping Ye. 2024. *INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection*. ICLR 2024. https://openreview.net/forum?id=Zj12nzlQbz

- Zhenliang Zhang, Xinyu Hu, Huixuan Zhang, Junzhe Zhang, and Xiaojun Wan. 2025. *ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs*. Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, pages 17986–18002. https://aclanthology.org/2025.acl-long.880/

- Amir Hameed Mir. 2025. *The Geometry of Truth: Layer-wise Semantic Dynamics for Hallucination Detection in Large Language Models*. arXiv:2510.04933. https://arxiv.org/abs/2510.04933

- Guy Bar-Shalom, Fabrizio Frasca, Yaniv Galron, Yftah Ziser, and Haggai Maron. 2025. *Beyond Token Probes: Hallucination Detection via Activation Tensors with ACT-ViT*. NeurIPS 2025. https://openreview.net/forum?id=aJ7AdfOfij

- An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2024. *Qwen2.5 Technical Report*. arXiv:2412.15115. https://arxiv.org/abs/2412.15115
