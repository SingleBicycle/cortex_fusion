# Hyperparameter Rationale

This note records why the default SSL configuration for `cortex_fusion` is:

- `input_mode=main5`
- `mask_strategy=hybrid`
- `mask_ratio=0.35`
- `hidden_dim=32`
- `dims=(32, 64, 128, 256, 128, 64, 32)`
- reconstruction-first SSL with weighted masked MSE

The intent is not to claim that one paper directly prescribes this exact setting. The defaults are an informed combination of masked graph modeling literature, masked autoencoder literature, and the geometry of cortical surface data.

## Reference Signals

### 1. Reconstruction-first objective

- `GraphMAE` shows that masked feature reconstruction is a strong self-supervised objective for graphs and avoids relying on labels or handcrafted positives/negatives.
- `GraphMAE2` strengthens that result by improving the decoder/training recipe rather than replacing masked reconstruction as the core pretext task.
- `MAE` and `SimMIM` both support the broader masked-autoencoding pattern: corrupt the input, reconstruct missing content, then use the encoder representation downstream.

Why that maps here:

- Our downstream goal is a subject embedding, but the available vertex-level signals are geometric and morphometric rather than text-like tokens.
- Reconstruction is a natural fit because the model can learn local cortical regularities without needing class supervision.
- The optional CE head stays available for ablations, but reconstruction remains the default optimization target.

### 2. Mask ratio

- `MAE` popularized relatively aggressive masking in highly redundant spatial data.
- `GraphMAE` also uses meaningful corruption levels rather than tiny perturbations, because the masked prediction task must be hard enough to avoid trivial copying.
- `GraphMAE2` continues in that direction: masked graph learning improves when the corruption is strong enough and the decoder/training setup can support it.

Why `0.35` here:

- Cortical surface features are redundant locally, but less redundant than raw image patches.
- A ratio below `0.25` is likely too easy for smooth cortical geometry.
- A ratio above `0.45` risks making the task unstable for the low-dimensional morph channels, especially for `ablation2`.
- `0.35` is a middle point that keeps the task meaningfully masked without erasing too much local context.

### 3. Patch masking vs random masking

- Most masked graph papers start from random masking.
- Structured masking is motivated by the fact that graph encoders aggregate over neighborhoods, so masking contiguous local regions can create a harder and more topology-aware reconstruction task.
- In vision MIM, the same idea appears as spatially coherent masking rather than independent pixel/token dropout.

Why `hybrid` here:

- Cortical surface graphs are smooth manifolds with strong local spatial correlation.
- Pure random masking is useful for feature coverage, but it under-stresses local completion.
- Pure patch masking can over-focus on contiguous holes and reduce feature diversity.
- `hybrid` keeps both behaviors: local holes that test neighborhood reasoning, plus random points that prevent the model from overfitting to one corruption pattern.

This is partly an inference from the literature rather than a direct one-to-one prescription. The cortical manifold structure makes local patch corruption more plausible here than in generic graph benchmarks.

### 4. Hidden dimension choice

- `GraphMAE` and `MAE` both support a common trend in SSL: once the corruption task is meaningful, too-small encoders become the bottleneck.
- At the same time, cortical surface training in this repo is subject-level, graph-based, and memory-bound by high vertex counts rather than huge batch sizes.

Why `hidden_dim=32` and a symmetric ADGCN width schedule:

- `32` is large enough to fuse geometry and morphometry into a non-trivial latent before the ADGCN backbone.
- It is still small enough to keep memory reasonable for two hemispheres, reconstruction heads, and masked-vertex evaluation.
- The `(32, 64, 128, 256, 128, 64, 32)` schedule preserves the existing ADGCN shape while expanding capacity relative to the older narrow default.

### 5. Weighted masked MSE

- `MAE` and `SimMIM` show that simple reconstruction losses can work well when the prediction target is normalized.
- Here the target channels are heterogeneous: xyz-like geometry and morphometry live in different semantic groups even after normalization.

Why weighted masked MSE here:

- We only score masked vertices, which keeps the loss aligned with the corruption task.
- We average per-dimension masked errors before weighting, which keeps `baseline8`, `main5`, and `ablation2` comparable.
- Default weights are currently uniform because all channels are normalized, but the weighting hook is important for later adjustments if geometry dominates numerically or if morph channels need emphasis.

This weighting choice is an engineering adaptation for multimodal cortical features, not a claim that one cited paper specifically recommends this exact loss.

## Why The Default Config Fits Cortical Surfaces

`main5` is the recommended default because it keeps one geometric frame (`mid_xyz`) plus the two morph channels, which is usually enough to capture cortical folding and thickness variation without doubling the coordinate channels as in `baseline8`.

`hybrid` masking is the best initial choice because cortical surfaces have strong local smoothness. Local holes force actual neighborhood-based completion, while the random component keeps the task from becoming too spatially narrow.

`mask_ratio=0.35` is a practical compromise for this domain:

- hard enough to require inference
- not so high that the 2-channel morph-only ablation becomes unstable
- still compatible with subject-level graph training on two hemispheres

The reconstruction-first objective is preferred because the embedding is meant to be reusable. A supervised head can always be added later, but a supervised-centric default would bias the representation toward the currently available label space.

## Initial Ablation Plan

| Factor | Settings | Primary question |
| --- | --- | --- |
| `input_mode` | `baseline8`, `main5`, `ablation2` | How much full pial/white geometry helps beyond mid-surface geometry, and how much performance survives in morph-only mode |
| `mask_strategy` | `random`, `patch`, `hybrid` | Whether local contiguous masking improves cortical completion relative to iid random masking |
| `mask_ratio` | `0.25`, `0.35`, `0.45` | Where the corruption level best balances non-trivial reconstruction with stable optimization |

Suggested run order:

1. Fix `input_mode=main5`, compare masking strategy at `mask_ratio=0.35`.
2. Fix the best masking strategy, sweep `mask_ratio`.
3. Fix the best corruption setup, compare `baseline8` vs `main5` vs `ablation2`.

## References

- Hou et al. `GraphMAE: Self-Supervised Masked Graph Autoencoders are Scalable Learners.` KDD 2022. https://arxiv.org/abs/2205.10803
- Hou et al. `GraphMAE2: A Decoding-Enhanced Masked Self-Supervised Graph Learner.` 2023. https://arxiv.org/abs/2304.04779
- Hu et al. `Strategies for Pre-training Graph Neural Networks.` ICLR 2020. https://arxiv.org/abs/1905.12265
- He et al. `Masked Autoencoders Are Scalable Vision Learners.` CVPR 2022. https://arxiv.org/abs/2111.06377
- Xie et al. `SimMIM: A Simple Framework for Masked Image Modeling.` CVPR 2022. https://arxiv.org/abs/2111.09886
