"""Fill result placeholders in 02_task2.md (run from report/)."""
from pathlib import Path

p = Path("02_task2.md")
s = p.read_text()

R = {}

R["BASELINE_TABLE"] = """Table 2.1 — Baseline CNN, layer by layer (input 224×224×3)

| Layer | Output shape | Parameters |
|---|---|---|
| Input `image` | (224, 224, 3) | 0 |
| Rescaling 1/255 | (224, 224, 3) | 0 |
| Conv2D 32 × 3×3 → BatchNorm → ReLU | (224, 224, 32) | 864 + 128 |
| MaxPooling 2×2 | (112, 112, 32) | 0 |
| Conv2D 64 × 3×3 → BatchNorm → ReLU | (112, 112, 64) | 18,432 + 256 |
| MaxPooling 2×2 | (56, 56, 64) | 0 |
| Conv2D 128 × 3×3 → BatchNorm → ReLU | (56, 56, 128) | 73,728 + 512 |
| MaxPooling 2×2 | (28, 28, 128) | 0 |
| Conv2D 256 × 3×3 → BatchNorm → ReLU | (28, 28, 256) | 294,912 + 1,024 |
| MaxPooling 2×2 | (14, 14, 256) | 0 |
| GlobalAveragePooling2D | (256) | 0 |
| Dropout 0.4 | (256) | 0 |
| Dense 1, sigmoid | (1) | 257 |
| **Total** | | **390,113** (389,153 trainable; 960 BatchNorm moving statistics) |"""

R["TRAINABLE_LAYERS"] = """Table 2.2 — What is trainable in each stage (`reports/runs/mnv2_finetune/trainable_layers.json`)

| Stage | Trainable layers | Trainable parameters | Learning rate |
|---|---|---|---|
| 1 — frozen backbone | `p_ai` (Dense head) only | 1,281 | 1e-3 |
| 2 — fine-tune last block | `block_16_expand`, `block_16_depthwise`, `block_16_project`, `Conv_1` (+ head); BatchNorm layers stay frozen | 880,321 | 1e-5 |

The full model has 2.26 M weights; in stage 1 only 0.06 % of them learn, in stage 2 about 39 % (the last block and the final 1×1 convolution hold most of MobileNetV2's parameters). The `unfreeze40` experiment instead unfreezes from `block_13_*` onward (13 backbone layers with weights)."""

R["TRANSFER_CURVES"] = """![Figure 2.2 — MobileNetV2, stage 1 (epochs 1–15, frozen backbone) and stage 2 (epochs 16–30, last block fine-tuned at lr 1e-5). Dotted line: fine-tuning starts; dashed: best epoch.](../reports/runs/mnv2_finetune/training_curves.png){width=100%}

![Figure 2.3 — Baseline CNN training curves (early-stopped after 17 epochs).](../reports/runs/cnn_baseline/training_curves.png){width=100%}

*Reading the curves.* In stage 1 validation loss falls steadily from 0.68 to 0.48 while only the head learns; it was still improving slowly at epoch 15. When the last block is unfrozen (epoch 16) training accuracy jumps from ~0.80 to ~0.85 and keeps climbing to 0.91, and validation loss continues down to **0.434** — fine-tuning helps, but the growing train/validation gap (0.91 vs 0.80 accuracy) shows the model starting to memorise 681 images, which is exactly why the learning rate is 100× smaller and early stopping/checkpointing watch validation loss. The CNN from scratch learns far more slowly: validation loss barely moves below 0.64, validation AUC stays around 0.68 and early stopping ends training after 17 epochs."""

R["BEST_EPOCHS"] = """Table 2.3 — Best epoch (lowest validation loss, marked green in the curves)

| Run | Epochs trained | Best epoch | Note |
|---|---|---|---|
| Baseline CNN | 17 of 30 | **11** | early stopping (patience 6) |
| MobileNetV2 fine-tuned (chosen) | 15 + 15 | **27** (epoch 12 of fine-tuning) | reached the epoch limit; checkpoint keeps epoch 27 |
| MobileNetV2 head only | 15 | 15 | still improving — more epochs would help a little |

`ReduceLROnPlateau` cut the CNN's learning rate from 1e-3 to 9e-5 by epoch 10, which is visible as the flattening of its curves."""

R["FINAL_METRICS"] = """Table 2.4 — Test-set metrics (147 images, evaluated once, threshold chosen on validation)

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR real→AI | FNR AI→real |
|---|---|---|---|---|---|---|---|---|
| Baseline CNN | 0.67 | 0.592 | 0.769 | 0.370 | 0.500 | 0.735 | 0.136 | 0.630 |
| **MobileNetV2 fine-tuned** | **0.60** | **0.796** | **0.881** | **0.728** | **0.797** | **0.868** | **0.121** | **0.272** |

Confusion counts (TN / FP / FN / TP): CNN 57 / 9 / 51 / 30; MobileNetV2 58 / 8 / 22 / 59."""

R["EVAL_DISCUSSION"] = """*Interpretation.* Transfer learning wins clearly on every metric: ROC-AUC 0.868 vs 0.735 and F1 0.797 vs 0.500. With only 681 training images, a network trained from scratch cannot learn good visual features; MobileNetV2 brings features learned from 1.2 M ImageNet images and only has to learn how to combine them.

The chosen threshold for MobileNetV2 is **0.60**, not 0.5: on validation, 0.5 would call about 25 % of real images AI-generated (Figure 2.4); 0.60 is the F1-best threshold that keeps that rate under the 15 % cap. On the test set it produced a **12.1 % false-positive rate** (8 of 66 real images flagged) while catching 73 % of AI images, with 88 % precision — when VisionGuard says "AI-generated" it is right 88 % of the time. For the CNN the same rule forces a high threshold (0.67) and the model then misses 63 % of AI images: a weak ranker cannot satisfy the constraint without giving up recall. The remaining false positives are why the result page carries a caveat."""

R["ERROR_TABLE"] = """Table 2.5 — Hypotheses for the 10 most confident mistakes (9 AI images called real, 1 real image called AI)

| # | True | P(AI) | What the image is | Hypothesis |
|---|---|---|---|---|
| 1 | AI | 0.08 | Vintage-print collage of a politician in a bathtub of flowers | Paper texture, film grain and muted print colours imitate a scanned photo; no glossy "AI look" |
| 2 | AI | 0.10 | Flat pop-art poster of lips | Graphic design with flat colour areas — similar to the posters/illustrations in the *Real* folder; no photographic texture to judge |
| 3 | AI | 0.18 | App-icon-like sailboat logo on black | Not a generated scene but a logo/screenshot (label noise); almost no texture |
| 4 | AI | 0.19 | Painted mountain landscape with a stock-site watermark | Watermark and illustration style resemble the stock images in the *Real* class |
| 5 | AI | 0.21 | 3D wireframe hand holding a framed picture | Clean CGI render on a plain background — the model has learned "busy, saturated detail ⇒ AI", which this lacks |
| 6 | AI | 0.22 | Photorealistic portrait of an elderly man | Highly photorealistic generator output; artefacts (skin, hair) disappear when 1024 px is downsampled to 224 px |
| 7 | AI | 0.22 | Photorealistic portrait of a young woman with shallow depth of field | Same as 6; the generator's colour-bar signature in the corner becomes a few pixels after resizing |
| 8 | Real | 0.97 | Photo of a painter's palette and brushes | Chaotic, highly saturated paint texture looks like "AI art" — the style shortcut works against a real photo |
| 9 | AI | 0.23 | Studio portrait of an elderly man with round glasses | Photorealistic portrait lighting, like 6 |
| 10 | AI | 0.29 | Split image: segmentation sketch beside a photo-like landscape | Half the image is a flat colour map (looks like a diagram); the other half is photorealistic |

**Pattern.** The mistakes are dominated by *photorealistic* AI portraits (6, 7, 9) and by *non-photographic* AI images — posters, logos, CGI (2, 3, 5). The model has largely learned a **style** cue (saturated, over-detailed fantasy art ⇒ AI) rather than generator artefacts, which is consistent with the dataset: most AI images are fantasy art, while the *Real* folder mixes photos with paintings and graphics. Resizing to 224 px also removes fine high-frequency evidence. Image size and format are not the explanation here — nine of ten are ordinary JPEGs of typical size."""

R["GRADCAM_DISCUSSION"] = """*What the model looks at.* For correct AI predictions the heat concentrates on texture-rich regions — the fur and feathers of the fantasy creature (P(AI)=0.88) and the colourful panels of the AI collage (1.00): the over-detailed, glossy texture typical of generators. For correct real predictions the evidence is weak and diffuse (P(AI)=0.06 and 0.41), i.e. "real" is mostly the *absence* of such texture. The wrong predictions show the shortcut: the real photo of an artist in front of a vivid dotted artwork is called AI because the model attends to the saturated pattern (0.67), and the AI pop-art poster is called real because the activation sits on a small edge region rather than the content (0.10). The model mostly judges *style and texture*, not semantic giveaways such as malformed hands or text. Grad-CAM at MobileNetV2's last layer is only 7×7, so the maps are coarse."""

for k, v in R.items():
    assert k in s, k
    s = s.replace(k, v)
p.write_text(s)
print("filled", list(R))
