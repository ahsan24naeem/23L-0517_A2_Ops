---
title: "VisionGuard: AI vs Real Image Classification and Deployment"
subtitle: "MLOps — Assignment 2 · Project 23L-0517_A2_Ops"
---

| | |
|---|---|
| **Name** | Ahsan Naeem |
| **Roll number** | 23L-0517 |
| **Project name** | 23L-0517_A2_Ops (GitHub, DagsHub, Oracle Cloud, Docker Compose project) |
| **GitHub** | https://github.com/ahsan24naeem/23L-0517_A2_Ops |
| **DagsHub** | https://dagshub.com/ahsan24naeem/23L-0517_A2_Ops |
| **MLflow (DagsHub)** | https://dagshub.com/ahsan24naeem/23L-0517_A2_Ops.mlflow |
| **Model API (Oracle Cloud)** | http://ORACLE_IP:8000  (health: http://ORACLE_IP:8000/health) |
| **Graded commit** | COMMIT_HASH |

**Disclosure of AI assistance.** I used Claude (Anthropic) as an AI assistant throughout this assignment: it wrote most of the code (data audit, tf.data pipeline, models, MLflow instrumentation, Flask app, model API, Dockerfiles, Compose, CI), ran the training experiments in a cloud workspace, and drafted this report. I supplied the dataset, ran every step that needed my accounts or my machine (GitHub, DagsHub, Docker Desktop, Oracle Cloud, Postman), took all screenshots, checked the results and edited the text. No other tutorials were followed beyond the official documentation of TensorFlow/Keras, MLflow, DVC, DagsHub, Docker and Oracle Cloud.

**How to read this report.** Each chapter matches one task; each sub-section is a sub-task ID. Under each one I explain what I did, why, how it works and what went wrong. Grey boxes marked *Screenshot* are the evidence captured on my own machine (terminal prompt `ahsannaeem@…`, date and project name visible).

**System overview.**

```
 Browser ──► Flask web app (localhost / Docker Compose) ──► PostgreSQL (history, named volume)
                     │  INFERENCE_MODE=local: loads champion.keras once
                     │  INFERENCE_MODE=remote: HTTPS/HTTP + X-API-Key
                     ▼
              Model API on Oracle Cloud VM (Gunicorn + TensorFlow CPU, champion.keras mounted)
 Training: Kaggle data ─DVC─► DagsHub storage │ tf.data ► Keras ► MLflow (DagsHub) ► Model Registry @champion
```

# Task 1 — Data Understanding and Versioning

## 1.1 Environment

**What and why.** I created an isolated Python 3.11 virtual environment so the project's library versions cannot clash with anything else installed, and pinned exact versions in `requirements.txt` so anyone (and the Docker image, and CI) installs the same thing. Python 3.11 was chosen because every library in the stack (TensorFlow 2.19, MLflow 2.22, DVC 3.59) supports it.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_env.py      # prints versions and whether TensorFlow sees a GPU
```

| Library | Version | Library | Version |
|---|---|---|---|
| Python | 3.11 | MLflow | 2.22.1 |
| TensorFlow (CPU) | 2.19.0 | DVC | 3.59.1 |
| Keras | 3.15.1 | NumPy / pandas | 2.1.3 / 2.3.3 |
| Keras Tuner | 1.4.7 | Pillow | 13.0.0 |
| scikit-learn | 1.9.1 | Flask / Gunicorn | 3.1.3 / 23.0.0 |

**GPU.** `tf.config.list_physical_devices("GPU")` returns an empty list: TensorFlow sees only the CPU, so all training ran on CPU. `requirements.txt` uses environment markers: `tensorflow-cpu` on x86-64 Linux (no CUDA libraries, much smaller) and the regular `tensorflow` wheel on macOS/ARM, where it is CPU-only anyway.

**What went wrong.** (1) The newest `pathspec` release broke `dvc add` (`cannot import name '_DIR_MARK'`), so I pinned `pathspec==0.12.1`. (2) SQLAlchemy 2.1 removed a class MLflow 2.22 imports (`FallbackAsyncAdaptedQueuePool`); I pinned SQLAlchemy 2.0.44. Both are typical "unpinned transitive dependency" failures, which is exactly why the versions are now pinned.

> **Screenshot 1.1** — Terminal on my Mac: `source .venv/bin/activate && python scripts/check_env.py` showing user@host, date, all versions and "GPU available: False".

## 1.2 Dataset audit

**What.** `scripts/audit_dataset.py` walks `data/raw/<class>/`, opens every file with Pillow and records class, path, extension, file size, width, height, colour mode, decoded format, a SHA-256 content hash and whether the file fully decodes (`im.load()` catches truncated files, not just bad headers). Exact duplicates are files with the same SHA-256. Output: `reports/audit.csv` (one row per image) and `reports/audit_summary.md`.

**Why.** Before training, I need to know whether files are readable, whether duplicates could leak between train and test, and whether the two classes differ in trivial ways (size, format) that a model could exploit instead of learning what "AI-generated" looks like.

Table 1.1 — Files per class

| Class | Files | Readable | Corrupt | Duplicate copies |
|---|---|---|---|---|
| AI-generated (`AiArtData`) | 539 | 539 | 0 | 1 |
| Real (`RealArt`) | 436 | 436 | 0 | 0 |
| **Total** | **975** | **975** | **0** | **1** |

Table 1.2 — Image dimensions and file size (min / median / max)

| Class | Width (px) | Height (px) | File size (KB) |
|---|---|---|---|
| AI-generated | 225 / 1024 / 6000 | 135 / 800 / 4001 | 11.5 / 195.1 / 5494.3 |
| Real | 183 / 920 / 8495 | 158 / 720 / 7091 | 11.0 / 141.4 / 9956.4 |

Table 1.3 — Decoded format and colour mode

| Class | JPEG | PNG | RGB | RGBA | P (palette) | L (grey) |
|---|---|---|---|---|---|---|
| AI-generated | 429 | 110 | 489 | 35 | 14 | 1 |
| Real | 403 | 33 | 407 | 14 | 9 | 6 |

**Findings.**

- **Corrupt files: 0.** Every file decodes completely.
- **Exact duplicates: 1 group** — `AiArtData/image33.jpg` and `image34.jpg` are byte-identical. No duplicate spans both classes. The second copy is dropped before splitting so the same picture cannot be in train and test.
- **Extensions lie.** The file extension does not always match the content: 9 `.png` files are really JPEG, 4 `.jpg` are PNG, the 3 `.webp` and 1 `.gif` files are JPEG, and one file has no extension at all. This is why the audit records the *decoded* format, and why the serving code validates by decoding, never by extension alone.
- **Format imbalance.** 20 % of AI images are PNG versus 8 % of real images. A PNG has no JPEG compression artefacts, so a model could partly learn "no JPEG blocking ⇒ AI" — a shortcut, not real understanding. I keep this in mind in the error analysis.
- **Hidden files.** Two genuine images have names starting with a dot (`.amazonaws.com2F…png`). My first version skipped all dot-files and counted 973 images; DVC counted 975. I changed the filter to skip only `.DS_Store` and `._*` macOS metadata.

> **Screenshot 1.2** — Terminal: `python scripts/audit_dataset.py` printing the summary tables.

## 1.3 Visual EDA

![Figure 1.1 — Eight random images per class (displayed square; titles give the original size).](../reports/figures/eda_sample_grid.png){width=100%}

*Interpretation.* AI-generated images are dominated by saturated, highly detailed fantasy and psychedelic art, smooth "perfect" portraits and stock-like renders. The "Real" folder is not only photographs: it contains paintings (Mona Lisa), cartoons, magazine covers and at least one image that looks generated (a pink fantasy palace). So the real task is closer to "generated art vs everything else on the web", and some labels are noisy — this caps the accuracy any model can reach.

![Figure 1.2 — Width, height and file-size histograms per class.](../reports/figures/eda_histograms.png){width=100%}

*Interpretation.* Both classes span 135–8500 px, with medians near 1000×750 px, so every image must be resized to a fixed input size. AI images cluster at generator-typical widths (≈1024 and ≈1280 px) and are larger files (median 195 KB vs 141 KB) because more of them are PNG; real images have a long tail of very small thumbnails (<20 KB) and very large camera photos. Resolution and file size are not used as features, and because everything is resized to 224×224 the model cannot read them directly — but resampling a 6000-px photo versus a 300-px thumbnail leaves different blur, which is a possible weak shortcut.

![Figure 1.3 — Aspect ratios and decoded formats per class.](../reports/figures/eda_aspect_formats.png){width=95%}

*Interpretation.* Most images are landscape (ratio 1.3–1.8) with a spike of squares at 1.0, which is more common for AI images (generators often output 1:1). Resizing to a square therefore distorts most images somewhat; the decision is discussed in 2.1. The format panel visualises the PNG imbalance from 1.2.

## 1.4 DVC (data version control)

**Toy part.** `labs/dvc_toy/run_dvc_toy.sh` does every step in a new empty folder and echoes each command so the output can be read top-to-bottom:

1. `git init` + `dvc init` — Git will version code and *pointer files*, DVC the data.
2. A 3-row `weather.csv` is tracked with `dvc add weather.csv`. DVC moves the content into its cache (`.dvc/cache`), writes the pointer `weather.csv.dvc` and adds `weather.csv` to `.gitignore`. I commit the pointer: *data v1*.
3. Two rows are appended, `dvc add` again, commit: *data v2*. `git diff` shows that only the md5 and size inside the pointer changed.
4. **Going back:** `git checkout HEAD~1 -- weather.csv.dvc` restores the old pointer and `dvc checkout` replaces the CSV with the cached v1 content (3 rows). Then the same two commands bring v2 back.
5. **Remote:** `dvc remote add -d origin https://dagshub.com/ahsan24naeem/23L-0517_A2_Ops-dvc-toy.dvc`; the user name and token go into `.dvc/config.local`, which is never committed.
6. `dvc push` uploads the content; I then delete both `weather.csv` *and* `.dvc/cache` and run `dvc pull` — the file comes back from DagsHub, proving the remote holds the data.

**Apply.** The 975-image dataset (≈504 MB) is tracked the same way: `dvc add data/raw` produced `data/raw.dvc` (`md5: 8f23d6dc9b06d363c3981642c50856c0.dir`, `nfiles: 975`), the remote points at the project's DagsHub storage, and `dvc push` uploaded the images. The md5 is also logged as the `dvc_data_version` tag on every MLflow run, so each model can be traced back to the exact data.

**What the pointer file holds.** A few lines of YAML: the md5 hash of the content (for a folder, the hash of a manifest listing every file's hash), the total size, the number of files and the path. It is tiny, human-readable and changes whenever the data changes, so Git history *is* the data history.

**Why large data does not belong in Git.** Git stores every version of every file forever and every clone downloads the whole history: 500 MB of images, re-committed after each change, would make the repository huge and slow for everyone, and GitHub rejects files over 100 MB. Git's line-based diffs and merges are also meaningless for binary images. DVC keeps the heavy bytes in object storage, deduplicated by hash, and lets Git version only the small pointer.

> **Screenshot 1.3** — Toy run: `git log --oneline` with the "data v1" and "data v2" commits and the pointer diff.
> **Screenshot 1.4** — Toy run: `dvc checkout` back to v1 (3 rows), then `rm -rf weather.csv .dvc/cache` → `dvc pull` restoring the file.
> **Screenshot 1.5** — DagsHub web page of the toy repository showing the commits and the DVC-tracked `weather.csv`.
> **Screenshot 1.6** — `cat data/raw.dvc` and `dvc push` for the real dataset; DagsHub project page showing the `data/raw` folder tracked by DVC.

**What went wrong.** `dvc add data/raw` first failed because my `.gitignore` already ignored `/data/raw` (DVC refuses to create a pointer for a Git-ignored path — it needs to manage that ignore entry itself). Removing the line fixed it.

# Task 2 — Model Development with TensorFlow/Keras

**Splits.** `scripts/make_splits.py` drops corrupt files (none) and the duplicate copy, then makes a **stratified 70/15/15 split** with seed 42: train 681 (376 AI / 305 real), validation 146 (81 / 65), test 147 (81 / 66). The lists are saved as `splits/{train,val,test}.csv` (path, label, class, sha256), committed to Git and attached to every MLflow run, so every model sees exactly the same images. Validation is used for early stopping, threshold choice and model selection; **the test set is used once per model, after training, for the reported numbers.**

## 2.1 Input pipeline

`visionguard/data.py → make_dataset()` builds a `tf.data` pipeline from a split list:

1. **`from_tensor_slices((paths, labels))`** — a dataset of file paths and 0/1 labels (1 = AI-generated).
2. **`map(load, num_parallel_calls=AUTOTUNE)`** — decode + resize. It wraps `visionguard.preprocess.load_image` in `tf.numpy_function`: Pillow decodes the file (TensorFlow's own decoder cannot read the WEBP/odd files in this dataset), takes the first frame of animated images, applies EXIF rotation, converts every colour mode (L, P, RGBA) to **RGB**, and resizes to **224×224 with bilinear interpolation**, returning `uint8`. `ensure_shape` restores the static shape Keras needs.
3. **`cache(file)`** — decoded images are cached on disk per split and size. Decoding a 6000-px PNG is the slowest step; with the cache it happens once instead of every epoch (epoch time fell from ~33 s to ~13–20 s).
4. **`shuffle(len(train), reshuffle_each_iteration=True)`** — training only; a full buffer gives a true random order each epoch, seeded for reproducibility.
5. **`batch(32)`** and cast to float32.
6. **Augmentation — training data only** — a Keras `Sequential` of `RandomFlip("horizontal")`, `RandomRotation(±5 % of a turn ≈ ±18°)`, `RandomZoom(±10 %)` and `RandomContrast(±10 %)`, applied per batch. It enlarges the effective training set (681 images is tiny) without changing the label. I deliberately **did not** add blur, noise or JPEG re-compression: those would destroy the subtle high-frequency generator artefacts the model should learn. Validation and test data are never augmented, so their metrics measure the real images.
7. **`prefetch(AUTOTUNE)`** — prepares the next batch on the CPU while the model trains on the current one.

**Image size: 224×224** — the native resolution MobileNetV2's ImageNet weights were trained at, and a reasonable compromise on CPU. A 160-px variant is one of the experiments (3.5).

**Aspect ratio.** Images are **squashed** to a square, not cropped or padded. Most images are landscape (ratio ≈1.3–1.8), so objects get narrower, but the *whole* picture — including corners, borders and watermarks that can be informative — stays visible, and no padding bars (which differ between classes) are introduced. Because serving calls the identical function, the distortion is the same in training and production.

![Figure 2.1 — One augmented training batch (first 16 images; label above each).](../reports/figures/augmented_batch.png){width=95%}

## 2.2 Baseline CNN

Designed from scratch in `visionguard/models.py → build_baseline_cnn()`:

BASELINE_TABLE

Why each layer is there:

- **Rescaling(1/255)** — maps pixels to [0, 1] inside the model, so serving feeds raw pixels.
- **Four Conv(3×3, same) → BatchNorm → ReLU → MaxPool(2) blocks with 32→64→128→256 filters.** 3×3 convolutions learn local patterns (edges, textures, generator artefacts); stacking them grows the receptive field. Doubling filters while pooling halves the resolution keeps computation balanced and lets deeper layers represent more complex patterns. **BatchNorm** normalises activations so training is faster and less sensitive to the learning rate (bias is dropped from the conv because BN has its own shift). **ReLU** adds non-linearity. **MaxPooling** downsamples and gives small translation invariance.
- **GlobalAveragePooling** — one number per feature map; far fewer parameters than `Flatten` + Dense (which would have millions of weights and overfit 681 images).
- **Dropout(0.4)** — randomly zeroes 40 % of features during training to reduce overfitting.
- **Dense(1, sigmoid)** — one output = P(AI-generated).

**Loss, optimiser, metrics.** *Binary cross-entropy* is the correct likelihood loss for a single sigmoid output and gives calibrated probabilities. *Adam* (lr 1e-3) adapts the step size per weight and works well without tuning. Metrics: *accuracy* (easy to read), *ROC-AUC* (threshold-independent ranking quality — the main selection metric), *precision* and *recall* (the two error types matter differently here, see 2.5).

> **Screenshot 2.1** — `model.summary()` of the baseline CNN (text also in `reports/runs/cnn_baseline/model_summary.txt` and attached to the MLflow run).

## 2.3 Transfer learning (MobileNetV2)

`build_transfer_model("mobilenetv2")` loads `keras.applications.MobileNetV2(include_top=False, weights="imagenet")` — the ImageNet feature extractor without its 1000-class classifier — and adds my head: **GlobalAveragePooling → Dropout(0.3) → Dense(1, sigmoid)**.

**Preprocessing.** MobileNetV2 was trained on pixels scaled to **[-1, 1]** (`preprocess_input` = x/127.5 − 1). Feeding [0, 1] or [0, 255] would shift every activation away from what the pretrained filters expect and waste the pretraining. I implemented it as `Rescaling(1/127.5, offset=-1)` *inside* the model, so the saved model contains its own preprocessing. (For the ResNet50 experiment the model contains the Caffe-style preprocessing ResNet50 expects instead: RGB→BGR and ImageNet mean subtraction, no scaling.)

**Stage 1 — frozen backbone.** `base.trainable = False`; only the head trains (lr 1e-3, ≤15 epochs). The backbone is called with `training=False`, which keeps its BatchNorm layers in inference mode even later, so their ImageNet statistics are not overwritten by small batches.

**Stage 2 — fine-tune the last block.** I unfreeze only MobileNetV2's last block (`block_16_*` + the final `Conv_1` 1×1 convolution), keep all BatchNorm layers frozen, recompile with a **100× smaller learning rate (1e-5)** and train ≤15 more epochs. The small rate adapts high-level features to "AI vs real" without destroying what the pretrained weights know.

TRAINABLE_LAYERS

TRANSFER_CURVES

## 2.4 Training discipline

- **Seeds:** `keras.utils.set_random_seed(42)` seeds Python, NumPy and TensorFlow; the shuffle and augmentation layers also get seed 42; the split uses `random_state=42`. (CPU training is then repeatable up to small floating-point differences from multi-threading.)
- **EarlyStopping(monitor=val_loss, patience=4, restore_best_weights=True)** — stops when validation loss has not improved for 4 epochs (6 for the CNN).
- **ModelCheckpoint(val_loss, save_best_only=True)** — keeps the single best model across both stages on disk; the final evaluation reloads it.
- **ReduceLROnPlateau(val_loss, factor=0.3, patience=2)** — cuts the learning rate when progress stalls.

BEST_EPOCHS

## 2.5 Evaluation

**Which error is worse?** For VisionGuard, **calling a real photo fake (false positive)** is worse. The product will be used to judge journalism, evidence, art submissions and personal photos; a false "AI-generated" verdict publicly accuses an honest person of fabrication and can damage reputations, while missing a fake (false negative) leaves the situation as it would have been without the tool. Users can also apply other checks to a suspicious image, but a false accusation is hard to undo.

**Threshold choice — on validation data.** For each threshold from 0.05 to 0.95 I compute F1 and the false-positive rate (real called AI) on the *validation* set and pick the threshold with the **highest F1 subject to FPR ≤ 15 %**. The cap encodes the product decision above; F1 then balances precision and recall within it. Choosing the threshold on the test set would leak test information into the model and overstate performance.

![Figure 2.4 — Threshold selection on validation data for the chosen model.](../reports/figures/threshold_selection.png){width=80%}

FINAL_METRICS

![Figure 2.5 — Test-set ROC curves of both models.](../reports/figures/roc_both_models.png){width=62%}

![Figure 2.6 — Test-set confusion matrices at the validation-chosen thresholds.](../reports/figures/confusion_both_models.png){width=85%}

EVAL_DISCUSSION

## 2.6 Error analysis

![Figure 2.7 — The 10 most confident mistakes of the chosen model on the test set (title: true class, P(AI), original size and format).](../reports/figures/misclassified_grid.png){width=100%}

ERROR_TABLE

## 2.7 Tool lab: Keras Tuner

**Toy part.** `labs/keras_tuner_toy.py` tunes a one-hidden-layer network on **Fashion-MNIST** with **Hyperband** (max 6 epochs, factor 3), searching `units ∈ {32…256}` and `learning_rate ∈ [1e-4, 1e-2]` (log scale), objective validation accuracy. Hyperband trains many configurations for 2 epochs, promotes the best third to 6 epochs, etc., so bad settings are dropped cheaply.

TOY_TUNER

**Apply.** `scripts/tune.py` runs **RandomSearch with TUNER_TRIALS trials** on MobileNetV2 (224 px), searching learning rate ∈ [1e-5, 1e-3] (log), dropout ∈ {0.1 … 0.5} and **number of unfrozen backbone layers ∈ {0, 10, 20, 40}**. Each trial trains for up to 5 epochs with early stopping and is scored on validation AUC; the table is logged to MLflow (experiment `VisionGuard-tuning`).

TUNER_RESULTS

**Why choosing the best validation score can overfit the validation set.** Each trial's validation score is a noisy estimate (146 images: one image is 0.7 % accuracy). Trying many configurations and keeping the maximum selects partly for *luck on these particular 146 images* — the winner's score is optimistically biased, and the more trials, the stronger the bias ("winner's curse"). The validation set has effectively been used for fitting the hyper-parameters. That is why the final numbers are reported on the untouched test set, and why small differences between trials (±0.01 AUC) should not be over-interpreted; with more data, nested or k-fold cross-validation would give a more reliable choice.

## Bonus — Grad-CAM

![Figure 2.8 — Grad-CAM heat maps for 4 correct (left) and 4 wrong (right) predictions of the chosen model.](../reports/figures/gradcam.png){width=100%}

GRADCAM_DISCUSSION

# Task 4 — Flask Web Application

## 4.1 Inference module

`visionguard/inference.py` contains a `Predictor` class. Its constructor runs **once**, when the app starts: it loads `champion.keras`, reads the companion `champion.json` (decision threshold, model name, registry version), records the input size from the model itself, and runs one warm-up prediction so the first user does not pay TensorFlow's graph-building cost. `predict(image)` then (1) decodes and resizes the upload with `visionguard.preprocess.load_image` — **the same function the training `tf.data` pipeline calls** — (2) runs the model, (3) compares P(AI) with the threshold and returns a label, the confidence in that label (P(AI) or 1 − P(AI)), the raw probability, the threshold, model name/version and latency.

**Why loading per request is unacceptable.** Loading the model means reading ~10 MB from disk, deserialising the Keras graph and initialising TensorFlow — about 2–4 seconds and a few hundred MB of memory each time. Per request, every user would wait seconds, memory would spike with concurrent users (each request holding its own copy) and the server would fall over under modest load. Loading once and sharing the in-memory model (protected by a lock for thread safety) makes each prediction ~100 ms.

**Why training and serving preprocessing must match.** The model only learned the mapping from pixels prepared one particular way (Pillow decode → RGB → bilinear squash to 224×224 → raw 0–255, with MobileNetV2's `[-1, 1]` scaling inside the model). If serving used, say, TensorFlow's decoder with a different resize filter, cropping instead of squashing, BGR order or 0–1 scaling, the input distribution would shift and accuracy would silently drop — the classic *training/serving skew*. Putting the preprocessing in one shared function, and the model-specific scaling inside the saved model, removes that risk by construction.

```text
$ python -c "from visionguard.inference import Predictor; p=Predictor('models/champion.keras'); ..."
INFERENCE_CONSOLE_OUTPUT
```

> **Screenshot 4.1** — Console output of the inference module for one known real image and one known AI image.

## 4.2 Routes and validation

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Upload page |
| `/predict` | POST | Validates, predicts, writes a DB row, then redirects (303) to `/result/<id>` |
| `/result/<id>` | GET | Verdict, confidence bar, probability, threshold, model + version, latency, caveat |
| `/history` | GET | Paginated table of all requests from the database + summary counts |
| `/api/history` | GET | Same history as JSON |
| `/health` | GET | JSON: model loaded, model name/version/threshold, DB reachable, debug flag; 200 or 503 |

Validation (`visionguard/validation.py`, shared with the cloud API):

- **Wrong file type → 415.** The extension must be one of jpg/jpeg/png/webp/gif/bmp.
- **Too large → 413.** `MAX_CONTENT_LENGTH` (from `MAX_UPLOAD_MB`, default 5 MB) makes Flask reject the body before reading it all, and the validator double-checks the size.
- **Looks like an image but cannot be decoded → 422.** Pillow must both `verify()` and fully `load()` the bytes — a text file renamed `photo.jpg` or a truncated JPEG is rejected.
- **File-name sanitising.** `werkzeug.utils.secure_filename` turns `../../etc/My Photo.png` into `etc_My_Photo.png`; only this cleaned name is stored or displayed.
- **The upload never touches disk.** Werkzeug normally spools uploads larger than 500 KB to a temporary file; a custom `Request` class overrides `_get_file_stream` to return an in-memory `BytesIO`. The bytes are hashed (SHA-256), predicted on and discarded — only metadata is stored.
- **Friendly error pages.** Every rejection renders `error.html` with a plain-language title and message and the right status code; rejected uploads are not written to the history.

The pytest suite (`tests/test_app.py`, `tests/test_validation.py`) checks each of these cases automatically.

> **Screenshot 4.2** — Error page "Unsupported file type" (uploading a `.txt`).
> **Screenshot 4.3** — Error page "File too large" (uploading a > 5 MB image).
> **Screenshot 4.4** — Error page "Not a valid image" (a text file renamed `.jpg`).

## 4.3 HTML and CSS

Jinja templates (`app/templates/`) share a `base.html` layout with a top menu — **Upload · History · Health** — and a footer that always shows which model version and inference mode are active. The upload page previews the chosen image in the browser before sending it. The result page shows the verdict in large type (green for Real, red for AI-generated), a confidence bar, the probability and threshold, the model and version, the latency and an explicit caveat that the verdict is a hint, not proof. The History page reads from the database and shows totals plus a table of every request (time, file, verdict, confidence, P(AI), model/version, mode, latency); each row links back to its result page. One plain CSS file, no framework; it adapts to dark mode and phone widths.

> **Screenshot 4.5** — Upload page.  **Screenshot 4.6** — Result page for an AI image.  **Screenshot 4.7** — Result page for a real photo.  **Screenshot 4.8** — History page.

## 4.4 Configuration and production server

Every setting is read from environment variables in `app/config.py`: `MODEL_PATH`, `THRESHOLD` (empty = use the validated threshold stored in `champion.json`), `DATABASE_URL`, `INFERENCE_MODE`, `MODEL_API_URL`, `MODEL_API_KEY`, `MAX_UPLOAD_MB`, `SECRET_KEY`, `FLASK_DEBUG`. Locally they come from `.env`; in Compose from `env_file`; in the cloud from the VM's `.env`. The same code therefore runs everywhere and no secret or machine-specific path is in Git.

`python -m app` starts the built-in Flask server with `debug=False` and the reloader off; inside Docker the app runs under **Gunicorn** (`gunicorn app.wsgi:app`, 1 worker × 4 threads). Debug mode stays off because: the Werkzeug debugger gives anyone who can trigger an error an interactive Python console (remote code execution); detailed tracebacks leak paths, settings and code; and the auto-reloader starts the process twice, loading the model twice. The built-in server is single-process and not hardened, which is why a production WSGI server is used in the container.

```json
HEALTH_JSON
```

> **Screenshot 4.9** — Browser at `http://localhost:8000/health` showing `"status": "ok"`, `"debug": false`, model version and `"database": {"backend": "postgresql", "ok": true}`.

# Task 5 — Database Trail

**Choice: PostgreSQL in Docker Compose (SQLite for local development and tests).** Both are reached through SQLAlchemy Core (`app/db.py`), so only `DATABASE_URL` changes. PostgreSQL was chosen for the deployed stack because it is a real server that several app processes can write to concurrently and it lives in its own container with a persistent named volume; SQLite is a single file, perfect for tests.

The table `predictions` is created automatically at start-up. **One row is inserted in the `/predict` view after every successful prediction**, before redirecting to the result page; the result and history pages read back from it.

| Column | Answers |
|---|---|
| `id`, `created_at` (UTC, timezone-aware) | **when** |
| `label`, `is_ai`, `p_ai`, `threshold` | **what was decided** (and the raw score behind it) |
| `confidence` | **how confidently** |
| `model_name`, `model_version`, `inference_mode` | **by which model** (registry version, local or remote API) |
| `filename` (sanitised), `file_sha256`, `file_size_bytes`, `image_width`, `image_height` | which input — without storing the image itself |
| `latency_ms` | how long it took |

Storing the SHA-256 instead of the image lets me detect repeated uploads and audit decisions without keeping user content. The **History** menu item renders this table (25 rows per page, newest first) with totals; `/api/history` returns the same rows as JSON. Because PostgreSQL publishes no port, the trail can only be queried from inside the Compose network, e.g.:

```bash
docker compose exec db psql -U visionguard -d visionguard \
  -c "select id, created_at, label, round(confidence::numeric,3), model_version, inference_mode from predictions order by id desc limit 5;"
```

> **Screenshot 5.1** — History page after several uploads.
> **Screenshot 5.2** — The same rows queried with `psql` inside the `db` container.

# Task 6 — Containerization with Docker and Docker Compose

## 6.1 Docker warm-up

`labs/docker_warmup/warmup.sh` practises every basic action with the ready-made `nginx:alpine` image: `docker pull`; `docker run -d --name vg-warmup -p 8080:80 -v "$PWD/site":/usr/share/nginx/html:ro nginx:alpine`; `curl localhost:8080` (the page comes from my mounted folder, and an edit on the host shows up immediately); `docker logs vg-warmup`; `docker exec -it vg-warmup sh` (shell inside); `docker ps` / `docker image ls` / `docker history`; `docker stop`, `docker rm`, `docker rmi`.

- **Image** — a read-only, versioned template: a filesystem snapshot plus metadata (command, ports, environment). Built once, run many times.
- **Container** — a running (or stopped) instance of an image: an isolated process with its own filesystem view, network and a thin writable layer on top of the image.
- **Layer** — each Dockerfile instruction adds a layer, a diff of files on top of the previous ones. Layers are content-addressed, cached and shared between images, which makes rebuilds and downloads fast.
- **Volume** — storage that lives outside the container's writable layer, so data survives when the container is deleted (named volumes are managed by Docker; bind mounts map a host folder).
- **Port mapping** — `-p HOST:CONTAINER` forwards a port on my machine to a port inside the container's private network; without it the service is unreachable from outside.

> **Screenshots 6.1–6.6** — run with port and volume; `curl` output; `docker logs`; shell inside; `docker ps`/`image ls`; `rm` and `rmi`.

## 6.2 Dockerfile for the web app

`docker/Dockerfile` (one image used for both the web app and the cloud API; `APP_MODULE` selects which):

1. `FROM python:3.11-slim` — Debian slim base (~50 MB compressed) instead of the 1 GB full image.
2. `ENV` — no `.pyc` files, unbuffered logs, `PIP_NO_CACHE_DIR=1`, quiet TensorFlow.
3. `COPY requirements-app.txt` then `RUN pip install …` — **dependencies before source code.** Docker reuses a cached layer as long as the instruction and its inputs are unchanged. The requirements file changes rarely, so the slow ~1 GB TensorFlow install is cached; when I edit Python code only the final `COPY` layers are rebuilt (seconds, not minutes). If the code were copied first, every code edit would invalidate the cache and reinstall TensorFlow.
4. **TensorFlow package chosen deliberately:** `tensorflow-cpu==2.19.0` on amd64 — the servers have no GPU, and the CPU build omits the CUDA/cuDNN libraries (hundreds of MB). On arm64 (Apple Silicon, Oracle Ampere) the standard `tensorflow` wheel is already CPU-only. `keras==3.15.1` is pinned to the exact version that saved the model.
5. Delete `__pycache__` and bundled test folders in the same `RUN`, so they never enter a layer.
6. `useradd --uid 10001 vg` and `USER vg` — **non-root**: if the app is compromised, the attacker cannot modify the image, install packages or write to the read-only model mount.
7. `COPY --chown=vg:vg visionguard/ app/ api/` — source last.
8. `HEALTHCHECK` calls `/health` every 30 s (90 s start period for model loading); Docker/Compose mark the container healthy/unhealthy.
9. `CMD gunicorn --bind 0.0.0.0:8000 --workers 1 --threads 4 $APP_MODULE`.

`.dockerignore` excludes `.git`, `.env*`, `data/`, `models/`, `mlflow.db`, reports, tests and labs. This keeps the build context small and — importantly — **keeps secrets out of the image**: `.env` can never be copied in, and secrets are only injected at run time through `env_file`.

```text
$ docker compose exec web whoami && docker compose exec web id
WHOAMI_OUTPUT
```

> **Screenshot 6.7** — `docker compose exec web sh -c 'whoami; id'` showing `vg`, uid 10001.

## 6.3 Compose stack

`docker-compose.yml` defines two services on a private bridge network `backend`:

- **`db`** — `postgres:16-alpine`; credentials from `.env`; data in the **named volume `pgdata`**; a health check runs `pg_isready` every 5 s; **no `ports:` section**, so it is reachable only from containers on `backend`.
- **`web`** — built from `docker/Dockerfile`; `DATABASE_URL=postgresql+psycopg2://…@db:5432/…` — the database is addressed by its **service name `db`**, which Compose's internal DNS resolves; `depends_on: db: condition: service_healthy` so the app starts only after Postgres accepts connections (plus a small connection-retry loop in the app); port 8000 published.

All passwords come from an untracked `.env` (template: `.env.example`).

**How the model reaches the container: mounted, not built in.** `./models` is bind-mounted read-only at `/models`. Reasons: the image stays independent of any particular model (one image, many model versions); promoting a new champion means replacing one file and restarting — no rebuild or re-push of a 1.5 GB image; the model file is versioned by DVC/MLflow, not by Docker layers; and the read-only mount means the app cannot alter it. `models/champion.keras` arrives on a machine via `dvc pull` (or `python scripts/register_model.py`, which exports the `@champion` version from the registry once). The serving app never calls DagsHub per request. The trade-off — the container is not self-contained — is acceptable because Compose declares the mount explicitly.

```text
$ docker compose ps
COMPOSE_PS
$ nc -vz localhost 5432
NC_OUTPUT
```

> **Screenshot 6.8** — `docker compose up -d --build` followed by `docker compose ps` with both services `(healthy)`.
> **Screenshot 6.9** — Proof the DB port is closed: `docker compose port db 5432` prints nothing / `nc -vz localhost 5432` fails, while `docker compose exec web python -c "import socket; socket.create_connection(('db',5432)); print('db reachable from web')"` succeeds.

## 6.4 Image analysis

IMAGE_ANALYSIS

> **Screenshot 6.10** — `docker image ls` showing both images and `docker history 23l-0517_a2_ops-web` showing the largest layers.

## 6.5 Operations and troubleshooting

`scripts/ops_demo.sh ops` demonstrates `docker compose up -d`, `stop web`, `start web`, `restart web`, `logs web` / `logs db`, and a shell inside each service (`exec web sh`, `exec db psql …`).

**Deliberate failure — wrong database password.** `scripts/ops_demo.sh break-password` changes `POSTGRES_PASSWORD` in `.env` and recreates only the web container. The database keeps its original password, because Postgres reads `POSTGRES_PASSWORD` only the first time the volume is initialised.

*Diagnosis from logs only.* `docker compose ps` shows `web` restarting. `docker compose logs web` shows:

```text
OPS_FAILURE_LOG
```

The key line is `FATAL: password authentication failed for user "visionguard"`: the network and the DB container are fine (the connection reached Postgres), but the credentials are rejected — so the problem is configuration, not code. *Fix:* restore the original password in `.env` (`ops_demo.sh fix-password`) and recreate `web`; `docker compose ps` shows both services healthy again. (If the password really had to change, the correct fix is `ALTER USER … PASSWORD …` inside Postgres, or recreating the volume.) A second scenario, `break-model`, renames `champion.keras`: the logs show `MODEL LOAD FAILED: FileNotFoundError` and `/health` returns 503 with the error, and the container turns `unhealthy`.

> **Screenshots 6.11–6.13** — ops commands; the failure logs; healthy again after the fix.

## 6.6 Tool lab: GitHub Actions (CI)

**Toy part.** Throwaway repository `23L-0517_A2_Ops-ci-toy` (`labs/ci_toy/`): `calc.py`, `tests/test_calc.py` and `.github/workflows/test.yml`, triggered on every push, with a **matrix of Python 3.10 and 3.12** — the job runs twice in parallel. I pushed a bug (`add` subtracting) → **red** run, both matrix legs failing on `test_add`; reverted it → **green** run.

**Apply.** `.github/workflows/ci.yml` for VisionGuard runs on every push and pull request:

1. `lint-and-test`: checkout → Python 3.11 with pip cache → install `requirements-app.txt` + pytest + flake8 → `flake8` → `pytest` (20 tests; a fake predictor replaces the Keras model, so CI does not download TensorFlow for the tests).
2. `docker-build` (after tests pass): Buildx builds `docker/Dockerfile` with GitHub Actions layer cache, then smoke-tests the image (`whoami`, `import tensorflow, app, api`).

The status badge is at the top of `README.md`.

**Where secrets live.** In the repository's *Settings → Secrets and variables → Actions*; a workflow reads them as `${{ secrets.NAME }}` and GitHub masks them in logs. They must never be written in the workflow file because that file is committed: anyone who can read the repository (or its history, forever, even after deletion) could read the secret, and forks would copy it. This workflow needs none (it builds but does not push the image); a registry token for pushing would be stored as a secret.

> **Screenshot 6.14** — Toy repository Actions tab: red run (matrix 3.10 / 3.12) and green run.
> **Screenshot 6.15** — VisionGuard Actions run: both jobs green. **Screenshot 6.16** — README with the passing badge.

# Task 7 — API Deployment (Oracle Cloud, pathway A)

**What runs where.** As the task specifies, the **model API** is deployed in the cloud and the **Flask web app runs on my laptop** and calls it over the internet. The API (`api/__init__.py`) is a small Flask service served by Gunicorn in the same Docker image as the web app (`APP_MODULE=api.wsgi:app`), with the champion model mounted read-only. It is stateless — the history database stays with the web app.

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /` | none | service info: model name, registry version, threshold, input size |
| `GET /health` | none | `status`, model info, uptime, requests served (200 / 503) |
| `POST /predict` | `X-API-Key` header | multipart field `image` → `label`, `is_ai`, `confidence`, `p_ai`, `threshold`, `model_name`, `model_version`, `latency_ms`, sanitised `filename`, `sha256`, size |

The key is compared in constant time (`hmac.compare_digest`) and comes from the VM's `.env`; without it the API answers 401. The same upload validation as the web app (type 415, size 413, undecodable 422) applies.

## Research before deploying

- **Current free limits (checked in November 2026).** Oracle's documentation now lists Ampere A1 Always Free as **2 OCPUs and 12 GB RAM** in total (1,500 OCPU-hours and 9,000 GB-hours per month) — halved from 4 OCPU / 24 GB in June 2026. Also free: up to two **E2.1.Micro** x86 VMs (1/8 OCPU, 1 GB RAM, 50 Mbps), **200 GB** block storage (boot volumes included) and **10 TB/month** outbound traffic. My console's *Limits, Quotas and Usage* page confirmed the numbers for my tenancy (Screenshot 7.1).
- **Arm vs x86.** A1 is Arm64 (Ampere Altra); E2.1.Micro is x86-64. TensorFlow import alone needs ~600 MB RAM, so the 1 GB micro VM would need swap and be slow; I used **one A1 instance with 1 OCPU / 6 GB** and an **Ubuntu 22.04 aarch64** image. The Dockerfile handles both architectures (`TARGETARCH`: `tensorflow-cpu` on amd64, `tensorflow` on arm64), and the image is built on the VM itself so it is native arm64.
- **Free-instance availability.** A1 capacity is often exhausted ("Out of host capacity"); the fix is to try another availability domain or retry later.
- **Idle-instance reclamation.** Oracle may reclaim an Always Free instance if, over 7 days, its 95th-percentile CPU, network and (for A1) memory utilisation are all below 20 %. A demo API is mostly idle, so I keep the instance only for the grading period (a cron job generating synthetic load would be against the spirit of the rule).
- **The two firewalls.** Traffic must pass (1) the **VCN Security List** (or a Network Security Group) in the OCI console — an ingress rule for TCP 8000 from 0.0.0.0/0 — *and* (2) the **VM's own iptables**, because Oracle's Ubuntu images ship rules that reject everything except SSH. Opening only one of them is the most common reason the API "does not respond".

## Step-by-step deployment

1. **Create the VM.** OCI Console → Compute → Instances → *Create instance*: name `23L-0517_A2_Ops`, image *Canonical Ubuntu 22.04 (aarch64)*, shape *VM.Standard.A1.Flex* 1 OCPU / 6 GB, assign a public IPv4, upload my SSH public key. *(Screenshot 7.2)*
2. **Open port 8000 in the VCN.** Instance → Subnet → Default Security List → *Add Ingress Rule*: source `0.0.0.0/0`, TCP, destination port `8000`. *(Screenshot 7.3)*
3. **SSH in and prepare the VM.** `ssh ubuntu@ORACLE_IP`, then `bash deploy/oracle/setup_vm.sh` (from the repo) installs Docker Engine + Compose, adds the user to the `docker` group, opens TCP 8000 in iptables (`iptables -I INPUT 6 … --dport 8000 -j ACCEPT` + `netfilter-persistent save`) and adds a 2 GB swap file. *(Screenshot 7.4)*
4. **Get the code and the model.** `git clone https://github.com/ahsan24naeem/23L-0517_A2_Ops.git && cd 23L-0517_A2_Ops`; from my laptop `scp models/champion.keras models/champion.json ubuntu@ORACLE_IP:~/23L-0517_A2_Ops/models/` (alternatively `dvc pull models/champion.keras` with DagsHub credentials). The model is copied once at deploy time — the API never contacts DagsHub while serving.
5. **Secrets.** `cp .env.example .env` and set `MODEL_API_KEY` to a long random value (`python3 -c "import secrets;print(secrets.token_urlsafe(32))"`). `.env` stays on the VM only.
6. **Build and start.** `docker compose -f deploy/oracle/docker-compose.yml up -d --build`, then `docker compose -f deploy/oracle/docker-compose.yml ps` (healthy) and `logs` (`model loaded: visionguard v2`). *(Screenshot 7.5)*
7. **Test from the VM and from outside.** `curl http://localhost:8000/health` on the VM; `curl http://ORACLE_IP:8000/health` from my laptop **on a different network (mobile hotspot)**. *(Screenshot 7.6)*

**API URL:** `http://ORACLE_IP:8000` — health check `http://ORACLE_IP:8000/health`, prediction `POST http://ORACLE_IP:8000/predict`.

**Mapping of Compose elements to the provider.** Compose service → Docker on the VM; published port → VCN security-list rule + iptables rule; `.env` → file on the VM (could be OCI Vault); bind-mounted model → file on the VM's boot volume; restart policy `unless-stopped` → containers return after a VM reboot.

## Testing with Postman

`deploy/postman/VisionGuard.postman_collection.json` is imported into Postman with collection variables `base_url` and `api_key`. Each request has test scripts:

| # | Request | Expected | Logged result |
|---|---|---|---|
| 1 | `GET /health` | 200, `status: ok` | POSTMAN_1 |
| 2 | `POST /predict` real photo + key | 200, label + confidence | POSTMAN_2 |
| 3 | `POST /predict` AI image + key | 200 | POSTMAN_3 |
| 4 | `POST /predict` without key | 401 | POSTMAN_4 |
| 5 | `POST /predict` a `.txt` file | 415 | POSTMAN_5 |

> **Screenshot 7.7** — Postman: `/predict` with the API key, response JSON and passing tests.  **Screenshot 7.8** — Postman: 401 without the key.  **Screenshot 7.9** — Postman Runner / Console with all five results.  **Screenshot 7.10** — `docker compose logs api` on the VM showing one log line per request.

**What went wrong / trade-offs.** The API is plain HTTP, so the API key travels unencrypted; for a real product I would put Caddy or Nginx with a TLS certificate in front (e.g. a free `*.sslip.io` hostname with Let's Encrypt). The A1 instance is built natively for arm64, which is why the Dockerfile selects the TensorFlow package per architecture.

# Task 8 — Integrating Flask and the API

The web app has two interchangeable back-ends behind one interface (`app/predictors.py`): `LocalPredictor` loads the Keras model in-process (used in Docker Compose) and `RemotePredictor` forwards the upload to the cloud API with the `X-API-Key` header and a timeout. Setting

```bash
INFERENCE_MODE=remote
MODEL_API_URL=http://ORACLE_IP:8000
MODEL_API_KEY=<same key as on the VM>
python -m app            # Flask on http://127.0.0.1:5000, debug off
```

switches the running app to the cloud model without any code change. At start-up the app calls the API's `/health` to learn the model name, version and threshold. For every upload the Flask view validates the file locally (rejecting bad files before they cost a network round trip), sends the bytes to `POST /predict`, stores the API's answer in the local database with `inference_mode = remote` and the measured round-trip latency, and redirects to the result page. If the API is down or the key is wrong, the user sees a friendly "Model service error" page (502) instead of a crash, and nothing is written to the history.

INTEGRATION_TABLE

> **Screenshots 8.1–8.6** — Six uploads through `localhost:5000` (three real, three AI) with the result pages showing *Served by: remote*.
> **Screenshot 8.7** — History page listing those requests with model version and mode `remote`.
> **Screenshot 8.8** — `/health` of the local app in remote mode (`"inference_mode": "remote"`, `api_url`), and the VM's API logs for the same requests.

