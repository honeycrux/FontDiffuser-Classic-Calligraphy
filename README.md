# FontDiffuser Classic Calligraphy: A Study On FontDiffuser's Capability To Generate Calligraphy

This document is provided by the FYP24 project group.

## 🌟 Introduction

This project is derived from the work "FontDiffuser: One-Shot Font Generation via Denoising Diffusion with Multi-Scale Content Aggregation and Style Contrastive Learning" by Yang et al. ([arXiv](https://arxiv.org/abs/2312.12142)) ([GitHub](https://github.com/yeungchenwa/FontDiffuser)). Consequently, this page contains numerous references to the README page of the FontDiffuser project by its original authors, which is included in [FontDiffuser.md](./FontDiffuser.md).

This project explores ways to modify FontDiffuser to generate Chinese Calligraphy. We approached this problem by changing the model from one-shot to few-shot, allowing the model to infer the style from multiple samples from the target distribution.

This project currently has 5 maintained branches using different proposed methods as described below.

One shot methods:
- `main`: The original FontDiffuser model with our code enhancements.

Few-shot methods:
- `release/naive-few-shot`: The Naive Few-shot method, which takes an average of the features of style samples to infer encodings of a single style.
- `release/conv-few-shot`: The Convolution Few-shot method, which uses a combination of linear and convolutional layers on the features of style samples to infer encodings of a single style. There are multiple types of implementation that can be swapped in `k_feature_extractor.py`.
- `release/attn-few-shot`: The Attention Few-shot method, which uses attention blocks on the features of style samples to infer encodings of single style.
- `release/style-reconst`: The Style Reconstruction method, which uses attention blocks on the features of content image and style samples to infer one style encoding, and uses the multi-scale content encodings of a random style sample.

This project contains the following code enhancements to support research:
- Add the ability to enable validation split and the calculation of validation loss during training.
- Add the ability to resume training.
- Add pre-training checks against the font dataset to satisfy training requirements.
- Add the ability to use multiple images of the same character using the format `<style>+<character>+<sequence_number>.png`.
- Add an evaluation script to support the evaluation methodology used in this paper.
- Add auto image padding for non-square character images.
- Dependency fixes and better compliance to Pylance "standard" type checking.

## 📅 Timeline

- **August 2025**: The data preparation scripts are released.
- **March 2025**: Introduced the Style Reconstruction method.
- **February 2025**: Introduced the Attention Few-shot method.
- **November 2024**: Introduced the Convolutional Few-shot method.
- **October 2024**: Introduced the Naive Few-shot method.

## 🛠️ Installation

For the installation process, refer to [FontDiffuser#Installation](./FontDiffuser.md#️-installation).

## 🛠️ Development

We specifically perform merges in the following way to propagate changes:
- `main` commits, containing overall improvements, are merged into `release/naive-few-shot`.
- `release/naive-few-shot` commits, containing overall improvements and adaptations to few-shot generation, are merged into `release/conv-few-shot`, `release/attn-few-shot`, and `release/stlye-reconst`.

All branches contain the same readme documents but different model implementations.

## 🔥 Models

Different branches contain the code to train and run different models. Each model has its own modification for experimentation. One may switch to a specific branch to use a specific model.

| Model | Branch |
| ----- | ------ |
| Original FontDiffuser Model (OFD) | `main` |
| Naive Few Shot (NFS) | `release/naive-few-shot` |
| Convolution Few Shot (CFS) | `release/conv-few-shot` (see Choosing Conv Models section) |
| Fully-connected Few Shot (FCFS) | `release/conv-few-shot` (see Choosing Conv Models section) |
| Hybrid Few Shot (HFS) | `release/conv-few-shot` (see Choosing Conv Models section) |
| Attention Few Shot (AFS) | `release/attn-few-shot` |
| Style Reconstruction (SR) | `release/style-reconst` |

Except for OFD and NFS, we provide `scripts/train-phase-3.sh` for additional training using OFD weights (resulting weights by the authors of FontDiffuser). OFS and NFS directly use OFD weights and do not have a phase-3 training script.

For SR, we provide `scripts/train-phase-3-sr-only.sh` instead to indicate that we only train the StyleReconstructor unit without training the other parts. We only tried this technique with the SR model.

Except for OFD, the models can take multiple style images. The number of style images accepted can be set with `k_shot` and/or `max_k`. Check configurations to find available config arguments to a specific model:

- `configs/fontdiffuser.py` (common parameters for training and sampling)
- `sample.py > arg_parse()` (specific parameters for sampling)

The files `lantingjixu_sample.py` and `scripts/train-phase-3.sh` (if provided) can be used as example configurations to training/sampling.

### Choosing Conv Models

A few models use the `release/conv-few-shot` branch, each with different layers. You can choose the model used by changing the variable `k_feature_extractor.py` > `class KFeatureExtractor` > `unit_used` to one of the following classes:

| Model | Class To Use |
| ----- | ------------ |
| Convolution Few Shot (CFS) | `KFeatureExtractorUnit_CFS` |
| Fully-connected Few Shot (FCFS) | `KFeatureExtractorUnit_FCFS` |
| Hybrid Few Shot (HFS) | `KFeatureExtractorUnit_HFS` |

`unit_used` is originally set to `KFeatureExtractorUnit_CFS`.

### Model Objectives

We have two objective paths that we try to use our models to achieve:
1. **General Calligraphy Model**: The objective is to generate authentic calligraphy given any calligraphy work.
2. **Single-style Calligraphy Model**: The objective is to generate one style only.

## 🔥 Dataset Preparation Scripts

See https://github.com/honeycrux/Font-Datasets-fyp24.

## 🔥 The Lantingji Xu Dataset

The Lantingji Xu dataset we used is available in the `data_lantingjixu` folder, with the following content:
- `lantingjixu_title.txt`: The title of Lantingji Xu.
- `lantingjixu_authentic.txt`: The authentic transcription of Lantingji Xu.
- `lantingjixu_text.txt`: The transcription of Lantingji Xu with obscure characters substituted with more commonly-used Chinese characters. The FYP24 group uses this in place of the authentic version used by the FYP23 group.
- `all`: The whole Lantingji Xu dataset, based on `lantingjixu_text.txt`, placed under the `ContentImage`/`TargetImage` subdirectories according to the training data file tree used by FYP24 models. In the training context where the whole LTJX dataset is unseen, the full dataset is used for testing by the FYP24 group.
- `train`: The train set, subset of the full dataset. In the training context where the LTJX dataset is used for finetuning, the train/test split is used.
    - `lantingjixu_train.txt`: A list of 169 words in the train set, which is an unordered subset of `lantingjixu_text.txt`.
- `test`: The test set, subset of the full dataset.
    - `lantingjixu_test.txt`: A list of 36 words in the test set, which is an unordered subset of `lantingjixu_text.txt`.

## 📺 Sampling

For preparation of model checkpoints and usage of shell scripts, refer to [FontDiffuser#Sampling](./FontDiffuser.md#-sampling), except for the changes listed in the next section.

In addition to the shell scripts, we provide more python scripts to trigger the generate image process. The configurations are located at the start of the main function.

**(1) Generate images using text from a text file.**
```bash
python lantingjixu_sample.py
```

**(2) Print generated images onto a grid as if writing on a paper.**
```bash
python lantingjixu_grid.py
```

### Parameters and Added Features

All sampling parameters can be found in `configs/fontdiffuser.py` (common parameters for training and sampling) and `sample.py > arg_parse()` (parameters for sampling only).

**Changed Parameters**

- ~~**Original**: `style_image_path`: The style/reference image path.~~
- **New**: `style_image_path`: For one-shot methods, the style/reference image path. For few-shot methods, the directory with style/reference images.

**New Parameters**

We did not add any new sampling parameter.

## 📐 Evaluation

We provide python scripts for evaluating a model with FID, SSIM, LPIPS, and L1. The configurations are located at the start of the main function.

**(1) Run the whole evaluation process**
```bash
python lantingjixu_eval.py
```

This evaluation script runs the whole process of our evaluation method. A directory D of ground truth images of a style is used, e.g. the Lantingji Xu characters. Let's say we want to run R rounds, the directory D has C characters, and the model uses K style images (reference images) for generation. The script detects and loads the specified test profile used for evaluation. If not exist, the script generates one. A test profile is a directory with R test files named `test_INDEX.yaml` where INDEX is a number starting at 0. Each test file consists of a seed (randomly chosen) and a list of C test cases, one for each character in D. Each test case then has 1 character image (the target image) and K style images (randomly sampled). The content image of a test case is generated from the character extracted from target image name. The script will then run the evaluation process according to the test profile, reporting the round performances, their mean and SD, and overall performance in the output `eval_results.yaml`. By using the same test profile, the same setting can be used to evaluate every model.

**(2) Evaluate generated images with ground truth images by specifying folders**
```bash
python lantingjixu_eval_by_folder.py
```

This is an older method for evaluation by specifying folders to generated images and ground truth images. You would first generate the images using `lantingjixu_sample.py`, then evaluate using `lantingjixu_eval_by_folder.py`.

## 🏋️ Training

Our work is focused on finetuning the original FontDiffuser model from FontDiffuser authors, completed with Phase 1 and 2 training. The training we add starts with Phase 3. Scripts in the `scripts/` directory contain the configuration we used for the training we added.

For Phase 1 training, Phase 2 training, and data construction, refer to [FontDiffuser#Training](./FontDiffuser.md#️-training), except for the changes listed in the next section.

### Parameters and Added Features

All training parameters can be found in `configs/fontdiffuser.py` (common parameters for training and sampling).

**Changed Paramters**

- ~~**Original**: `phase_2`: Tag to phase 2 training.~~
- **New**: `training_phase`: The training phase number.
- ~~**Original**: `phase_1_ckpt_dir`: The model checkpoints saving directory after phase 1 training.~~
- **New**: `last_phase_ckpt_dir`: The model checkpoints saving directory after the last phase's training.

**New Paramters**

New parameters to support in-sample validation:
- `use_validation`: Whether to run validation during training. If true, it will compute validation losses with the following settings.
- `validation_factor`: The factor of validation data (1/factor of data is split for validation).
- `validation_batch_size`: Batch size (per device) for the validation dataloader.
- `validation_interval`: The interval for validation.

Validation is only useful to see whether in-sample overfitting occurs. This validation does not affect training result and is seldom useful in practice. Also, validation can consume a long time for large datasets. You may disable validation by removing the `use_validation` parameter (or lower the validation factor if you prefer).

New parameters to support resume training:
- `resume_training`: Whether this training is a resumption of a training in the past. If true, the global step value and model/optimizr/scheduler states will inherit from saved values in `whole_model.pth` retrieved from `resume_ckpt_dir`.
- `resume_ckpt_dir`: The directory of the model checkpoints to resume training.

New parameters to few-shot generation (not available in `main`, which is a one-shot method):
- `k_shot`: The maximum number of style images used.

### Data Construction Changes
Our file structure for training data augments the original file structure by allowing training on multiple images of the same character through the use of sequence identifiers, as shown below. A sequence identifier is a string that follows a `+` and is used to uniquely identify an image within a directory.
```
├──data_examples
│   └── train
│       ├── ContentImage
│       │   ├── char0.png
│       │   ├── char1.png
│       │   └── ...
│       └── TargetImage
│           ├── style0
│           │     ├──style0+char0.png    <-- Without sequence identifier
│           │     ├──style0+char0+1.png  <-- With sequence identifier
│           │     ├──style0+char0+2.png
│           │     └── ...
│           ├── style1
│           │     ├──style1+char0.png
│           │     ├──style1+char1.png
│           │     └── ...
│           └── ...
```

## 📱 Web UI

The FontDiffuser authors offer a Web UI for demonstration of their work. For usage, refer to [FontDiffuser#Run Web UI](./FontDiffuser.md#-run-webui)

However, it has not been adapted for the few-shot methods, so it only works on one-shot methods (`main`).

## 💙 Acknowledgement
- [FontDiffuser](https://github.com/yeungchenwa/FontDiffuser)
- [Diffusers](https://github.com/huggingface/diffusers)
