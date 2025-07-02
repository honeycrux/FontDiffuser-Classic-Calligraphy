# FontDiffuser For Classic Calligraphy: A Study On FontDiffuser's Capability To Generate Calligraphy

This document is provided by the FYP24 project group.

## 🌟 Introduction

This project is derived from "FontDiffuser: One-Shot Font Generation via Denoising Diffusion with Multi-Scale Content Aggregation and Style Contrastive Learning" by Yang et al. Consequently, this page contains numerous references to the README page of the FontDiffuser project by its original authors, which is included in [FontDiffuser.md](./FontDiffuser.md).

This project explores ways to modify FontDiffuser to generate Chinese Calligraphy. We approached this problem by changing the model from one-shot to few-shot, allowing the model to infer the style from multiple samples from the target distribution.

This project currently has 5 maintained branches using different proposed methods as described below.

One shot methods:
- `main`: The original FontDiffuser model with our code enhancements.

Few-shot methods:
- `release/naive-few-shot`: The Naive Few-shot method, which takes an average of the features of style samples to infer encodings of a single style.
- `release/conv-few-shot`: The Convolution Few-shot method, which uses a combination of linear and convolutional layers on the features of style samples to infer encodings of a single style.
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

- **Coming soon**: The data preparation scripts are released.
- **Coming soon**: The models are released.
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

## 🔥 Models

In model training, we produce two types of models:
1. **General Calligraphy Model**: The objective is to generate authentic calligraphy given any calligraphy work.
2. **Single-style Calligraphy Models**: The objective is to generate one style only.

(Downloads coming soon)

## 🔥 Dataset Preparation Scripts

(Coming soon)

## 🔥 The Lantingji Xu Dataset

The Lantingji Xu dataset we used is available in the `data_lantingjixu` folder, with the following content:
- `lantingjixu_title.txt`: The title of Lantingji Xu.
- `lantingjixu_authentic.txt`: The authentic transcription of Lantingji Xu.
- `lantingjixu_text.txt`: The transcription of Lantingji Xu with obscure characters substituted with more commonly-used Chinese characters. We use this in place of the authentic version.
- `all`: The whole Lantingji Xu dataset, based on `lantingjixu_text.txt`, placed under the `ContentImage`/`TargetImage` subdirectories according to the training data file tree standard.
- `train`: The train set, subset of the Lantingji Xu dataset.
    - `lantingjixu_train.txt`: The characters in the train set, which is an unordered subset of `lantingjixu_text.txt`.
- `eval`: The eval set, subset of the Lantingji Xu dataset.
    - `lantingjixu_eval.txt`: The characters in the eval set, which is an unordered subset of `lantingjixu_text.txt`.
- `legacy`: Files not used by us but are used by FYP23 project group. These files are kept for reference.
    - `strokelist.txt`: [The Diff-Font stroke list](https://github.com/HensonChen/Diff-font/blob/main/traditional_chinese_stroke.txt) consisting of stroke information of 3000 Chinese characters.
    - `wordlist.txt`: A list of 3000 words appearing in the Diff-Font stroke list used by the FYP23 project group. There are 169 unique words in Lantingji Xu that appear in this list. Subsequently, the 169 unique words are chosen to be the train set. The other 40 words are chosen to be the eval set. We inherited their choice of train-eval split (Minor differences: We switched to `lantingjixu_text.txt` instead of the authentic version that they use. As a result, our split by character count is actually 169-36 instead of 169-40. Additionally, they strictly use unique characters in training, but we allow images of the same characters. As a result, our split by image count is 255-50.)
    - `lacklist.txt`: A list of 40 words appearing in`lantingjixu_authentic.txt` but not in `wordlist.txt`.
    - `lacklist_strokes.txt`: Stroke information of the 40 words from `lacklist.txt`.

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

All sampling parameters can be found in `configs/fontdiffuser.py` (common parameters for training and sampling) and `sample.py > arg_parse()` (specific parameters for sampling).

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

Our work is focused on finetuning the original FontDiffuser model from FontDiffuser authors, completed with Phase 1 and 2 training. The training we add assumes phase numbers starting from 3.

For Phase 1 training, Phase 2 training, and data construction, refer to [FontDiffuser#Training](./FontDiffuser.md#️-training), except for the changes listed in the next section.

### Parameters and Added Features

All training parameters can be found in `configs/fontdiffuser.py` (common parameters for training and sampling).

**Changed Paramters**

- ~~**Original**: `phase_2`: Tag to phase 2 training.~~
- **New**: `training_phase`: The training phase number.
- ~~**Original**: `phase_1_ckpt_dir`: The model checkpoints saving directory after phase 1 training.~~
- **New**: `last_phase_ckpt_dir`: The model checkpoints saving directory after the last phase's training.

**New Paramters**

New parameters to support validation:
- `use_validation`: Whether to run validation during training. If true, it will compute validation losses with the following settings.
- `validation_factor`: The factor of validation data (1/factor of data is split for validation).
- `validation_batch_size`: Batch size (per device) for the validation dataloader.
- `validation_interval`: The interval for validation.

New parameters to support resume training:
- `resume_training`: Whether this training is a resumption of a training in the past. If true, the global step value and model/optimizr/scheduler states will inherit from saved values in `whole_model.pth` retrieved from `resume_ckpt_dir`.
- `resume_ckpt_dir`: The directory of the model checkpoints to resume training.

New parameters to few-shot generation (not available in `main`, which is a one-shot method):
- `k_shot`: The maximum number of style images used.

## 📱 Web UI

The FontDiffuser authors offer a Web UI for demonstration of their work.

However, it has not been adapted for the few-shot methods, so it only works on one-shot methods (`main`).

## 💙 Acknowledgement
- [FontDiffuser](https://github.com/yeungchenwa/FontDiffuser)
- [Diffusers](https://github.com/huggingface/diffusers)
