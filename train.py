# This script is provided by authors of FontDiffuser.
# This script is the training process of FontDiffuser.
# For usage, also refer to scripts/train_phase_*.sh.

import os
import math
import time
import logging
from tqdm.auto import tqdm
import decimal

import torch
import torch.utils.data
import torch.nn.functional as F
from torchvision import transforms

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler

from dataset.font_dataset import FontDataset
from dataset.collate_fn import CollateFN
from configs.fontdiffuser import get_parser
from src import (FontDiffuserModel,
                 ContentPerceptualLoss,
                 build_unet,
                 build_style_encoder,
                 build_content_encoder,
                 build_ddpm_scheduler,
                 build_scr,
                 build_k_feature_extractor)
from utils import (save_args_to_yaml,
                   x0_from_epsilon, 
                   reNormalize_img, 
                   normalize_mean_std,
                   get_transform_function)


logger = get_logger(__name__)

def get_args():
    parser = get_parser()
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    return args


def main():

    args = get_args()

    use_scr = args.training_phase in [2,]
    load_basic_models = args.training_phase >= 2
    # freeze_basic_models = args.training_phase >= 3 # when only training our models (K-feature extractor)
    freeze_basic_models = False # when fine-tuning the whole model

    logging_dir = f"{args.output_dir}/{args.logging_dir}"

    accelerator_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        kwargs_handlers=[accelerator_kwargs])

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        filename=f"{args.output_dir}/fontdiffuser_training.log",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    # Set training seed
    if args.seed is not None:
        set_seed(args.seed)

    # Load model and noise_scheduler
    unet = build_unet(args=args)
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)
    k_feature_extractor = build_k_feature_extractor(args=args)
    noise_scheduler = build_ddpm_scheduler(args)

    if args.resume_training:
        assert args.resume_ckpt_dir is not None, "resume_traning is True, but resume_ckpt_dir is None."
        unet.load_state_dict(torch.load(f"{args.resume_ckpt_dir}/unet.pth"))
        style_encoder.load_state_dict(torch.load(f"{args.resume_ckpt_dir}/style_encoder.pth"))
        content_encoder.load_state_dict(torch.load(f"{args.resume_ckpt_dir}/content_encoder.pth"))
        k_feature_extractor.load_state_dict(torch.load(f"{args.resume_ckpt_dir}/k_feature_extractor.pth"))
    elif load_basic_models:
        assert args.last_phase_ckpt_dir is not None, "training requires basic models, but last_phase_ckpt_dir is None."
        unet.load_state_dict(torch.load(f"{args.last_phase_ckpt_dir}/unet.pth"))
        style_encoder.load_state_dict(torch.load(f"{args.last_phase_ckpt_dir}/style_encoder.pth"))
        content_encoder.load_state_dict(torch.load(f"{args.last_phase_ckpt_dir}/content_encoder.pth"))

    model = FontDiffuserModel(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder,
        k_feature_extractor=k_feature_extractor,)

    # Build content perceptaual Loss
    perceptual_loss = ContentPerceptualLoss()

    # If necessary, load SCR module for supervision
    if use_scr:
        scr = build_scr(args=args)
        scr.load_state_dict(torch.load(args.scr_ckpt_path))
        scr.requires_grad_(False)

    # If necessary, freeze corresponding model parameters to train the K-feature extractor
    if freeze_basic_models:
        unet.requires_grad_(False)
        style_encoder.requires_grad_(False)
        content_encoder.requires_grad_(False)

    # Load the datasets
    content_transforms = get_transform_function(args.content_image_size)
    style_transforms = get_transform_function(args.style_image_size)
    target_transforms = get_transform_function((args.resolution, args.resolution))
    train_font_dataset = FontDataset(
        args=args,
        phase='train', 
        transforms=[
            content_transforms, 
            style_transforms, 
            target_transforms],
        scr=use_scr,
        need_validation_split=True,
        is_validation_mode=False)
    train_dataloader = torch.utils.data.DataLoader(
        train_font_dataset, shuffle=True, batch_size=args.train_batch_size, collate_fn=CollateFN())
    validate_font_dataset = FontDataset(
        args=args,
        phase='train', 
        transforms=[
            content_transforms, 
            style_transforms, 
            target_transforms],
        scr=use_scr,
        need_validation_split=True,
        is_validation_mode=True)
    validate_dataloader = torch.utils.data.DataLoader(
        validate_font_dataset, shuffle=True, batch_size=args.validate_batch_size, collate_fn=CollateFN())

    # print(f"Train dataset size: {len(train_font_dataset)}")
    # print(f"Validation dataset size: {len(validate_font_dataset)}")

    # Build optimizer and learning rate
    if args.resume_training and args.resume_learning_rate is not None:
        args.learning_rate = args.resume_learning_rate
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,)

    # Accelerate preparation
    model, optimizer, train_dataloader, validate_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, validate_dataloader, lr_scheduler)
    ## move scr module to the target deivces
    if use_scr:
        scr = scr.to(accelerator.device)

    def compute_loss(samples):
        content_images = samples["content_image"]
        style_images = samples["style_images"]
        target_images = samples["target_image"]
        nonorm_target_images = samples["nonorm_target_image"]

        # Sample noise that we'll add to the samples
        noise = torch.randn_like(target_images)
        bsz = target_images.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=target_images.device)
        timesteps = timesteps.long()

        # Add noise to the target_images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_target_images = noise_scheduler.add_noise(target_images, noise, timesteps)

        # Classifier-free training strategy
        context_mask = torch.bernoulli(torch.zeros(bsz) + args.drop_prob)
        for i, mask_value in enumerate(context_mask):
            if mask_value==1:
                content_images[i, :, :, :] = 1 # [N, C, H, W]
                style_images[i, :, :, :, :] = 1 # k-shot: [N, K, C, H, W]

        # Predict the noise residual and compute loss
        noise_pred, offset_out_sum = model(
            x_t=noisy_target_images, 
            timesteps=timesteps, 
            style_images=style_images,
            content_images=content_images,
            content_encoder_downsample_size=args.content_encoder_downsample_size)
        diff_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        offset_loss = offset_out_sum / 2
        
        # output processing for content perceptual loss
        pred_original_sample_norm = x0_from_epsilon(
            scheduler=noise_scheduler,
            noise_pred=noise_pred,
            x_t=noisy_target_images,
            timesteps=timesteps)
        pred_original_sample = reNormalize_img(pred_original_sample_norm)
        norm_pred_ori = normalize_mean_std(pred_original_sample)
        norm_target_ori = normalize_mean_std(nonorm_target_images)
        percep_loss = perceptual_loss.calculate_loss(
            generated_images=norm_pred_ori,
            target_images=norm_target_ori,
            device=target_images.device)
        
        loss = diff_loss + \
                args.perceptual_coefficient * percep_loss + \
                    args.offset_coefficient * offset_loss
        
        if use_scr:
            neg_images = samples["neg_images"]
            # sc loss
            sample_style_embeddings, pos_style_embeddings, neg_style_embeddings = scr(
                pred_original_sample_norm, 
                target_images, 
                neg_images, 
                nce_layers=args.nce_layers)
            sc_loss = scr.calculate_nce_loss(
                sample_s=sample_style_embeddings,
                pos_s=pos_style_embeddings,
                neg_s=neg_style_embeddings)
            loss += args.sc_coefficient * sc_loss

        return loss

    def get_submodel(model, submodule_name):
        # If the model is wrapped with DDP, we need to access the submodule with model.module
        if hasattr(model, "module"):
            return getattr(model.module.config, submodule_name)
        # If the model is not wrapped with DDP, we can access the submodule directly
        return getattr(model.config, submodule_name)

    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.experience_name)
        save_args_to_yaml(args=args, output_file=f"{args.output_dir}/{args.experience_name}_config.yaml")

    # Convert to the training epoch
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Count global step
    global_step = 0
    if args.resume_training and args.resume_step is not None:
        global_step = args.resume_step

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(initial=global_step, total=args.max_train_steps, disable=not accelerator.is_local_main_process, desc="Train steps", position=0)

    for epoch in range(num_train_epochs):
        train_loss = []
        for step, samples in enumerate(train_dataloader):
            model.train()
            with accelerator.accumulate(model):
                loss = compute_loss(samples)

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss.append(avg_loss.item())

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            is_on_global_step = accelerator.sync_gradients

            # Checks if the accelerator has performed an optimization step behind the scenes
            if is_on_global_step:
                global_step += 1
                train_loss_value = sum(train_loss) / len(train_loss)
                # progress_bar.write(f"Proc: {accelerator.process_index} Global Step: {global_step}, Train Loss: {train_loss_value}, Train Loss Size: {len(train_loss)}")
                accelerator.log({"train_loss": train_loss_value}, step=global_step)
                train_loss = []

            # Log progress for all processes
            last_lr = lr_scheduler.get_last_lr()[0]
            logs = {"step_loss": loss.detach().item(), "lr": last_lr}
            progress_bar.set_postfix(**logs)

            step_idx = accelerator.process_index
            step_loss = loss.detach().item()
            step_idx = accelerator.gather_for_metrics((step_idx,))
            step_loss = accelerator.gather_for_metrics((step_loss,))

            accelerator.wait_for_everyone() # I added this as a precaution. Not sure if it is necessary

            if is_on_global_step and accelerator.is_main_process:
                # Log training loss
                precise_last_lr = decimal.Decimal.from_float(lr_scheduler.get_last_lr()[0])
                if global_step % args.validate_interval == 0 or global_step >= args.max_train_steps:
                    for step_idx_, step_loss_ in zip(step_idx, step_loss):
                        logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}] Proc {step_idx_}: Global Step {global_step} => train_loss = {step_loss_}, lr = {precise_last_lr}")

                # Save checkpoint
                if global_step % args.ckpt_interval == 0 or global_step >= args.max_train_steps:
                    save_dir = f"{args.output_dir}/global_step_{global_step}"
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(get_submodel(model, "unet").state_dict(), f"{save_dir}/unet.pth")
                    torch.save(get_submodel(model, "style_encoder").state_dict(), f"{save_dir}/style_encoder.pth")
                    torch.save(get_submodel(model, "content_encoder").state_dict(), f"{save_dir}/content_encoder.pth")
                    torch.save(get_submodel(model, "k_feature_extractor").state_dict(), f"{save_dir}/k_feature_extractor.pth")
                    torch.save(model, f"{save_dir}/total_model.pth")
                    logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}] Save the checkpoint on global step {global_step}")
                    progress_bar.write("Save the checkpoint on global step {}".format(global_step))

            if is_on_global_step:
                progress_bar.update(1)

            if is_on_global_step:
                # Do validation
                if global_step % args.validate_interval == 0 or global_step >= args.max_train_steps:
                    if accelerator.is_main_process:
                        progress_bar.write(f"Computing validation loss on global step {global_step}")

                    validation_losses = []

                    model.eval()
                    val_progress_bar = tqdm(validate_dataloader, desc="Validation", leave=False)
                    for val_step, val_samples in enumerate(val_progress_bar):
                        with torch.no_grad():
                            val_loss = compute_loss(val_samples)

                        val_logs = {"val_step": val_step, "val_loss": val_loss.detach().item()}
                        val_progress_bar.set_postfix(**val_logs)

                        val_loss = accelerator.gather_for_metrics(val_loss)

                        validation_losses.append(val_loss)

                    if accelerator.is_main_process:
                        validation_loss = torch.stack(validation_losses).mean().item()
                        progress_bar.write(f"Validation loss: {validation_loss}")
                        logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}] Global Step {global_step} => validation_loss = {validation_loss}")
                        accelerator.log({"validation_loss": validation_loss}, step=global_step)

            # Quit
            if global_step >= args.max_train_steps:
                break

    accelerator.end_training()

if __name__ == "__main__":
    main()
