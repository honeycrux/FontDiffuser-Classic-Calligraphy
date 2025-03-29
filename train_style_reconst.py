# This script is provided by the FYP24 project group.
# This script is for training the style reconstructor, without the other parts.
# For usage, also refer to scripts/train_phase_*.sh.

import os
import math
import time
import logging
from tqdm.auto import tqdm

import torch
import torch.utils.data
import torch.nn.functional as F

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler

from dataset.font_dataset import FontDataset
from dataset.collate_fn import CollateFN
from configs.fontdiffuser import get_parser
from src import (
    build_style_encoder,
    build_content_encoder,
    build_style_reconstructor,
)
from utils import (
    save_args_to_yaml,
    get_transform_function,
)


logger = get_logger(__name__)

def get_local_time():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

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

    logging_dir = f"{args.output_dir}/{args.logging_dir}"

    # Prepare accelerator
    accelerator_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        kwargs_handlers=[accelerator_kwargs],
    )

    # Prepare logging
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()
    
    logging.basicConfig(
        filename=f"{args.output_dir}/fontdiffuser_training.log",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    # Set training seed
    if args.seed is not None:
        set_seed(args.seed)

    # Load model
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)
    model = build_style_reconstructor(args=args)

    assert isinstance(args.last_phase_ckpt_dir, str) and os.path.exists(args.last_phase_ckpt_dir), f"Expect the last phase checkpoint directory exists, but got {args.last_phase_ckpt_dir}"
    style_encoder.load_state_dict(torch.load(f"{args.last_phase_ckpt_dir}/style_encoder.pth"))
    content_encoder.load_state_dict(torch.load(f"{args.last_phase_ckpt_dir}/content_encoder.pth"))

    # If necessary, freeze corresponding model parameters
    style_encoder.requires_grad_(False)
    content_encoder.requires_grad_(False)

    # Load transform functions
    content_transforms = get_transform_function(target_size=args.content_image_size, normalize=True)
    style_transforms = get_transform_function(target_size=args.style_image_size, normalize=True)
    target_transforms = get_transform_function(target_size=(args.resolution, args.resolution), normalize=True)

    # Load training dataset
    train_dataset = FontDataset(
        args=args,
        phase='train', 
        transforms=[
            content_transforms, 
            style_transforms, 
            target_transforms,
        ],
        is_validation_mode=False,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=args.train_batch_size, collate_fn=CollateFN())

    # Load validation dataset
    validation_dataset = None
    validation_dataloader = None
    if args.use_validation:
        validation_dataset = FontDataset(
            args=args,
            phase='train', 
            transforms=[
                content_transforms, 
                style_transforms, 
                target_transforms,
            ],
            is_validation_mode=True,
        )
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset, shuffle=False, batch_size=args.validation_batch_size, collate_fn=CollateFN())
    
    # Build optimizer and learning rate
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

    # Initialize global step
    global_step = 0

    # Load the model for resume training
    if args.resume_training:
        assert isinstance(args.resume_ckpt_dir, str) and os.path.exists(args.resume_ckpt_dir), f"Expect the resume checkpoint directory exists, but got {args.resume_ckpt_dir}"
        print(f"Resuming training from {args.resume_ckpt_dir}")
        whole_model = torch.load(f"{args.resume_ckpt_dir}/whole_model.pth")
        model.load_state_dict(whole_model["model"])
        optimizer.load_state_dict(whole_model["optimizer"])
        lr_scheduler.load_state_dict(whole_model["lr_scheduler"])
        global_step = whole_model["global_step"]
        logging.info(f"[{get_local_time()}] Resume training from global step {global_step}")
    else:
        print("Starting new training")

    # Accelerate preparation
    model, optimizer, train_dataloader, validation_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, validation_dataloader, lr_scheduler)
    ## Move external modules to target deivce
    style_encoder = style_encoder.to(accelerator.device)
    content_encoder = content_encoder.to(accelerator.device)

    # Initialize trackers automatically on the main process
    if accelerator.is_main_process:
        accelerator.init_trackers(args.experience_name)
        save_args_to_yaml(args=args, output_file=f"{args.output_dir}/{args.experience_name}_config.yaml")

    # Prepare progress bar
    # Only show the progress bar once on each machine
    progress_bar = tqdm(
        initial=global_step,
        total=args.max_train_steps,
        disable=not accelerator.is_local_main_process,
        desc="Steps",
        position=0,
    )

    # Compute training numbers (convert training steps to epochs, etc.)
    # PyTorch: len(dataloader)/num_batches is the max number of batches that can fit into len(dataset)
    # Accelerator: Gradient accum must fit into an epoch (and the final accum may have fewer steps)
    # i.e. num_update_steps = ceil(num_batches / gradient_accumulation_steps)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    num_validation_steps = len(validation_dataloader) if validation_dataloader is not None else None

    def compute_loss(samples):
        content_images = samples["content_image"]
        style_images = samples["style_image"]
        target_images = samples["target_image"]

        # Varying number of styles during training: Pick only L <= K styles
        _, K, _, _, _ = style_images.shape
        L = torch.randint(1, K + 1, ())
        style_images = style_images[:, :L, ...] # shape: (B, L, C, H, W)

        # Simulate the preparation process in FontDiffuserModel

        # Part I: Get style and content features from style and content images

        ### Initialization
        style_batch = style_images

        ### Get style feature from style image *list*
        # style_batch are in the shape of (N, K, C, H, W)
        style_style_feature_list = []
        for style_batch_item in style_batch:
            style_style_feature, _, _ = style_encoder(style_batch_item)
            style_style_feature_list.append(style_style_feature)
        style_style_feature_batch = torch.stack(style_style_feature_list)

        ### Get content feature from content image
        content_content_feture, content_content_residual_features = content_encoder(content_images)
        content_content_residual_features.append(content_content_feture)

        ### Get content feature from style image *list*
        style_content_residual_features_batch_transpose = []
        for style_batch_item in style_batch:
            style_content_feature, style_content_residual_features = content_encoder(style_batch_item)
            style_content_residual_features.append(style_content_feature)
            style_content_residual_features_batch_transpose.append(style_content_residual_features)
        style_content_residual_features_batch = []
        for fs_idx in range(len(style_content_residual_features_batch_transpose[0])):
            # stack Fs_i columns
            Fs_i = [Ic_i[fs_idx] for Ic_i in style_content_residual_features_batch_transpose]
            style_content_residual_features_batch.append(torch.stack(Fs_i))

        # Part II: infer *one* style_style_feature from K of them
        # and infer *one* style_content_residual_features from K of them

        style_feature_pred = model(
            style_style_feature=style_style_feature_batch,
            style_content_residual_features=style_content_residual_features_batch,
            content_content_residual_features=content_content_residual_features,
        )

        # Part III: Compute loss

        style_feature_real, _, _ = style_encoder(target_images)
        style_reconstruction_loss = F.mse_loss(style_feature_pred.float(), style_feature_real.float(), reduction="mean")

        loss = style_reconstruction_loss

        return loss

    # Training loop
    for epoch in range(num_train_epochs):
        # Accumulated train loss in a global step, which may include multiple gradient accumulation steps
        acc_train_loss = [] # distributed loss (average across all processes)
        acc_local_train_loss = [] # local loss (only for the current process)

        for step, samples in enumerate(train_dataloader):
            # Training
            model.train()
            with accelerator.accumulate(model):
                ## Forward pass
                loss = compute_loss(samples)
                acc_local_train_loss.append(loss.item())

                ## Gather the losses across all processes
                distributed_losses = accelerator.gather(loss.repeat(args.train_batch_size))
                assert isinstance(distributed_losses, torch.Tensor)
                distributed_loss = distributed_losses.mean()
                acc_train_loss.append(distributed_loss.item())

                ## Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            is_on_global_step = accelerator.sync_gradients # the accelerator has performed an optimization step behind the scenes
            is_on_main_process = accelerator.is_main_process
            process_idx = accelerator.process_index

            # Log to progress bar
            if is_on_main_process:
                last_lr = lr_scheduler.get_last_lr()[0]
                logs = {"step_loss": distributed_loss.detach().item(), "lr": last_lr}
                progress_bar.set_postfix(**logs)

            # Update progress bar and global step states
            if is_on_global_step:
                progress_bar.update(1)
                global_step += 1

            is_last_step = global_step >= args.max_train_steps
            is_logging_step = is_on_global_step and (is_last_step or global_step % args.log_interval == 0)
            is_checkpoint_step = is_on_global_step and (is_last_step or global_step % args.ckpt_interval == 0)
            is_validation_step =  is_on_global_step and args.use_validation and (is_last_step or global_step % args.validation_interval == 0)

            # Compute and log loss values
            if is_on_global_step:
                avg_train_loss = sum(acc_train_loss) / len(acc_train_loss)
                avg_local_train_loss = sum(acc_local_train_loss) / len(acc_local_train_loss)

                ## Log information to tensorboard
                accelerator.log({"train_loss": avg_train_loss}, step=global_step)

                ## Log information to file
                if is_logging_step and is_on_main_process:
                    logging.info(f"[{get_local_time()}] Global Step {global_step} => avg_train_loss = {avg_train_loss}")

                ## Wait for everyone
                accelerator.wait_for_everyone()

                ## Log information to file for each process
                if is_logging_step:
                    logging.info(f"[{get_local_time()}] Proc {process_idx}: Global Step {global_step} => acc_local_train_loss = {acc_local_train_loss}")

                ## Reset the accumulated loss states
                acc_train_loss = []
                acc_local_train_loss = []

            # Wait for everyone
            accelerator.wait_for_everyone()

            # Save checkpoint
            if is_checkpoint_step and is_on_main_process:
                save_dir = f"{args.output_dir}/global_step_{global_step}"
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), f"{save_dir}/style_reconstructor.pth")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "global_step": global_step,
                }, f"{save_dir}/whole_model.pth")
                logging.info(f"[{get_local_time()}] Save the checkpoint on global step {global_step}")
                progress_bar.write("Save the checkpoint on global step {}".format(global_step))

            # Validation
            if is_validation_step:
                assert validation_dataloader is not None
                assert num_validation_steps is not None
                if is_on_main_process:
                    progress_bar.write(f"Computing validation loss on global step {global_step}")

                ## Initialize validation loss states
                all_val_losses: list[torch.Tensor] = []

                ## Prepare validation progress bar
                val_progress_bar = tqdm(
                    validation_dataloader,
                    total=num_validation_steps,
                    disable=not accelerator.is_local_main_process,
                    desc="Validation",
                    leave=False,
                )

                model.eval()
                for val_step, val_samples in enumerate(val_progress_bar):
                    ## Compute validation loss
                    with torch.no_grad():
                        val_loss = compute_loss(val_samples)

                    ## Gather the losses across all processes
                    distributed_val_losses = accelerator.gather_for_metrics(val_loss)
                    assert isinstance(distributed_val_losses, torch.Tensor)
                    distributed_val_loss = distributed_val_losses.mean()
                    all_val_losses.append(distributed_val_loss)

                    ## Log to validation progress bar
                    if is_on_main_process:
                        val_logs = {"val_loss": distributed_val_loss.detach().item()}
                        val_progress_bar.set_postfix(val_logs)

                ## Compute and log validation loss values
                if accelerator.is_main_process:
                    validation_loss = sum(all_val_losses) / len(all_val_losses)
                    progress_bar.write(f"Validation loss: {validation_loss}")
                    logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}] Global Step {global_step} => validation_loss = {validation_loss}")
                    accelerator.log({"validation_loss": validation_loss}, step=global_step)

            # Quit
            if global_step >= args.max_train_steps:
                break

    accelerator.end_training()

if __name__ == "__main__":
    main()
