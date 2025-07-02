# This script is provided by authors of FontDiffuser.
# This script is the configuration file for command line arguments available in FontDiffuser.
# Also note that additional arguments for sampling are added in sample.py::arg_parse().

import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="Training config for FontDiffuser.")
    ################# Experience #################
    parser.add_argument(
        "--seed", type=int, default=123, help="A seed for reproducible training."
    )
    parser.add_argument("--experience_name", type=str, default="fontdiffuser_training")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="The font dataset root path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    # Model
    parser.add_argument(
        "--resolution",
        type=int,
        default=96,
        help="The resolution for input images, all the images in the train/validation \
                            dataset will be resized to this.",
    )
    parser.add_argument(
        "--unet_channels",
        type=tuple,
        default=(64, 128, 256, 512),
        help="The channels of the UNet.",
    )
    parser.add_argument(
        "--style_image_size", type=int, default=96, help="The size of style images."
    )
    parser.add_argument(
        "--content_image_size", type=int, default=96, help="The size of content images."
    )
    parser.add_argument(
        "--content_encoder_downsample_size",
        type=int,
        default=3,
        help="The downsample size of the content encoder.",
    )
    parser.add_argument(
        "--channel_attn",
        type=bool,
        default=True,
        help="Whether to use the se attention.",
    )
    parser.add_argument(
        "--content_start_channel",
        type=int,
        default=64,
        help="The channels of the fisrt layer output of content encoder.",
    )
    parser.add_argument(
        "--style_start_channel",
        type=int,
        default=64,
        help="The channels of the fisrt layer output of content encoder.",
    )

    # Training
    parser.add_argument("--training_phase", type=int, help="The training phase number.")
    parser.add_argument(
        "--last_phase_ckpt_dir",
        type=str,
        default=None,
        help="The trained ckpt directory during the last phase.",
    )
    ## SCR
    parser.add_argument(
        "--use_scr", action="store_true", help="Whether to use the SCR."
    )
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--mode", type=str, default="refinement")
    parser.add_argument("--scr_image_size", type=int, default=96)
    parser.add_argument("--scr_ckpt_path", type=str, default=None)
    parser.add_argument(
        "--num_neg", type=int, default=16, help="Number of negative samples."
    )
    parser.add_argument("--nce_layers", type=str, default="0,1,2,3")
    parser.add_argument("--sc_coefficient", type=float, default=0.01)
    ## train batch size
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    ## loss coefficient
    parser.add_argument("--perceptual_coefficient", type=float, default=0.01)
    parser.add_argument("--offset_coefficient", type=float, default=0.5)
    ## step
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=440000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--ckpt_interval",
        type=int,
        default=40000,
        help="The checkpoint saving interval when training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--log_interval", type=int, default=100, help="The log interval of training."
    )
    ## learning rate
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="linear",
        help="The scheduler type to use. Choose between 'linear', 'cosine', \
                            'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10000,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    ## classifier-free
    parser.add_argument(
        "--drop_prob",
        type=float,
        default=0.1,
        help="The uncondition training drop out probability.",
    )
    ## scheduler
    parser.add_argument(
        "--beta_scheduler",
        type=str,
        default="scaled_linear",
        help="The beta scheduler for DDPM.",
    )
    ## optimizer
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires \
                            PyTorch >= 1.10. and an Nvidia Ampere GPU.",
    )
    ## validation
    parser.add_argument(
        "--use_validation",
        action="store_true",
        help="Whether to run validation during training.",
    )
    parser.add_argument(
        "--validation_factor",
        type=int,
        default=10,
        help="The factor of validation data (1/factor of data is split for validation).",
    )
    parser.add_argument(
        "--validation_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the validation dataloader.",
    )
    parser.add_argument(
        "--validation_interval",
        type=int,
        default=100,
        help="The interval of validation.",
    )
    ## resume training
    parser.add_argument(
        "--resume_training",
        action="store_true",
        help="Whether this training is a resumption of a training in the past.",
    )
    parser.add_argument(
        "--resume_ckpt_dir",
        type=str,
        default=None,
        help="The directory of the ckpt to resume training (requires the `whole_model.pth` file).",
    )

    # Sampling
    parser.add_argument(
        "--algorithm_type",
        type=str,
        default="dpmsolver++",
        help="Algorithm for sampleing.",
    )
    parser.add_argument(
        "--guidance_type",
        type=str,
        default="classifier-free",
        help="Guidance type of sampling.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale of the classifier-free mode.",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=20, help="Sampling step."
    )
    parser.add_argument(
        "--model_type", type=str, default="noise", help="model_type for sampling."
    )
    parser.add_argument(
        "--order", type=int, default=2, help="The order of the dpmsolver."
    )
    parser.add_argument(
        "--skip_type", type=str, default="time_uniform", help="Skip type of dpmsolver."
    )
    parser.add_argument(
        "--method", type=str, default="multistep", help="Multistep of dpmsolver."
    )
    parser.add_argument(
        "--correcting_x0_fn",
        type=str,
        default=None,
        help="correcting_x0_fn of dpmsolver.",
    )
    parser.add_argument(
        "--t_start", type=str, default=None, help="t_start of dpmsolver."
    )
    parser.add_argument("--t_end", type=str, default=None, help="t_end of dpmsolver.")

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    return parser
