"""Defines argument parser for Detectron2."""

from email.policy import default
from configargparse import Parser


def parse_args():
    """Parses command line arguments for Detectron2."""

    # Instantiates config arg parser with required config file.
    parser = Parser(
        args_for_setting_config_path=["-c", "--cfg", "--config"],
        config_arg_is_required=True,
    )
    parser.add(
        "--transformer",
        action="store_true",
        help="setups transformer training",
    ) 
    parser.add(
        "--resume",
        action="store_true",
        help="setups transformer training",
    )    
    # Sets input arguments.
    parser.add(
        "--train_dirs",
        nargs="+",
        metavar="DIR",
        help="Paths to train dirs",
    )
    parser.add(
        "--train_jsons",
        nargs="+",
        metavar="FILE",
        help="Paths to train annotations (COCO jsons)",
    )
    parser.add(
        "--test_dir",
        nargs="+",
        metavar="DIR",
        help="Path to test dir",
    )
    parser.add(
        "--test_json",
        nargs="+",
        metavar="FILE",
        help="Path to test annotations (COCO json)",
    )
    parser.add(
        "--output_dir",
        metavar="DIR",
        help="Path to output dir",
    )

    # Sets task arguments.
    parser.add(
        "--eval",
        action="store_true",
        help="Activates evaluation mode (skips training)",
    )
    parser.add(
        "--predict",
        action="store_true",
        help="Predicts bounding boxes on labeled data",
    )
    parser.add(
        "--unlabled_predict",
        action="store_true",
        help="Predicts bounding boxes on unlabeled data",
    )
    parser.add(
        "--visualize",
        action="store_true",
        help="gradcam/tsne viz on labeled data",
    )
    parser.add(
        "--cm",
        action="store_true",
        help="confusion matrix",
    )
    parser.add(
        "--tsne",
        action="store_true",
        help="tsne viz on labeled data",
    )
    parser.add(
        "--grad_cam",
        metavar="NUM",
        type=int,
        help="gradcam, if -1 do not run.",
        default=-1,
    )

    parser.add(
        "--synth_grad_cam",
        metavar="NUM",
        type=int,
        help="synth gradcam, if -1 do not run.",
        default=-1,
    )

    parser.add(
        "--guided_gradcam",
        metavar="NUM",
        type=int,
        help="guided_gradcam, if -1 do not run.",
        default=-1,
    )
    parser.add(
        "--specific_image",
        nargs = "+",
        default=[],
    )
    parser.add(
        "--deep_dream",
        action="store_true",
        help="deep_dream?",
    )
    parser.add(
        "--score_cam",
        action="store_true",
        help="score_cam?",
    )
    parser.add(
        "--lrp",
        action="store_true",
        help="score_cam?",
    )
    parser.add(
        "--drise",
        action="store_true",
        help="drise?",
    )

    parser.add(
        "--layer_name",
        metavar="LAYER_NAME",
        type=str,
        help="layer of gradcam to visualize",
        default = "backbone.bottom_up.res5.2.conv3",
    )
    parser.add(
        "--train",
        action="store_true",
        help="Predicts bounding boxes on unlabeled data",
    )
    # Sets GPU arguments.
    parser.add(
        "--gpus",
        metavar="NUM",
        type=int,
        help="Number of GPUs for distributed training",
        default = 1,
    )

    # Sets model arguments.
    parser.add(
        "--classes",
        metavar="NUM",
        type=int,
        help="Number of classes (not including background)",
    )
    parser.add(
        "--proposals",
        metavar="NUM",
        type=int,
        help="Number of object proposals per image",
    )
    parser.add(
        "-w",
        "--weights",
        metavar="FILE",
        type=str,
        help="Weights path (pretrained for training, or trained for testing)",
    )
    parser.add(
        "--workers",
        default=4,
        metavar="NUM",
        type=int,
        help="Number of worker threads",
    )
    
    # Sets implementation arguments.
    parser.add(
        "--lr",
        "--learning-rate",
        dest="lr",
        metavar="LR",
        type=float,
        help="Initial learning rate",
    )
    parser.add(
        "--iters",
        type=int,
        help="Number of iterations to train the model",
    )
    parser.add(
        "-b",
        "--batch_size",
        metavar="SIZE",
        type=int,
        help="Number of images per batch",
        default=8,
    )
    parser.add(
        "--scheduler",
        choices=("WarmupMultiStepLR", "WarmupCosineLR"),
        type=str,
        help="Learning rate scheduler",
    )
    parser.add(
        "--steps",
        nargs="*",
        type=int,
        help="The iterations to decrease the learning rate at" \
             " when using the WarmupMultiStepLR scheduler",
        )
    parser.add(
        "--gamma",
        type=float,
        help="Factor to decrease learning rate by at each step" \
             " when using the WarmupMultiStepLR scheduler",
    )
    parser.add(
        "--warmup_method",
        choices=["constant", "linear"],
        type=str,
        help="Learning rate warmup method",
    )
    parser.add(
        "--warmup_factor",
        metavar="FACTOR",
        type=float,
        help="Factor to scale learning rate by during warmup",
    )
    parser.add(
        "--warmup_iters",
        metavar="ITERS",
        type=int,
        help="Number of iterations for learning rate warmup",
    )

    # Parses arguments.
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    parse_args()

