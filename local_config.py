"""
 @file: local_config.py
 @Time    : 2023/1/10
 @Author  : Peinuan qin
 """
import argparse


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", default="local", type=str)

    parser.add_argument("--requires_grad", default=False, type=bool)
    parser.add_argument("--data_root", default="./data", type=str)
    parser.add_argument("--content_dir_name", default="content_images", type=str)
    parser.add_argument("--style_dir_name", default="style_images", type=str)
    parser.add_argument("--output_dir_name", default="output_images", type=str)

    parser.add_argument("--use_relu", default=True, type=bool)

    parser.add_argument("--content_img_name", default="golden_gate.jpg", type=str)
    parser.add_argument("--style_img_name", default="vg_houses.jpg", type=str)

    parser.add_argument("--img_height", default=400, type=int)

    parser.add_argument("--content_weight", type=float, help="weight factor for content loss", default=1e5)
    parser.add_argument("--style_weight", type=float, help="weight factor for style loss", default=3e4)
    parser.add_argument("--tv_weight", type=float, help="weight factor for total variation loss", default=1e0)

    parser.add_argument("--optimizer", type=str, choices=['lbfgs', 'adam'], default='lbfgs')
    # parser.add_argument("--optimizer", type=str, choices=['lbfgs', 'adam'], default='adam')
    parser.add_argument("--model", type=str, choices=['vgg16', 'vgg19'], default='vgg19')
    parser.add_argument("--init_method", type=str, choices=['random', 'content', 'style'], default='content')
    parser.add_argument("--saving_freq", type=int,
                        help="saving frequency for intermediate images (-1 means only final)", default=100)
    parser.add_argument("--img_format", default=(4, ".jpg"), type=tuple)
    args = parser.parse_known_args()[0]
    return args
