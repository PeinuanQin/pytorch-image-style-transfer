"""
 @file: util.py
 @Time    : 2023/1/11
 @Author  : Peinuan qin
 """
import matplotlib.pyplot as plt
import numpy as np
import os.path
import cv2
import torch
from torch.autograd.variable import Variable
from torch.nn import MSELoss
from torch.optim import Adam, LBFGS
from torchvision import transforms
from vgg_models import VGG16, VGG19
from torchvision.models import vgg16, vgg19

IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]

def load_imgs(args, device):
    """
    load the content image, style image and generate opim image
    :param args:
    :param device:
    :return:
    """
    data_root = args.data_root
    content_dir_name = args.content_dir_name
    style_dir_name = args.style_dir_name

    content_img_name = args.content_img_name
    style_img_name = args.style_img_name

    output_dir_name = args.output_dir_name
    img_height = args.img_height

    content_img_path = os.path.join(data_root, content_dir_name, content_img_name)
    style_img_path = os.path.join(data_root, style_dir_name, style_img_name)

    # opencv read image
    def read_img(img_path, new_height, device):
        # opencv read B, G, R -> R, G, B
        img = cv2.imread(img_path)[:, :, ::-1]

        if new_height:
            # if new height is a tuple, the first position is width, second position is height
            if isinstance(new_height, tuple):
                img = cv2.resize(img, (new_height[1], new_height[0]), interpolation=cv2.INTER_CUBIC)

            # if new height only represents the height, the width would be resized according to the same ratio
            else:
                # current img height and width
                cur_height, cur_width = img.shape[:2]

                # get the resize ratio
                resize_ratio = new_height / cur_height

                # adjust the new width according to the resize ratio
                new_width = int(cur_width * resize_ratio)

                # resize the img
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # ensure the img data type is float32
        img = img.astype(np.float32)

        # pixel value is in [0,1]
        # img /= 255.

        # make img to tensor and do normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.mul(255)),
            transforms.Normalize(IMAGENET_MEAN_255, IMAGENET_STD_NEUTRAL)
        ])

        img = transform(img)

        # change the img to (batch, c, w, h) so that we can put it into the neural network
        batch_img = img.unsqueeze(0)

        # remove the image to device
        batch_img.to(device)
        return batch_img

    content_img = read_img(content_img_path, img_height, device)
    style_img = read_img(style_img_path, img_height, device)

    # make initial image
    if args.init_method == 'random':
        # use normal distribution for initialization
        init_img = np.random.normal(0, 90, content_img.shape).astype(np.float32)
        # remove the
        init_img = torch.from_numpy(init_img).float().to(device)

    elif args.init_method == 'content':
        # use the content img for initialization
        init_img = content_img

    else:
        # use the style image for initialization and keep it the same
        # width and height as the content img
        init_img = read_img(style_img_path, content_img.shape[2:], device)

    """
    take the init_img as trainable parameters, 
    the final task is to optimize this init_img
    """
    init_img = init_img.to(device)
    optimize_img = Variable(init_img, requires_grad=True)

    return content_img.to(device), style_img.to(device), optimize_img


def model_selection(args, device):
    """
    select the model according to args
    :param args:
    :param device:
    :return:
    """
    if args.model == 'vgg16':
        model = VGG16(args=args
                      , base_model=vgg16(True, True)
                      , out_num=4
                      , anchors=[4, 9, 16, 23]
                      , content_index=1
                      , style_indexs=[0, 1, 2, 3])

    elif args.model == 'vgg19':
        # the intermediate layer output are all activated by relu
        if args.use_relu:
            model = VGG19(args=args
                          , base_model=vgg19(True, True)
                          , out_num=6
                          , anchors=[2, 7, 12, 21, 22, 30]
                          , content_index=4
                          , style_indexs=[0, 1, 2, 3, 5])
        else:
            # only use the output of the conv layers, without the relu activation operation
            model = VGG19(args=args
                          , base_model=vgg19(True, True)
                          , out_num=6
                          , anchors=[1, 6, 11, 20, 21, 29]
                          , content_index=4
                          , style_indexs=[0, 1, 2, 3, 5])

    return model.to(device).eval()


def optimizer_selection(args, optim_img):
    """
    select an optimizer
    :param args:
    :param optim_img:
    :return:
    """
    if args.optimizer == 'adam':
        optim = Adam((optim_img,), lr=1e1)
        iterations = 3000
        return optim, iterations

    elif args.optimizer == 'lbfgs':
        iterations = 1000
        optim = LBFGS((optim_img,)
                      , max_iter=iterations
                      , line_search_fn='strong_wolfe')

        return optim, iterations


def calculate_loss(args
                   , model
                   , optim_img
                   , content_representation
                   , style_representations
                   ):
    """

    :param args:
    :param model:
    :param optim_img: the img with grad that need to be optimized
    :param content_representation: the conv layer output of content image in 'content_index' layer (single layer)
    :param style_representations: conv layer output of the style image in 'style_indexs' layers (multiple layers)
    :return:
    """
    # use the optim outputs to calculate style loss with style representations
    # and content loss with content representations
    content_criterion = MSELoss(reduction='mean')
    style_criterion = MSELoss(reduction='sum')

    optim_outputs = model(optim_img)
    content_index = model.content_index
    style_indexs = model.style_indexes

    optim_content_representation = optim_outputs[content_index].squeeze(axis=0)
    optim_style_outputs = [optim_outputs[i] for i in style_indexs]

    optim_style_representations = [get_grams(optim_style_output) for optim_style_output in optim_style_outputs]

    # since the content representation is only an output of one conv layer
    # so it is easy to be calculate

    content_loss = content_criterion(content_representation
                                     , optim_content_representation)

    # the style representations contain more than one conv layer output,
    # so we use "for" to cumulate all representation loss

    style_loss = 0.

    for i in range(len(style_representations)):
        gram = style_representations[i].squeeze(axis=0)
        optim_gram = optim_style_representations[i].squeeze(axis=0)
        style_loss += style_criterion(gram, optim_gram)

    style_loss /= len(style_representations)

    # total variation loss: make sure there is no abrupt change in neighbor pixel

    def total_variation(optim_img):
        """
        given the optimize image, using the neighbor pixels to calculate
        its smooth degree
        :param optim_img: the initial image that need to be optimized
        :return:
        """
        # the last 2 axis are h and w
        # optim_img[:, :, :, :-1] and optim_img[:, :, :, 1:] represents that
        # in the origin figure, the data differ by 1 position in the w-dim
        # optim_img[:, :, :-1, :] and optim_img[:, :, 1:, :] are the same operation
        # in the h-dim

        return torch.sum(torch.abs(optim_img[:, :, :, :-1] - optim_img[:, :, :, 1:])) + \
               torch.sum(torch.abs(optim_img[:, :, :-1, :] - optim_img[:, :, 1:, :]))

    total_variation_loss = total_variation(optim_img)

    # the weights that combine different part of losses
    alpha = args.content_weight
    beta = args.style_weight
    gamma = args.tv_weight

    total_loss = alpha * content_loss \
                 + beta * style_loss \
                 + gamma * total_variation_loss

    return total_loss, content_loss, style_loss, total_variation_loss


def get_grams(style_output, normalize=True):
    """
    give a conv layer output (b, c, h, w)
    calculate the similarity (dot) of each single feature map
    with other single feature map
    :param style_output:
    :param normalize:
    :return:
    """
    (b, c, h, w) = style_output.size()
    flatten_features = style_output.view(b, c, w * h)
    # (b, w*h, c)
    flatten_features_T = flatten_features.transpose(1, 2)
    # matrix multiply
    gram = flatten_features.bmm(flatten_features_T)
    if normalize:
        gram /= (c * h * w)
    return gram


def save_img_and_show(args, optim_img, iter, max_iter, display=False):
    """
    save the generated images every 'save_freq' iterations
    :param args:
    :param optim_img:
    :param iter:
    :param max_iter:
    :param display:
    :return:
    """
    # (b, c, h, w): (1, c, h, w)
    # 0-255
    # (c, h, w)
    optim_img = optim_img.squeeze(0).cpu().numpy()

    # (c, h, w) -> (h, w, c)
    # optim_img = optim_img.transpose(0, 1).transpose(2, 1)
    # swap channel from 1st to 3rd position: ch, _, _ -> _, _, ch
    optim_img = np.moveaxis(optim_img, 0, 2)

    if (iter==max_iter-1) or (iter % args.saving_freq == 0):
        number_fill_length, expansion = args.img_format
        content_name, style_name = args.content_img_name, args.style_img_name
        save_name = f"{content_name}_{style_name}_{str(iter).zfill(number_fill_length)}.{expansion}"

        dump_img = np.copy(optim_img)
        # add mean value in each channel of the image
        dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
        dump_img = np.clip(dump_img, 0, 255).astype('uint8')

        save_dir = os.path.join(args.data_root, args.output_dir_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, save_name)
        # reverse the numpy image r,g,b channel to b,g,r so that we
        # can use opencv to imwrite
        cv2.imwrite(save_path, dump_img[:,:, ::-1])
        print("saving ...")

    if display:
        def transfer_to_uint8(img):
            if isinstance(img, np.ndarray):
                img -= np.min(img)
                img /= np.max(img)
                img *= 255
                return img

        uint_img = np.copy(optim_img)
        uint_img = np.uint8(transfer_to_uint8(uint_img))
        plt.imshow(uint_img)
        plt.show()
        print("showing ...")