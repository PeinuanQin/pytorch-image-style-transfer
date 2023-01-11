"""
 @file: train.py
 @Time    : 2023/1/10
 @Author  : Peinuan qin
 """


def print_loss_info(i
                    , total_loss
                    , content_loss
                    , style_loss
                    , total_variation_loss
                    , optimizer_type):
    """
    print all losses
    :param i:
    :param total_loss:
    :param content_loss:
    :param style_loss:
    :param total_variation_loss:
    :param optimizer_type:
    :return:
    """
    with torch.no_grad():
        loss_dict = {'total_loss': total_loss.item()
            , 'content_loss': content_loss.item()
            , 'style_loss': style_loss.item()
            , 'tv_loss': total_variation_loss.item()}
        loss_info = reduce(lambda x, y: x + y, [f"{k}: {v} \t" for k, v in loss_dict.items()])
        print(f"{optimizer_type} | iteration: {i:03}, {loss_info}")


def main(args):
    # torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # select images
    content_img, style_img, optim_img = load_imgs(args, device)

    # select model
    model = model_selection(args, device)

    # when input the content img to the model, we can get
    # all conv outputs from these subnets in model
    content_outputs = model(content_img)
    style_outputs = model(style_img)

    content_index = model.content_index
    style_indexs = model.style_indexes

    # we only need one intermediate layer output for content loss calculation
    content_representation = content_outputs[content_index].squeeze(axis=0)
    style_outputs = [style_outputs[i] for i in style_indexs]

    # we use all conv layers' output will be used for style loss calculation
    # for each output (out1, out2, ...), calculate its gram
    style_representations = [get_grams(style_output) for style_output in style_outputs]

    optimizer, iterations = optimizer_selection(args, optim_img)

    # use adam optimizer for iterations
    if isinstance(optimizer, Adam):
        optimizer_type = 'Adam'
        for i in tqdm(range(iterations)):
            total_loss, content_loss, style_loss, total_variation_loss = calculate_loss(args
                                                                                        , model
                                                                                        , optim_img
                                                                                        , content_representation
                                                                                        , style_representations
                                                                                        )
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                print_loss_info(i, total_loss, content_loss, style_loss, total_variation_loss, optimizer_type)
                save_img_and_show(args, optim_img, i, iterations, True)

    # take lbfgs for iterations
    else:
        optimizer_type = 'LBFGS'

        iter = 0

        def closure():
            nonlocal iter
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            total_loss, content_loss, style_loss, tv_loss = calculate_loss(args
                                                                           , model
                                                                           , optim_img
                                                                           , content_representation
                                                                           , style_representations
                                                                           )
            if total_loss.requires_grad:
                total_loss.backward()
            with torch.no_grad():
                print_loss_info(iter, total_loss, content_loss, style_loss, tv_loss, optimizer_type)
                save_img_and_show(args, optim_img, iter, iterations, False)

            iter += 1
            return total_loss

        optimizer.step(closure)


if __name__ == '__main__':
    from local_config import init_config

    from functools import reduce
    from tqdm import tqdm
    from util import *

    args = init_config()
    main(args)
