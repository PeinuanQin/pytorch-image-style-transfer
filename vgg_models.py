"""
 @file: vgg_models.py
 @Time    : 2023/1/10
 @Author  : Peinuan qin
 """
from collections import namedtuple
from torch import nn


class Vgg_net(nn.Module):
    """
    the parent class of the vgg net that used in this project
    """
    def __init__(self, args, base_model, out_num, anchors):
        super(Vgg_net, self).__init__()
        self.args = args
        # the base vgg-16 / vgg-19 models
        self.base_model = base_model
        self.features = self.base_model.features
        # how many intermediate conv layer output we need for loss calculation
        self.out_num = out_num
        # the intermediate conv layer indexes
        self.anchors = anchors
        assert self.out_num == len(self.anchors)
        self.anchors = [0] + self.anchors
        # for vgg-16, it's like: [(0,4), (4,9), (9,16), (16,23)]
        self.start_end_tuples = list(zip(*[self.anchors[i:] for i in range(2)]))
        # construct subnets according to the start_end_tuples
        self.subnets = nn.ModuleList([self.make_sub_net(start, end) for start, end in self.start_end_tuples])
        # the intermediate output index that used for content loss calculation
        self.content_index = 1
        # naming for each output layer
        self.output_layer_names = [f'out{i}' for i in range(self.out_num)]
        # the intermediate output indexes that used for style loss calculation,
        # default to use all intermediate layers for style loss
        self.style_indexes = list(range(len(self.output_layer_names)))

        print("show subnets" + '*' * 35)
        for i in range(len(self.subnets)):
            print(f"subnet{i + 1}: {self.subnets[i]}")
            print("*" * 35)

        # use vgg only for feature extraction, so freeze all layers parameters
        if not self.args.requires_grad:
            print(f"freeze all layers ...")
            for para in self.parameters():
                para.requires_grad = False

    def forward(self, x):
        outs = []
        for i in range(len(self.subnets)):
            x = self.subnets[i](x)
            outs.append(x)

        VggOutputs = namedtuple("VggOutputs", self.output_layer_names)
        out = VggOutputs(*outs)
        return out

    def make_sub_net(self, start_index, end_index):
        """
        make subnet for stage conv out
        :param start_index:
        :param end_index:
        :return:
        """
        subnet = nn.Sequential()
        for i in range(start_index, end_index):
            module_name = str(i)
            module = self.features[i]
            subnet.add_module(module_name, module)
        return subnet


class VGG16(Vgg_net):
    def __init__(self
                 , args
                 , base_model
                 , out_num
                 , anchors
                 , content_index
                 , style_indexs):
        super(VGG16, self).__init__(args, base_model, out_num, anchors)
        # print(self.base_model)
        self.content_index = content_index
        self.style_indexes = style_indexs


class VGG19(Vgg_net):
    def __init__(self
                 , args
                 , base_model
                 , out_num
                 , anchors
                 , content_index
                 , style_indexs):
        super(VGG19, self).__init__(args, base_model, out_num, anchors)
        # print(self.base_model)
        self.content_index = content_index
        self.style_indexes = style_indexs


