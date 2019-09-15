import torch
# import test_models as models
from thop import profile
import torchvision
import models
import argparse


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='eca_resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: eca_resnet50)')

def main():
    global args
    args = parser.parse_args()
    model = models.__dict__[args.arch]()    
    print(model)
    input = torch.randn(1, 3, 224, 224)
    model.train()
    # model.eval()
    flops, params = profile(model, inputs=(input, ))
    print("flops = ", flops)
    print("params = ", params)
    flops, params = clever_format([flops, params], "%.3f")
    print("flops = ", flops)
    print("params = ", params)

def clever_format(nums, format="%.2f"):
    clever_nums = []

    for num in nums:
        if num > 1e12:
            clever_nums.append(format % (num / 1024 ** 4) + "T")
        elif num > 1e9:
            clever_nums.append(format % (num / 1024 ** 3) + "G")
        elif num > 1e6:
            clever_nums.append(format % (num / 1024 ** 2) + "M")
        elif num > 1e3:
            clever_nums.append(format % (num / 1024) + "K")
        else:
            clever_nums.append(format % num + "B")

    clever_nums = clever_nums[0] if len(clever_nums) == 1 else (*clever_nums, )

    return clever_nums


if __name__ == '__main__':
    main()
