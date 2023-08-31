from argparse import ArgumentParser
from utils.train import System
import torch

torch.manual_seed(123)  # 设置随机种子为123


def init_hparams():
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--num_class", type=int, default=5)
    parser.add_argument("--frac", type=float, default=0.1)
    parser.add_argument("--pretrain", type=bool, default=True)
    parser.add_argument("--dataset", type=str, choices=["apple", "crop", "weed"], default='crop')
    try:

        hparams = parser.parse_args()




    except:
        print('解析超参数失败，请检查超参数设置')
        hparams = parser.parse_args([])

    return hparams


if __name__ == "__main__":
    hparams = init_hparams()
    for EPOCH in range(1):
        mySys = System(hparams.batch_size, hparams.backbone, hparams.pretrain, hparams.num_class, hparams.dataset,
                       hparams.frac, hparams.cuda,
                       hparams.epochs, EPOCH, hparams.num_workers)
        mySys.run()
