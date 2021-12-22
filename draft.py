import argparse
import torch
import torch.nn.functional as F


def test(a, b, c=3):
    print(a + b + c)


def show(batch_size, num_epochs, model_name):
    print(batch_size, num_epochs, model_name)


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--model_name', type=str, default='macbert-base-chinese')

    return parser.parse_args()


if __name__ == '__main__':
    # d = {'a': 1, 'b': 2}
    # test(**d, c=5)
    a = torch.tensor([[1., 0., 1.], [0., 1., 1.]])
    b = torch.tensor([[-1., 1., 0.], [1., 1., 0.]])
    print(F.cosine_similarity(a, b))