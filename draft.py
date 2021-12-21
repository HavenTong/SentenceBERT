import argparse


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
    d = {'a': 1, 'b': 2}
    test(**d, c=5)