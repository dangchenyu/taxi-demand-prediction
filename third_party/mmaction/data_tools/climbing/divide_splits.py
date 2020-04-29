import os
import random
import argparse


def divide_splits(data_root, classes):
    train_file = open(os.path.join(data_root, 'annotations/trainlist01.txt'), 'w')
    test_file = open(os.path.join(data_root, 'annotations/testlist01.txt'), 'w')
    for i, clazz in enumerate(classes):
        seqs = list(os.listdir(os.path.join(data_root, 'videos', clazz)))
        seqs.sort()
        for seq in seqs:
            temp = random.random()
            if temp > 0.75:
                test_file.write('{}/{} {}\n'.format(clazz, seq, i + 1))
            else:
                train_file.write('{}/{} {}\n'.format(clazz, seq, i + 1))
    test_file.close()
    train_file.close()


def generate_class_indices(data_root, classes):
    inds_txt = open(os.path.join(data_root, 'annotations/classInd.txt'), 'w+')

    for i, clazz in enumerate(classes):
        inds_txt.write('{} {}\n'.format(i + 1, clazz))

    inds_txt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, help='Path to the climbing data directory')
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.data_root, 'annotations')):
        os.mkdir(os.path.join(args.data_root, 'annotations'))

    classes = list(os.listdir(os.path.join(args.data_root, 'videos')))
    classes.sort()

    generate_class_indices(args.data_root, classes)
    divide_splits(args.data_root, classes)
