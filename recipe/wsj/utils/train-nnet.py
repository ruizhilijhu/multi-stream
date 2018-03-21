
'Train MLP classifier.'

import argparse
import random
import copy
import numpy as np
import pickle
import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable


LOGMSG = 'epoch: {epoch}  mini-batch: {mbatch}  loss (train): {t_loss:.3f}  ' \
     'error rate (train): {t_er:.3%} loss (cv): {cv_loss:.3f} ' \
     'error rate (cv): {cv_er:.3%}'


def batches(archives_list, batch_size, to_torch_dataset):
    arlist = copy.deepcopy(archives_list)
    random.shuffle(arlist)
    for archive in arlist:
        data = np.load(archive)
        dataset = to_torch_dataset(data['features'], data['labels'])
        dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=batch_size, shuffle=True)
        for mb_data, mb_labels in dataloader:
            yield mb_data, mb_labels


def error_rate(model, features, labels, loss_fn):
    outputs = model(features)
    loss = loss_fn(outputs, labels)
    _, predicted = torch.max(outputs, dim=1)
    hits = (labels == predicted).float().sum()
    return loss.data[0], (1 - hits / labels.size(0)).data[0]


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fealist', help='list of "npz" archives')
    parser.add_argument('inputmodel', help='nnet to train')
    parser.add_argument('outmodel', help='output file')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--bsize', type=int, default=1000,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--lrate', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--validation-rate', type=int, default=10,
                        help='frequency of the validation')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='L2 regularization')
    args = parser.parse_args()

    # Load the nnet.
    with open(args.inputmodel, 'rb') as fid:
        model = pickle.load(fid)
    if args.gpu:
        model.cuda()

    # Load the archives
    with open(args.fealist, 'r') as fid:
        archives = [line.strip() for line in fid]

    # We need at list 2 elements in the list to have a training set
    # and a cross-validation set.
    assert len(archives) > 1

    # Define how to convert numpy array to torch objects.
    def to_torch_dataset(np_features, np_labels):
        fea, labels = torch.from_numpy(np_features).float(), \
            torch.from_numpy(np_labels).long()
        return torch.utils.data.TensorDataset(fea, labels)

    # We use the first archive of the list as the cross-validation set.
    cv_data = np.load(archives[0])
    cv_dataset = to_torch_dataset(cv_data['features'], cv_data['labels'])
    cv_fea, cv_labels = cv_dataset.data_tensor, cv_dataset.target_tensor
    if args.gpu:
        cv_fea, cv_labels = cv_fea.cuda(), cv_labels.cuda()

    # Loss function.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate,
                                 weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        t_loss = 0.0
        t_er = 0.0
        for i, data in enumerate(batches(archives[1:], args.bsize,
                                         to_torch_dataset)):
            inputs, labels = Variable(data[0]), Variable(data[1])
            if args.gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # Compute the error rate on the training set.
            _, predicted = torch.max(outputs, dim=1)
            hits = (labels == predicted).float().sum()
            t_er += (1 - hits / labels.size(0)).data[0]
            t_loss += loss.data[0]

            loss.backward()
            optimizer.step()

            if i % args.validation_rate == args.validation_rate - 1:
                t_loss /= args.validation_rate
                t_er /= args.validation_rate
                cv_loss, cv_er = error_rate(model, Variable(cv_fea),
                    Variable(cv_labels), loss_fn)
                print(LOGMSG.format(epoch=epoch+1, mbatch=i+1,
                    t_loss=t_loss, t_er=t_er, cv_loss=cv_loss, cv_er=cv_er))
                t_er = 0.0
                t_loss = 0.0


    if args.gpu:
        model = model.cpu()
    with open(args.outmodel, 'wb') as fid:
        pickle.dump(model, fid)


if __name__ == '__main__':
    run()

