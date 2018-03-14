
'Create and train MLP classifier.'

import argparse
import numpy as np
import pickle
import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable


def error_rate(model, features, labels, loss_fn):
    outputs = model(features)
    loss = loss_fn(outputs, labels)
    _, predicted = torch.max(outputs, dim=1)
    hits = (labels == predicted).float().sum()
    return loss.data[0], (1 - hits / labels.size(0)).data[0]


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trainfea', help='train features')
    parser.add_argument('trainlab', help='train labels')
    parser.add_argument('cvfea', help='cross-validation features')
    parser.add_argument('cvlab', help='cross-validation labels')
    parser.add_argument('nfilters', type=int, help='number of 2D filters')
    parser.add_argument('ntargets', type=int, help='number of targets')
    parser.add_argument('nlayers', type=int, help='number of hidden layers')
    parser.add_argument('nunits', type=int, help='number of units per leayer')
    parser.add_argument('outmodel', help='output file')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='add drop out after the first layer')
    parser.add_argument('--bsize', type=int, default=1000,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--lrate', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--mvnorm', action='store_true',
                        help='mean-variance normalization of the features')
    parser.add_argument('--validation-rate', type=int, default=10,
                        help='frequency of the validation')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='L2 regularization')
    args = parser.parse_args()

    assert args.nlayers > 0

    # Load the data.
    train_X, train_Y = np.load(args.trainfea), np.load(args.trainlab)
    cv_X, cv_Y = np.load(args.cvfea), np.load(args.cvlab)

    if args.mvnorm:
        mean = train_X.mean(axis=0)
        var = train_X.var(axis=0)
        train_X -= mean
        train_X /= np.sqrt(var)
        cv_X -= mean
        cv_X /= np.sqrt(var)


    # Input/output dimension of the MLP.
    feadim, targetdim = train_X.shape[1], args.ntargets

    # Build the MLP.
    structure = [nn.Linear(feadim, args.nfilters), nn.Tanh()]
    if args.dropout > 0:
        structure += [nn.Dropout(p=args.dropout)]
    player_nunits = args.nfilters
    for i in range(args.nlayers - 1):
        structure += [nn.Linear(player_nunits, args.nunits), nn.Tanh()]
        player_nunits = args.nunits
    structure += [nn.Linear(args.nunits, targetdim)]
    model = nn.Sequential(*structure)

    # Loss function.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate,
                                 weight_decay=args.weight_decay)

    train_X, train_Y = torch.from_numpy(train_X).float(), \
        torch.from_numpy(train_Y).long()
    cv_X, cv_Y = torch.from_numpy(cv_X).float(), \
        torch.from_numpy(cv_Y).long()
    v_train_X, v_train_Y = Variable(train_X), Variable(train_Y)
    v_cv_X, v_cv_Y = Variable(cv_X), Variable(cv_Y)

    dataset = torch.utils.data.TensorDataset(train_X, train_Y)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.bsize,
                                              shuffle=True)


    for epoch in range(args.epochs):
        t_loss = 0.0
        t_er = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = Variable(data[0]), Variable(data[1])
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
                cv_loss, cv_er = error_rate(model, v_cv_X, v_cv_Y, loss_fn)
                logmsg = 'epoch: {epoch}  mini-batch: {mbatch}  loss (train): {t_loss:.3f}  ' \
                         'error rate (train): {t_er:.3%} loss (cv): {cv_loss:.3f} ' \
                         'error rate (cv): {cv_er:.3%}'.format(
                         epoch=epoch+1, mbatch=i+1, t_loss=t_loss, t_er=t_er,
                         cv_loss=cv_loss, cv_er=cv_er)

                t_er = 0.0
                t_loss = 0.0
                print(logmsg)

    with open(args.outmodel, 'wb') as fid:
        pickle.dump(model, fid)


if __name__ == '__main__':
    run()
