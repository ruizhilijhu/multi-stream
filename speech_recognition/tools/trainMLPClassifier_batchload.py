#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:46:38 2018

@author: lucas ondel, minor changes by samik sadhu 
"""

'Prepare data, train MLP and do cross validation using batch loading'


import argparse
import numpy as np
import pickle
import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable
from os import listdir
import os 
from os.path import isfile, join
import sys


def tidyData(data_dir):
    
    print('%s: Checking for train and test data...' % sys.argv[0])
    allfiles = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    print('%s: In total %d train and test data files found..' % (sys.argv[0],len(allfiles)))
    train_files=[]; test_files=[]
    
    for i in range(len(allfiles)):
        if 'test' in allfiles[i]:
            test_files.append(allfiles[i])
            
    for i in range(len(allfiles)):
        if 'train' in allfiles[i]:
            train_files.append(allfiles[i])
    
    # Check the data dimension
    data=np.load(os.path.join(data_dir, train_files[0]))
    data_dim=data.shape[1]-1
    
    # Load all train and test data into big files 
    train_data=np.empty((0,data_dim)); test_data=np.empty((0,data_dim)); train_labels=np.array([]); test_labels=np.array([])
    
    print('%s: Loading training files...' % sys.argv[0])
    for i in range(len(train_files)):
        data=np.load(os.path.join(data_dir, train_files[i]))
        train_data=np.append(train_data,data[:,0:-1],axis=0)
        train_labels=np.append(train_labels,data[:,-1])
    
    print('%s: Loading test files...' % sys.argv[0])
    for i in range(len(test_files)):
        data=np.load(os.path.join(data_dir, test_files[i]))
        test_data=np.append(test_data,data[:,0:-1],axis=0)
        test_labels=np.append(test_labels,data[:,-1])
    
    return train_data, train_labels, test_data, test_labels

def error_rate(model, features, labels, loss_fn):
    outputs = model(features)
    #np.save('./test_output', outputs)
    #np.save('./test_labels', labels)
            
    loss = loss_fn(outputs, labels)
    _, predicted = torch.max(outputs, dim=1)      
    hits = (labels == predicted).float().sum()
    return loss.data[0], (1 - hits / labels.size(0)).data[0]

def get_sample_meanvar(train_files):
    
    size_acc=0
    data=np.load(train_files[0])
    data=data[:,0:-1]
    size_acc+=np.shape(data)[0] 
    mean_acc=data.mean(axis=0)
   
    print('%s: Getting mean of training samples...' % sys.argv[0])
    for i in range(1,len(train_files)):
        data=np.load(train_files[i])
        data=data[:,0:-1]
        size_now=np.shape(data)[0]        
        mean_now=data.mean(axis=0)
        mean_acc=(mean_now*size_now+mean_acc*size_acc)/(size_now+size_acc)
        size_acc+=size_now
    
    size_acc=0
    data=np.load(train_files[0])
    data=data[:,0:-1]
    size_acc+=np.shape(data)[0]
    var_acc=np.sum(np.square(data-mean_acc),axis=0)
    
    print('%s: Getting variance of training samples...' % sys.argv[0])   
    for i in range(1,len(train_files)):
        data=np.load(train_files[i])
        data=data[:,0:-1]
        size_now=np.shape(data)[0]
        size_acc+=size_now
        var_acc+=np.sum(np.square(data-mean_acc),axis=0)
    var_acc=var_acc/size_acc;
    
    return mean_acc, var_acc

def run(train_files,test_data,test_labels,mean,var,args):
    
    
    # Check the data dimension
    data=np.load(train_files[0])
    data_dim=data.shape[1]-1
    targetdim=args.ntargets
    
    if args.mvnorm:
        test_data -= mean
        test_data /= np.sqrt(var)
        
    # Build the MLP.
    print('%s: Building the MLP...' % sys.argv[0])
    if args.put_kink:
        
        structure = [nn.Linear(data_dim, 64), nn.Tanh()]
        for i in range(args.nlayers - 1):
            if i==0:
                structure += [nn.Linear(64, args.nunits), nn.Tanh()]
            else:
                structure += [nn.Linear(args.nunits, args.nunits), nn.Tanh()]
        structure += [nn.Linear(args.nunits, targetdim)]
        model = nn.Sequential(*structure)
    
    else:
        
        structure = [nn.Linear(data_dim, args.nunits), nn.Tanh()]
        for i in range(args.nlayers - 1):
            structure += [nn.Linear(args.nunits, args.nunits), nn.Tanh()]
        structure += [nn.Linear(args.nunits, targetdim)]
        model = nn.Sequential(*structure)
    
    if args.gpu is not None:
        with torch.cuda.device(args.gpu):
            model.cuda(args.gpu)
                  
    print('%s: Defining Loss Function...' % sys.argv[0])
    # Loss function.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate,
                                 weight_decay=args.weight_decay)

    
    
    test_data, test_labels = torch.from_numpy(test_data).float(), \
        torch.from_numpy(test_labels).long()
        
    #v_train_data, v_train_labels = Variable(train_data), Variable(train_labels)
    
    if args.gpu is not None:
        with torch.cuda.device(args.gpu):
            v_test_data, v_test_labels = Variable(test_data).cuda(), Variable(test_labels).cuda()
    else:
        v_test_data, v_test_labels = Variable(test_data), Variable(test_labels)


    print('%s: Start Training Iterations...' % sys.argv[0])
    if args.gpu is not None:
        with torch.cuda.device(args.gpu):
            for epoch in range(args.epochs):
                t_loss = 0.0
                t_er = 0.0
                meg_batch=0
                # Load data batch by batch for training
                train_data=np.empty((0,data_dim)); train_labels=np.array([]);
                for dat_set in range(len(train_files)):
                    data=np.load(train_files[dat_set])
                    train_data=np.append(train_data,data[:,0:-1],axis=0)
                    train_labels=np.append(train_labels,data[:,-1])
                    
                    if ((np.shape(train_data)[0]<=args.mega_bsize) and (dat_set<len(train_files)-1)):
                        continue
                    else:
                        meg_batch+=1
                    
                    if args.mvnorm:
                        train_data -= mean
                        train_data /= np.sqrt(var)
        
                    train_data, train_labels = torch.from_numpy(train_data).float(), \
                    torch.from_numpy(train_labels).long()
                    
                    dataset = torch.utils.data.TensorDataset(train_data, train_labels)
                    trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.bsize,
                                              shuffle=True)
    
                    for i, data in enumerate(trainloader):
                        
                        inputs, labels = Variable(data[0]).cuda(), Variable(data[1]).cuda()
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
                            cv_loss, cv_er = error_rate(model, v_test_data, v_test_labels, loss_fn)
                            logmsg = 'epoch: {epoch} mega-batch: {meg_batch} mini-batch: {mbatch}  loss (train): {t_loss:.3f}  ' \
                                     'error rate (train): {t_er:.3%} loss (cv): {cv_loss:.3f} ' \
                                     'error rate (cv): {cv_er:.3%}'.format( 
                                     epoch=epoch+1, meg_batch=meg_batch, mbatch=i+1, t_loss=t_loss, t_er=t_er,
                                     cv_loss=cv_loss, cv_er=cv_er)
                            
                            t_er = 0.0
                            t_loss = 0.0
                            print(logmsg)
                    train_data=np.empty((0,data_dim)); train_labels=np.array([]);
                        
            model=model.cpu()
            
            with open(args.outmodel, 'wb') as fid:
                pickle.dump(model, fid)
    else:
        
        for epoch in range(args.epochs):
                t_loss = 0.0
                t_er = 0.0
                
                # Load data batch by batch for training 
                for dat_set in range(len(train_files)):
                    data=np.load(train_files[dat_set])
                    train_data=data[:,0:-1]
                    train_labels=data[:,-1]
                    
                    if args.mvnorm:
                        train_data -= mean
                        train_data /= np.sqrt(var)
                        
                    train_data, train_labels = torch.from_numpy(train_data).float(), \
                    torch.from_numpy(train_labels).long()
                    
                    dataset = torch.utils.data.TensorDataset(train_data, train_labels)
                    trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.bsize,
                                              shuffle=True)
                    
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
                            cv_loss, cv_er = error_rate(model, v_test_data, v_test_labels, loss_fn)
                            logmsg = 'data set: {dat_set} epoch: {epoch}  mini-batch: {mbatch}  loss (train): {t_loss:.3f}  ' \
                                     'error rate (train): {t_er:.3%} loss (cv): {cv_loss:.3f} ' \
                                     'error rate (cv): {cv_er:.3%}'.format( dat_set=train_files[dat_set],
                                     epoch=epoch+1, mbatch=i+1, t_loss=t_loss, t_er=t_er,
                                     cv_loss=cv_loss, cv_er=cv_er)
                            
                            t_er = 0.0
                            t_loss = 0.0
                            print(logmsg)
        
        
        
        with open(args.outmodel, 'wb') as fid:
                pickle.dump(model, fid)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_directory', help='place to get all training and test data in .npy format')
    parser.add_argument('ntargets', type=int, help='number of targets')
    parser.add_argument('nlayers', type=int, help='number of hidden layers')
    parser.add_argument('nunits', type=int, help='number of units per leayer')
    parser.add_argument('outmodel', help='output file')
    parser.add_argument('--gpu',type=int,help='gpu device id (Ignore if you do not want to run on gpu!)')
    parser.add_argument('--bsize', type=int, default=1000,
                        help='batch size')
    parser.add_argument('--mega_bsize', type=int, default=1000*100,
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
    parser.add_argument('--put_kink', help='Puts a 64 dimension layer at the beginning to plot filters',action='store_true')
    args = parser.parse_args()

    assert args.nlayers > 0

    print('%s: Running  MLP training...' % sys.argv[0])
        
    allfiles = [f for f in listdir(args.data_directory) if isfile(join(args.data_directory, f))]
    
    train_files=[]; test_files=[]
               
    for i in range(len(allfiles)):
        if 'train' in allfiles[i]:
            train_files.append(os.path.join(args.data_directory,allfiles[i]))
            
    print('%s: In total %d train data files found..passing them for MLP training' % (sys.argv[0],len(train_files)))
            
    test_data=np.load(join(args.data_directory,'test_data.npy'))
    test_labels=np.load(join(args.data_directory,'test_labels.npy'))
    mean=np.load(join(args.data_directory,'data_mean.npy'))
    var=np.load(join(args.data_directory,'data_var.npy'))
    
    run(train_files,test_data,test_labels,mean,var,args)
