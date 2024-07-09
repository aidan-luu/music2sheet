import sys,os,signal

import numpy as np
import preprocess.musicnet as musicnet

import torch
from torch.autograd import Variable
from torch.nn.functional import conv1d, mse_loss

from time import time

import matplotlib.pyplot as plt
from IPython.display import Audio

from sklearn.metrics import average_precision_score

def worker_init(args):
    signal.signal(signal.SIGINT, signal.SIG_IGN) # ignore signals so parent can handle them
    np.random.seed(os.getpid() ^ int(time())) # approximately random seed for workers

def main():
    root = 'musicnet'
    batch_size = 100
    kwargs = {'num_workers': 8, 'pin_memory': True, 'worker_init_fn': worker_init}

    m = 128
    k = 500
    d = 4096
    window = 16384
    stride = 512
    regions = 1 + (window - d)//stride
    train_set = musicnet.MusicNet(root=root, train=True, download=False, window=window)#, pitch_shift=5, jitter=.1)
    test_set = musicnet.MusicNet(root=root, train=False, window=window, epoch_size=50000)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size,**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=batch_size,**kwargs)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Creates sine and cosine filters that extract frequencies from audio
    def create_filters(d,k,low=50,high=6000):
        x = np.linspace(0, 2*np.pi, d, endpoint=False)
        wsin = np.empty((k,1,d), dtype=np.float32)
        wcos = np.empty((k,1,d), dtype=np.float32)
        start_freq = low
        end_freq = high
        num_cycles = start_freq*d/44000.
        scaling_ind = np.log(end_freq/start_freq)/k
        window_mask = 1.0-1.0*np.cos(x)
        for ind in range(k):
            wsin[ind,0,:] = window_mask*np.sin(np.exp(ind*scaling_ind)*num_cycles*x)
            wcos[ind,0,:] = window_mask*np.cos(np.exp(ind*scaling_ind)*num_cycles*x)
        
        # Convert numpy arrays into tensors
        return torch.from_numpy(wsin).to(device), torch.from_numpy(wcos).to(device)

    wsin,wcos = create_filters(d,k)

    # Use mps since no cuda support
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS device")
    else:
        device = torch.device('cpu')
        print("Using CPU")
        
    def init_weights(scale=0.):
        wscale = 0.
        beta = Variable(wscale*torch.rand([regions*k,m]).to(device), requires_grad=True)
        betaavg = Variable(beta.data.clone(), requires_grad=False)
        return (beta,), (betaavg,)

    def forward(x, beta):
        zx = conv1d(x[:,None,:], wsin, stride=stride).pow(2) \
        + conv1d(x[:,None,:], wcos, stride=stride).pow(2)
        return torch.mm(torch.log(zx + musicnet.epsilon).view(x.data.size()[0],regions*k), beta)

    def L(y_hat, y, beta):
        # adjust for per-frame loss
        return mse_loss(y_hat, y)*128/2.# + .01*torch.mean(beta.pow(2.))

    weights, averages = init_weights()
    loss_history = []
    avgp_history = []
    avg = .9998
    optimizer = torch.optim.SGD(weights, lr=0.000001, momentum=.95)

    try:
        with train_set, test_set:
            print('square loss\tavg prec\ttime\t\tutime')
            for epoch in range(50):
                t = time()
                for i, (x, y) in enumerate(train_loader):
                    optimizer.zero_grad()
                    x, y = Variable(x.to(device), requires_grad=False), Variable(y.to(device), requires_grad=False)
                    loss = L(forward(x,*weights),y,*weights)
                    loss.backward()
                    optimizer.step()

                    for w,wavg in zip(weights, averages):
                        wavg.data.copy_(avg*wavg.data + (1.-avg)*w.data)

                t1 = time()
                avgp, loss = 0., 0.
                yground = torch.FloatTensor(batch_size*len(test_loader), m)
                yhat = torch.FloatTensor(batch_size*len(test_loader), m)
                for i, (x, y) in enumerate(test_loader):
                    x, y = Variable(x.to(device), requires_grad=False), Variable(y.to(device), requires_grad=False)
                    yhatvar = forward(x,*averages)
                    loss += L(yhatvar,y,*averages).item()
                    yground[i*batch_size:(i+1)*batch_size,:] = y.data
                    yhat[i*batch_size:(i+1)*batch_size,:] = yhatvar.data
                avgp = average_precision_score(yground.numpy().flatten(),yhat.numpy().flatten())
                loss_history.append(loss/len(test_loader))
                avgp_history.append(avgp)
                print('{:2f}\t{:2f}\t{:2f}\t{:2f}'.format(loss_history[-1],avgp_history[-1],time()-t, time()-t1))

    except KeyboardInterrupt:
        print('Graceful Exit')
    else:
        print('Finished')

if __name__ == '__main__':
    main()