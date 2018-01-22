import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import cv2
import numpy as np

class Solver(object):
    def __init__(self, mnist, svhn):
        self.mnist = mnist
        self.svhn = svhn
        self.learning_rage = learning_rate
        self.epcohs = epochs
        self.batch_size = batch_size

def zero_grad(f, g, D):
    f.zero_grad()
    g.zero_grad()
    D.zero_grad()

#hyperparameter
learning_rate = 0.0003
epochs = 100
batch_size = 32

#data load
mnist, svhn = data_loader()
mnist_iter = torch.utils.data.DataLoader(mnist, batch_size=batch_size,
                            shuffle=False, num_workers=2)
svhn_iter = torch.utils.data.DataLoader(svhn, batch_size=batch_size,
                                        shuffle=False, num_workers=2)

#sample svhn
sample_svhn = svhn_iter.__iter__().next()[0]
sample_svhn_v = Variable(sample_svhn)

#model build
f = _f()
g = _g()
D = _D()

#loss, optimizer
criterion = nn.BCELoss()
optimizer_f = optim.Adam(f.parameters(), lr=learning_rate)
optimizer_g = optim.Adam(g.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(D.parameters(), lr=learning_rate)

#cuda
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    f = nn.DataParallel(f)
    g = nn.DataParallel(g)
    D = nn.DataParallel(D)

if torch.cuda.is_available():
    f.cuda()
    g.cuda()
    D.cuda()
    criterion.cuda()
    sample_svhn_v = sample_svhn_v.cuda()

for epoch in range(epochs):
    for i, (mnist, svhn) in enumerate(zip(mnist_iter, svhn_iter)):
        mnist_image, mnist_label = mnist
        svhn_image, svhn_label = svhn

        mnist_image_v = Variable(mnist_image)
        svhn_image_v = Variable(svhn_image)

        real_label = Variable(torch.Tensor(mnist_image.size(0)).fill_(1))
        #print('real_label', real_label.shape)
        fake_label = Variable(torch.Tensor(mnist_image.size(0)).fill_(0))
        #print('fake_label', fake_label.shape)


        if torch.cuda.is_available():
            mnist_image_v = mnist_image_v.cuda()
            svhn_image_v = svhn_image_v.cuda()
            real_label = real_label.cuda()
            fake_label = fake_label.cuda()

        """
        train the model for source domain
        """
        zero_grad(f, g, D)

        #svhn to mnist
        fx = f(svhn_image_v)
        #print('fx', fx.shape)
        gfx = g(fx)
        #print('gfx', gfx.shape)
        Dgfx = D(gfx)
        #print('Dgfx', Dgfx.shape)
        fgfx = f(gfx)
        #print('fgfx', fgfx.shape)

        #loss
        #print(Dgfx)
        D_loss_src = criterion(Dgfx, fake_label)
        #print('D_loss_src', D_loss_src.shape)
        G_loss_src = criterion(Dgfx, real_label)
        f_loss_src = torch.mean((fx - fgfx)**2) * 15

        #backward
        D_loss_src.backward(retain_graph=True)
        G_loss_src.backward(retain_graph=True)
        f_loss_src.backward(retain_graph=True)

        #optimize
        optimizer_D.step()
        optimizer_g.step()
        optimizer_f.step()

        """
        train the model for target domain
        """
        zero_grad(f, g, D)

        #mnist to mnist
        fx = f(mnist_image_v)
        gfx = g(fx)
        Dgfx = D(gfx)
        Dx = D(mnist_image_v)

        #loss
        Dx_loss = criterion(Dx, real_label)
        Dgfx_loss = criterion(Dgfx, fake_label)
        D_loss_trg = Dx_loss + Dgfx_loss
        gfx_loss = criterion(Dgfx, real_label)
        g_const_loss = torch.mean((mnist_image_v - gfx)**2)*15
        g_loss_trg = gfx_loss + g_const_loss

        #backward
        D_loss_trg.backward(retain_graph=True)
        g_loss_trg.backward(retain_graph=True)

        #optimize
        optimizer_D.step()
        optimizer_g.step()

        #save sample
        if i % 100 == 0:
            vutils.save_image(sample_svhn,
                              './sample/real_sample.png',
                              normalize=True)
            svhn_to_mnist = g(f(svhn_image_v))
            vutils.save_image(svhn_to_mnist.data,
                              './sample/fake_sample%d_%d.png' % (epoch, i),
                              normalize=True)




