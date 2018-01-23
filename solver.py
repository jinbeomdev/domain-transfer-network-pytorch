import torch.nn as nn
import torch.functional as F
import torch
import torch.optim as optim
import model
from data_loader import data_loader
import torch.cuda
from torch.autograd import Variable
import torchvision.utils as vutils
import os

class Solver(object):

    def __init__(self, config):
        self.mode = config.mode

        #parameters
        self.epochs = config.epochs
        self.batchsize = config.batchsize

        #build model
        self.build_model()

        #load data
        if self.mode == 'train':
            self.mnist, self.svhn = data_loader(self.mode)
        else:
            self.svhn_train, self.svhn_test = data_loader(self.mode)

    def build_model(self):
        if self.mode == 'train':
            self.f = model._f(self.mode)
            self.g = model._g()
            self.D = model._D()

            self.CEL_criterion = nn.CrossEntropyLoss()
            self.MSL_criterion = nn.MSELoss()

            self.D_optimizer = optim.Adam(self.D.parameters())
            self.g_optimizer = optim.Adam(self.g.parameters())

            if torch.cuda.device_count() > 1:
                print("Let's use ", torch.cuda.device_count(), ' GPU')
                self.f = nn.DataParallel(self.f)
                self.g = nn.DataParallel(self.g)
                self.D = nn.DataParallel(self.D)

            if torch.cuda.is_available():
                self.f.cuda()
                self.g.cuda()
                self.D.cuda()
                self.CEL_criterion.cuda()
                self.MSL_criterion.cuda()

        else:
            self.f = model._f(self.mode)

            self.criterion = nn.CrossEntropyLoss()

            self.f_optimizer = optim.Adam(self.f.parameters())

            if torch.cuda.device_count() > 1:
                print("Let's use ", torch.cuda.device_count(), ' GPU')
                self.f = nn.DataParallel(self.f)

            if torch.cuda.is_available():
                self.f.cuda()
                self.criterion.cuda()

    def train(self):
        self.f.eval()
        self.f.load_state_dict(torch.load('./pretrain/checkpoint.pth'))
        #self.g.load_state_dict(torch.load('path'))
        #self.D.load_state_dict(torch.load('path'))

        fake_source_label = 0
        fake_target_label = 1
        real_target_label = 2

        sample_svhn = self.svhn.__iter__().next()[0]
        sample_svhn_v = Variable(sample_svhn)

        if torch.cuda.is_available():
            sample_svhn_v = sample_svhn_v.cuda()

        for epoch in range(self.epochs):
            for i, (mnist, svhn) in enumerate(zip(self.mnist, self.svhn)):
                mnist_image, _ = mnist
                svhn_image, _ = svhn

                mnist_image_v = Variable(mnist_image)
                svhn_image_v = Variable(svhn_image)

                if torch.cuda.is_available():
                    mnist_image_v = mnist_image_v.cuda()
                    svhn_image_v = svhn_image_v.cuda()

                #Train D
                self.g.eval()
                self.D.train()
                self.D.zero_grad()

                D_real = self.D(mnist_image_v)
                loss_D_real = self.CEL_criterion(D_real, Variable(torch.FloatTensor(self.batchsize).fill_(real_target_label).long()).cuda())
                #loss_D_real.backward()

                f_real = self.f(mnist_image_v)
                g_f_real = self.g(f_real)
                new_g_f_real = g_f_real.detach()
                D_g_f_real = self.D(new_g_f_real)
                loss_D_g_f_real = self.CEL_criterion(D_g_f_real, Variable(torch.FloatTensor(self.batchsize).fill_(fake_target_label).long().cuda()))
                #loss_D_g_f_real.backward()

                f_svhn = self.f(svhn_image_v)
                g_f_svhn = self.g(f_svhn)
                new_g_f_svhn = g_f_svhn.detach()
                D_g_f_svhn = self.D(new_g_f_svhn)
                loss_D_g_f_svhn = self.CEL_criterion(D_g_f_svhn, Variable(torch.FloatTensor(self.batchsize).fill_(fake_source_label).long().cuda()))
                #loss_D_g_f_svhn.backward()

                D_loss = loss_D_g_f_svhn + loss_D_g_f_real + loss_D_real
                D_loss.backward()
                self.D_optimizer.step()

                #Train G
                self.g.train()
                self.D.eval()
                self.g.zero_grad()

                f_g_f_svhn = self.f(g_f_svhn)
                l_cosnt = self.MSL_criterion(f_svhn, f_g_f_svhn.detach()) * 15
                #l_cosnt.backward()

                l_tid = self.MSL_criterion(g_f_real, mnist_image_v) * 15
                #l_tid.backward()

                new_g_f_real = g_f_real.detach()
                new_D_g_f_real = self.D(new_g_f_real)
                loss_new_g_f_real = self.CEL_criterion(new_D_g_f_real, Variable(torch.FloatTensor(self.batchsize).fill_(real_target_label).long().cuda()))
                #loss_new_g_f_real.backward()

                new_g_f_svhn = g_f_svhn.detach()
                new_D_g_f_svhn = self.D(new_g_f_svhn)
                loss_new_g_f_svhn = self.CEL_criterion(new_D_g_f_svhn, Variable(torch.FloatTensor(self.batchsize).fill_(real_target_label).long().cuda()))
                #loss_new_g_f_svhn.backward()

                G_loss = loss_new_g_f_svhn + loss_new_g_f_real + l_tid + l_cosnt
                G_loss.backward()
                self.g_optimizer.step()

                # print statistics
                if i % 64 == 63:
                    print('[%d, %5d] G_loss: %.3f D_loss: %.3f' %
                          (epoch + 1, i + 1, G_loss, D_loss))

            vutils.save_image(sample_svhn_v.data, './sample/real_sample.png', normalize=True)
            svhn_to_mnist = self.g(self.f(sample_svhn_v))
            vutils.save_image(svhn_to_mnist.data, './sample/fake_sample%d_%d.png' % (epoch, i), normalize=True)

    def pretrain(self):
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, (image, label) in enumerate(self.svhn_train):
                image_v, label_v = Variable(image), Variable(label)

                if torch.cuda.is_available():
                    image_v = image_v.cuda()
                    label_v = label_v.cuda()

                self.f.zero_grad()

                output = self.f(image_v)
                loss = self.criterion(output.squeeze(3).squeeze(2), label_v)

                loss.backward()
                self.f_optimizer.step()

                # print statistics
                running_loss += loss.data[0]
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        self.f.eval()
        correct = 0
        total = 0
        for i, (image, label) in enumerate(self.svhn_test):
            image_v, label_v = Variable(image), Variable(label)

            if torch.cuda.is_available():
                image_v, label_v = image_v.cuda(), label_v.cuda()

            output = self.f(image_v)
            _, predicted = torch.max(output.data, 1)
            total += label_v.size(0)
            correct += (predicted == label_v.data).sum()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))

        if not os.path.exists('./pretrain'):
            os.mkdir('./pretrain')
        torch.save(self.f.state_dict(), './pretrain/checkpoint.pth')