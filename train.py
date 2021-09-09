import numpy as np
import math
import itertools
from sklearn.utils import shuffle

import torch
import torch.optim as optim
from torch.autograd import Variable

from model import AE1, AE2, Fusion, Discrimiter
from utils.Dataset import Dataset
from utils.next_batch import next_batch
from utils.print_result import print_result

# set gpu device
torch.cuda.set_device(0)

if __name__ == "__main__":
    is_cuda = True
    is_cuda = is_cuda and torch.cuda.is_available()

    # Hyper parameter
    epochs_pre = 10
    epochs_total = 15
    epochs_h = 50

    batch_size = 100

    lr_pre = 0.001
    lr_ae = 0.001
    lr_dg = 0.001
    lr_h = 0.01

    latent_h = 64
    latent_z = 200

    data_h = Dataset('handwritten_2views')
    x1, x2, gt = data_h.load_data()
    x1 = data_h.normalize(x1, 0)
    x2 = data_h.normalize(x2, 0)
    xxx = np.concatenate((x1, x2), axis=1)

    H_mean = np.random.uniform(0, 1, [x1.shape[0], latent_h])
    H_logvar = np.random.uniform(0, 1, [x1.shape[0], latent_h])

    net_ae1 = AE1(dim_data=x1.shape[1], dim_latents=latent_z)
    net_ae2 = AE2(dim_data=x2.shape[1], dim_latents=latent_z)
    net_dg = Fusion(h_dim=latent_h, dim_latents=latent_z, batch_size=batch_size, is_cuda=is_cuda)
    net_ae1_D = Discrimiter()
    net_ae2_D = Discrimiter()
    net_dg1_D = Discrimiter()
    net_dg2_D = Discrimiter()

    if is_cuda:
        net_ae1.cuda()
        net_ae2.cuda()
        net_dg.cuda()
        net_ae1_D.cuda()
        net_ae2_D.cuda()
        net_dg1_D.cuda()
        net_dg2_D.cuda()

    # pre train
    opt_ae_pre = optim.Adam(params=itertools.chain(net_ae1.parameters(), net_ae2.parameters()), lr=lr_pre)
    for epoch in range(1, epochs_pre+1):
        net_ae1.train()
        net_ae2.train()
        x1, x2, gt = shuffle(x1, x2, gt)
        for batch_x1, batch_x2, batch_No in next_batch(x1, x2, batch_size):
            batch_x1 = torch.from_numpy(batch_x1).float()
            batch_x2 = torch.from_numpy(batch_x2).float()

            if is_cuda:
                batch_x1 = batch_x1.cuda()
                batch_x2 = batch_x2.cuda()

            opt_ae_pre.zero_grad()
            z1_mean, z1_logvar, z1 = net_ae1.get_z_half(batch_x1)
            z2_mean, z2_logvar, z2 = net_ae2.get_z_half(batch_x2)
            ae_pre_recon = 0.5*torch.nn.MSELoss()(batch_x1, net_ae1(batch_x1)) + 0.5*torch.nn.MSELoss()(batch_x2, net_ae2(batch_x2))
            KLD1 = -0.5 * torch.sum(1 + z1_logvar - z1_mean.pow(2) - z1_logvar.exp(), dim=1)
            KLD2 = -0.5 * torch.sum(1 + z2_logvar - z2_mean.pow(2) - z2_logvar.exp(), dim=1)
            ae_pre_loss = ae_pre_recon + 0.0001 * torch.mean(KLD1 + KLD2)

            ae_pre_loss.backward()
            opt_ae_pre.step()

        output = "Pre_epoch : {:.0f}".format(epoch)
        print(output)

    # train
    num_samples = x1.shape[0]
    num_batchs = math.floor(num_samples / batch_size)   # Fix the last batch
    for epoch in range(1, epochs_total+1):
        x1, x2, H_mean, H_logvar, gt = shuffle(x1, x2, H_mean, H_logvar, gt)
        for num_batch_i in range(int(num_batchs)):
            start_idx, end_idx = num_batch_i * batch_size, (num_batch_i + 1) * batch_size
            end_idx = min(num_samples, end_idx)
            batch_x1 = x1[start_idx: end_idx, ...]
            batch_x2 = x2[start_idx: end_idx, ...]
            batch_h_mean = H_mean[start_idx: end_idx, ...]
            batch_h_logvar = H_logvar[start_idx: end_idx, ...]
            batch_x1 = torch.from_numpy(batch_x1).float()
            batch_x2 = torch.from_numpy(batch_x2).float()
            batch_h_mean = torch.from_numpy(batch_h_mean).float()
            batch_h_logvar = torch.from_numpy(batch_h_logvar).float()

            # Adversarial ground truths
            valid = Variable(torch.FloatTensor(batch_x1.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(torch.FloatTensor(batch_x1.size(0), 1).fill_(0.0), requires_grad=False)

            if is_cuda:
                batch_x1 = batch_x1.cuda()
                batch_x2 = batch_x2.cuda()
                batch_h_mean = batch_h_mean.cuda()
                batch_h_logvar = batch_h_logvar.cuda()
                valid = valid.cuda()
                fake = fake.cuda()

            # step1: train ae
            para_ae_G = itertools.chain(net_ae1.parameters(), net_ae2.parameters())
            para_ae_D = itertools.chain(net_ae1_D.parameters(), net_ae2_D.parameters())
            opt_ae_G = optim.Adam(params=para_ae_G, lr=lr_ae)
            opt_ae_D = optim.Adam(params=para_ae_D, lr=lr_dg)

            z1_mean, z1_logvar, z1 = net_ae1.get_z_half(batch_x1)
            z2_mean, z2_logvar, z2 = net_ae2.get_z_half(batch_x2)
            g1, g2 = net_dg.get_g(batch_h_mean, batch_h_logvar)

            # train generator
            opt_ae_G.zero_grad()
            ae_recon = 0.5 * torch.nn.MSELoss()(batch_x1, net_ae1(batch_x1)) + 0.5 * torch.nn.MSELoss()(batch_x2, net_ae2(batch_x2))
            ae_degra = 0.5 * torch.nn.MSELoss()(g1.detach(), z1) + 0.5 * torch.nn.MSELoss()(g2.detach(), z2)
            ae_dav_G = 0.5 * torch.nn.BCELoss()(net_ae1_D(z1.detach()), valid) + 0.5 * torch.nn.BCELoss()(net_ae2_D(z2.detach()), valid)
            ae1_KL = -0.5 * torch.sum(1 + z1_logvar - z1_mean.pow(2) - z1_logvar.exp(), dim=1)
            ae2_KL = -0.5 * torch.sum(1 + z2_logvar - z2_mean.pow(2) - z2_logvar.exp(), dim=1)
            ae_loss = ae_recon + ae_degra + 0.0001 * torch.mean(ae1_KL + ae2_KL) + 0.01*ae_dav_G  # 这里为0.000001是因为torch.sum()比较大
            ae_loss.backward()
            opt_ae_G.step()

            # train discriminator
            opt_ae_D.zero_grad()
            ae_dav_D_1 = 0.5 * torch.nn.BCELoss()(net_ae1_D(z1.detach()), fake) + 0.5 * torch.nn.BCELoss()(net_ae2_D(z2.detach()), fake)
            ae_dav_D_2 = 0.5 * torch.nn.BCELoss()(net_ae1_D(g1.detach()), valid) + 0.5 * torch.nn.BCELoss()(net_ae2_D(g2.detach()), valid)
            ae_dav_D = ae_dav_D_1 + ae_dav_D_2
            ae_dav_D = 0.01 * ae_dav_D
            ae_dav_D.backward()
            opt_ae_D.step()

            # step2: train dg
            # fix h not update
            for i in range(30):
                para_dg_G = itertools.chain(net_dg.parameters())
                para_dg_D = itertools.chain(net_dg1_D.parameters(), net_dg2_D.parameters())
                opt_dg_G = optim.Adam(params=para_dg_G, lr=lr_dg)
                opt_dg_D = optim.Adam(params=para_dg_D, lr=lr_dg)

                _, _, z1 = net_ae1.get_z_half(batch_x1)
                _, _, z2 = net_ae2.get_z_half(batch_x2)
                g1, g2, h_mean, h_logvar = net_dg(z1, z2)

                # train generator
                opt_dg_G.zero_grad()
                dg_degra = 0.5 * torch.nn.MSELoss()(z1.detach(), g1) + 0.5 * torch.nn.MSELoss()(z2.detach(), g2)
                dg_KL = -0.5 * torch.sum(1 + h_logvar - h_mean.pow(2) - h_logvar.exp(), dim=1)
                dg_dav_G = 0.5 * torch.nn.BCELoss()(net_dg1_D(g1.detach()), valid) + 0.5 * torch.nn.BCELoss()(net_dg2_D(g2.detach()), valid)
                dg_loss = dg_degra + 0.0001 * torch.mean(dg_KL) + 0.01*dg_dav_G
                dg_loss.backward()
                opt_dg_G.step()

                # train discriminator
                opt_dg_D.zero_grad()
                dg_dav_D_1 = 0.5 * torch.nn.BCELoss()(net_dg1_D(g1.detach()), fake) + 0.5 * torch.nn.BCELoss()(net_dg2_D(g2.detach()), fake)
                dg_dav_D_2 = 0.5 * torch.nn.BCELoss()(net_dg1_D(z1.detach()), valid) + 0.5 * torch.nn.BCELoss()(net_dg2_D(z2.detach()), valid)
                dg_dav_D = dg_dav_D_1 + dg_dav_D_2
                dg_dav_D = 0.01 * dg_dav_D
                dg_dav_D.backward()
                opt_dg_D.step()

            # write updated h to H
            new_h_mean, new_h_logvar = net_dg.get_h(z1, z2)
            new_hh1 = z1
            new_hh2 = z2
            if is_cuda:
                new_h_mean = new_h_mean.cpu()
                new_h_logvar = new_h_logvar.cpu()
                new_hh1 = new_hh1.cpu()
                new_hh2 = new_hh2.cpu()
            H_mean[start_idx: end_idx, ...] = new_h_mean.detach().numpy()
            H_logvar[start_idx: end_idx, ...] = new_h_logvar.detach().numpy()

            output = "Epoch : {:.0f} -- Batch : {:.0f} ===> Total training loss = {:.4f} ".format((epoch),
                                                                                                  (num_batch_i + 1),
                                                                                                  ae_loss)
            print(output)

    # final result
    print_result(10, H_mean, gt)
