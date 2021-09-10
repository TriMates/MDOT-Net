import torch

from model import VAE, Fusion
from utils.Dataset import Dataset
from utils.print_result import print_result

# set gpu device
torch.cuda.set_device(0)

if __name__ == "__main__":
    is_cuda = False   # True for training
    is_cuda = is_cuda and torch.cuda.is_available()

    batch_size = 2000

    latent_h = 64
    latent_z = 200

    data_h = Dataset('handwritten_2views')
    x1, x2, gt = data_h.load_data()
    x1 = data_h.normalize(x1, 0)
    x2 = data_h.normalize(x2, 0)

    net_ae1 = VAE(dim_data=x1.shape[1], dim_latents=latent_z)
    net_ae2 = VAE(dim_data=x2.shape[1], dim_latents=latent_z)
    net_dg = Fusion(h_dim=latent_h, dim_latents=latent_z, batch_size=batch_size, is_cuda=is_cuda)

    # load model
    PATH = 'checkpoint/model.pth.tar'
    checkpoint = torch.load(PATH)
    net_ae1.load_state_dict(checkpoint['modelAE1'])
    net_ae2.load_state_dict(checkpoint['modelAE2'])
    net_dg.load_state_dict(checkpoint['modelDG'])

    ###  test
    test_x1 = torch.from_numpy(x1).float()
    test_x2 = torch.from_numpy(x2).float()
    net_ae1.eval()
    net_ae2.eval()
    net_dg.eval()
    _, _, z1 = net_ae1.get_z_half(test_x1)
    _, _, z2 = net_ae2.get_z_half(test_x2)
    h_mean, _ = net_dg.get_h(z1, z2)
    H = h_mean.detach().numpy()

    # print_result
    print_result(10, H, gt)
