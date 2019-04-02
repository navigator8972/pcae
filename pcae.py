'''
PointCloudAutoEncoder
'''

from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader

from gmm_torch import GaussianMixture

import numpy as np

from pointnet import PointNetfeat, feature_transform_reguliarzer
from ChamferDistancePyTorch import chamfer_distance_with_batch
from visdom_utils import VisdomInterface


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PointCloudAutoEncoder(nn.Module):
    def __init__(self, n_points=2048, latent_size=64):
        super(PointCloudAutoEncoder, self).__init__()

        self.n_points = n_points
        self.latent_size = latent_size

        #prepare encoder net, use PointNet structure. 
        self.pointnet_feature = PointNetfeat(global_feat=True, feature_transform=False)
        self.encode_transform = nn.Linear(1024, latent_size)

        #prepare decoder net, full fc
        self.fc_feature = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, n_points*3)
        )

        def init_xavier_normal(m):
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
        self.fc_feature.apply(init_xavier_normal)

        return
    
    def encode(self, x):
        feat, trans, _ = self.pointnet_feature(x)
        return self.encode_transform(feat), trans
    
    def decode(self, z):
        x_hat = self.fc_feature(z).view(-1, 3, self.n_points)
        return x_hat
    
    def fit(self, dataset, batch_size=64, n_epoch=500, lr=5e-4, visdom=True, outf='./models'):
        train_loader=torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True
        )

        opt = optim.Adam(self.parameters(), lr=lr, weight_decay=0)

        if visdom:
            vis = VisdomInterface(port=8097)

        recons_epoch_loss = []
        transreg_epoch_loss = []
        def train_epoch(epoch):
            recons_batch_loss = []
            transreg_batch_loss = []
            for batch_idx, data in enumerate(train_loader):
                x = Variable(data).to(device)
                x_hat, z, trans = self.forward(x)

                chamfer_loss = self.reconstruction_loss(x, x_hat)
                trans_reg_loss = feature_transform_reguliarzer(trans)
                loss = chamfer_loss + trans_reg_loss
                
                recons_batch_loss.append(chamfer_loss.item())
                transreg_batch_loss.append(trans_reg_loss.item())

                opt.zero_grad()
                loss.backward()
                opt.step()

                if batch_idx % 50 == 0 and True:        #suppress iteration output now
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Chamfer Loss: {:.6f}; Trans Reg Loss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), chamfer_loss.item(), trans_reg_loss.item()
                        ))

            recons_epoch_loss.append(np.mean(recons_batch_loss))
            transreg_epoch_loss.append(np.mean(transreg_batch_loss))
            return

        for i in range(n_epoch):
            train_epoch(i)
            if visdom:
                vis.update_losses(recons_epoch_loss, transreg_epoch_loss)

            if i == 0 or (i+1) % 50 == 0:
                torch.save(self.state_dict(), '%s/pcae_epoch_%d.pth' % (outf, i))
        return
    
    def forward(self, x):
        z, trans = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z, trans

    def reconstruction_loss(self, x, x_hat):
        chamfer_loss_1, chamfer_loss_2 = chamfer_distance_with_batch(x, x_hat)
        return chamfer_loss_1.mean() + chamfer_loss_2.mean()

    def load_model(self, mdl_fname, cuda=False):
        if cuda:
            self.load_state_dict(torch.load(mdl_fname))
            self.cuda()
        else:
            self.load_state_dict(torch.load(mdl_fname, map_location='cpu'))
        self.eval()

class PointCloudGenerativeGMM(nn.Module):
    def __init__(self, auto_encoder, n_components=3, covar_type='diag'):
        super(PointCloudGenerativeGMM, self).__init__()

        #care must be taken about the train() or eval() state of assigned autoencoder, as it uses pointnet features which contain batchnorm
        #i think by default it should use tracked statistics instead of the input batches
        self.auto_encoder = auto_encoder
        self.gmm = GaussianMixture(n_components=n_components, n_features=auto_encoder.latent_size, var_type=covar_type)
        return
    
    def fit_gmm(self, data, raw_feature=True, n_iter=1000, kmeans_init=True):
        if raw_feature:
            x = self.auto_encoder.encode(data)[0]
        else:
            x = data
        
        print('Fitting GMM in the latent space...')
        self.gmm.fit(x, n_iter=n_iter, kmeans_init=kmeans_init)
        print('Likelihood at convergence: {}'.format(np.average(self.gmm.likelihood(x))))
        return
    
    def load_model(self, mdl_fname, cuda=False):
        if cuda:
            self.load_state_dict(torch.load(mdl_fname))
            self.cuda()
        else:
            self.load_state_dict(torch.load(mdl_fname, map_location='cpu'))
        self.eval()
        return
    
    def save_model(self, mdl_fname):
        if not mdl_fname.endswith('.pth'):
            mdl_fname += '.pth'
        torch.save(self.state_dict(), mdl_fname)
        return
    
    def sample(self, n_samples=1):
        z = self.gmm.sample(n_samples=n_samples)
        return self.auto_encoder.decode(z)
    
    def fit_shapes(self, input_shapes, n_iters=10, lr=.1, ll_weight=5., init_from_input=True, verbose=False):
        #find latent variable leading to reconstructions that: 1. map input shapes with a small one-directional chamfer loss; 2. maximize the likelihood of GMM

        if init_from_input:
            encode_z = self.auto_encoder.encode(input_shapes)[0]
        else:
            encode_z = self.gmm.sample(n_samples=input_shapes.shape[0])
        encode_z = Variable(encode_z, requires_grad=True)
        
        #grad,  = torch.autograd.grad(tol_loss, encode_z, create_graph=True)

        #try adam optimizer
        solver = optim.Adam([encode_z], lr=lr, weight_decay=0)
        for i in range(n_iters):
            decode_x = self.auto_encoder.decode(encode_z)
            loss_input_to_decode, _ = chamfer_distance_with_batch(input_shapes, decode_x)     #this allows small cost when decoded_x contains input_shapes
            gmm_likelikelihood = self.gmm.likelihood(encode_z)
            tol_loss = loss_input_to_decode.sum() - ll_weight* gmm_likelikelihood.sum()
            if verbose:
                print('Iteration {} - Reconstruction Loss/GMM Likelihood: {}/{}'.format(i+1, loss_input_to_decode.sum().item(), gmm_likelikelihood.sum().item()))
            solver.zero_grad()
            tol_loss.backward()
            solver.step()
        
        return self.auto_encoder.decode(encode_z)

class PointCloudDataSet(Dataset):
    def __init__(self, npz_file, feature_fname='chair', norm=True, train=True, float_type=np.float32):
        if isinstance(feature_fname, str):
            full_data = np.load(npz_file)['data'].item()[feature_fname].astype(float_type)
            if train:
                self.pc_data = full_data[:int(full_data.shape[0]*.8)]
            else:
                self.pc_data = full_data[int(full_data.shape[0]*.8):]
        elif isinstance(feature_fname, list):
            full_data = np.load(npz_file)['data'].item()
            if train:
                self.pc_data = np.concatenate([full_data[k][:int(full_data[k].shape[0]*.8)].astype(float_type) for k in feature_fname], axis=0)
            else:
                self.pc_data = np.concatenate([full_data[k][int(full_data[k].shape[0]*.8):].astype(float_type) for k in feature_fname], axis=0)
        else:
            raise NotImplementedError

        return 
        
    def __len__(self):
        return len(self.pc_data)

    def __getitem__(self, idx):
        return self.pc_data[idx]

if __name__ == '__main__':
    # model = PointCloudAutoEncoder(n_points=2048, latent_size=8)
    # if torch.cuda.is_available():
    #     model.cuda()
    
    # # dummy_data = torch.randn(8, 3, 2048).to(device)
    # # x_hat, z, trans = model(dummy_data)
    # # reconstruction_loss = model.reconstruction_loss(dummy_data, x_hat)
    # #dataset
    # dataset = PointCloudDataSet('../data/3DShapeNet_PointCloud2048.npz')
    # print('Number of data:{}'.format(len(dataset)))

    # model.fit(dataset, batch_size=32)

    model = PointCloudAutoEncoder(n_points=1024, latent_size=8)
    if torch.cuda.is_available():
        model.cuda()
    #dataset = PointCloudDataSet('../data/ycb_18_dataset.npz', feature_fname='play_go_rainbow_stakin_cups_1_yellow')
    ycb_objs = ['black_and_decker_lithium_drill_driver', 'campbells_condensed_tomato_soup',
                'domino_sugar_1lb', 'morton_salt_shaker', 'rubbermaid_ice_guard_pitcher_blue', 'block_of_wood_6in',
                'cheerios_14oz', 'frenchs_classic_yellow_mustard_14oz', 'play_go_rainbow_stakin_cups_1_yellow',
                'soft_scrub_2lb_4oz', 'brine_mini_soccer_ball', 'clorox_disinfecting_wipes_35', 'melissa_doug_farm_fresh_fruit_banana',
                'play_go_rainbow_stakin_cups_2_orange', 'sponge_with_textured_cover', 'comet_lemon_fresh_bleach',
                'melissa_doug_farm_fresh_fruit_lemon', 'pringles_original']
    dataset = PointCloudDataSet('../data/ycb_18_dataset.npz', feature_fname=ycb_objs)   #train over all the types
    print('Number of data:{}'.format(len(dataset)))

    model.fit(dataset, batch_size=32)