#--------------------------------------------#
#---------- Importacion de  modulos----------#
#--------------------------------------------#

import os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR

# Eleccion manual de la semilla aleatoria
torch.manual_seed(123)


#----------------------------------------------#
#---------- Definicion de parámetros ----------#
#----------------------------------------------#

# batch_size (tamaño de lotes) = 100 (default)
batch_size = 100

# epochs (épocas) = 100 (default)
epochs = 100

# lr (learning rate) = 0.0001 (default)
lr = 0.0001

# dim_h (hidden dimension) = 128 (default)
dim_h = 128

# n_z (hidden dimension z) = 8 (default)
n_z = 8

# lambda (coef. regularizacion MMD) = 10 (default)
LAMBDA = 10

# n_channel (canales de entrada (color = 3, bn = 1)) = 1 (default)
n_channel = 1

# sigma (varianza de la hidden dimension) = 1 (default)
sigma = 1


#----------------------------------------------------#
#---------- Descarga del conjunto de datos ----------#
#----------------------------------------------------#

trainset = MNIST(root = './data/',
                 train = True,
                 transform = transforms.ToTensor(),
                 download = False)

testset = MNIST(root = './data/',
                train = False,
                transform = transforms.ToTensor(),
                download = False)


#-------------------------------------------------#
#---------- Carga del conjunto de datos ----------#
#-------------------------------------------------#

train_loader = DataLoader(dataset = trainset,
                          batch_size = batch_size,
                          shuffle = True)

test_loader = DataLoader(dataset = testset,
                         batch_size = 104,
                         shuffle = False)

# Para volver a actualizar los parámetros
def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

# Para dejar de actualizar los parámetros
def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


#-------------------------------------------------#
#---------- Clases del modelo de la red ----------#
#-------------------------------------------------#

# Codificador
class Encoder(nn.Module):
    
    # Constructor
    def __init__(self, batch_size, epochs, lr, dim_h, n_z, n_channel, LAMBDA, sigma):
        super(Encoder, self).__init__()

        self.n_channel = n_channel
        self.dim_h = dim_h
        self.n_z = n_z

        # Estructura secuencial de capas de la red
        # Conv2d: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        # BatchNorm2d: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
        # torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        # ReLU: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        # torch.nn.ReLU(inplace=False)
        self.main = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias = False),
            nn.ReLU(True),
            
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias = False),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.ReLU(True),
        )
        self.fc = nn.Linear(self.dim_h * (2 ** 3), self.n_z)

    # Alimentación de la red mediante imágenes
    def forward(self, x):
        x = self.main(x)
        x = x.squeeze()
        x = self.fc(x)
        return x

# Decodificador
class Decoder(nn.Module):
    
    # Constructor
    def __init__(self, batch_size, epochs, lr, dim_h, n_z, n_channel, LAMBDA, sigma):
        super(Decoder, self).__init__()

        self.n_channel = n_channel
        self.dim_h = dim_h
        self.n_z = n_z

        # Preparación de las entradas
        self.proj = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * 7 * 7),
            nn.ReLU()
        )

        # Estructura secuencial de capas de la red
        # Conv2d: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
        # torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
        # BatchNorm2d: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
        # torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        # ReLU: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        # torch.nn.ReLU(inplace=False)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.dim_h * 2, 1, 4, stride = 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.view(-1, self.dim_h * 8, 7, 7)
        x = self.main(x)
        return x


#-----------------------------------------------------------#
#---------- Definicion de los nucleos para el mmd ----------#
#-----------------------------------------------------------#

def imq_kernel(X: torch.Tensor, Y: torch.Tensor, h_dim: int):

    # Recuperamos el tamaño del lote
    batch_size = X.size(0)

    # Definimos la forma de medir la norma
    norms_x = X.pow(2).sum(1, keepdim=True)  
    prods_x = torch.mm(X, X.t()) 
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(1, keepdim=True)
    prods_y = torch.mm(Y, Y.t())
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / (batch_size)
        stats += res1 - res2

    return stats

#-----------------------------------------------------------#
#---------- Inicializacion para el entrenamiento  ----------#
#-----------------------------------------------------------#

# Inicializamos codificador y decodificador
encoder = Encoder(batch_size, epochs, lr, dim_h, n_z, n_channel, LAMBDA, sigma)
decoder = Decoder(batch_size, epochs, lr, dim_h, n_z, n_channel, LAMBDA, sigma)
criterion = nn.MSELoss()

encoder.train()
decoder.train()

# Utilizamos la paralelizacion en GPU si es posible
if torch.cuda.is_available():
    encoder, decoder = encoder.cuda(), decoder.cuda()

one = torch.Tensor([1])
mone = one * -1

if torch.cuda.is_available():
    one = one.cuda()
    mone = mone.cuda()

# Optimizadores (ADAM)
enc_optim = optim.Adam(encoder.parameters(), lr = lr)
dec_optim = optim.Adam(decoder.parameters(), lr = lr)

enc_scheduler = StepLR(enc_optim, step_size = 30, gamma = 0.5)
dec_scheduler = StepLR(dec_optim, step_size = 30, gamma = 0.5)

#----------------------------------------------#
#---------- Entrenamiento de la red  ----------#
#----------------------------------------------#

for epoch in range(epochs):
    step = 0
    # tqdm es para mostrar una barra de progreso
    for (images, _) in tqdm(train_loader):

        if torch.cuda.is_available():
            images = images.cuda()

        enc_optim.zero_grad()
        dec_optim.zero_grad()

        # ======== Entrenamiento del generador ======== #

        batch_size = images.size()[0]

        z = encoder(images)
        x_recon = decoder(z)

        recon_loss = criterion(x_recon, images)

        # ======== Pérdida del núcleo mmd ======== #

        z_fake = Variable(torch.randn(images.size()[0], n_z) * sigma)
        if torch.cuda.is_available():
            z_fake = z_fake.cuda()

        z_real = encoder(images)

        mmd_loss = imq_kernel(z_real, z_fake, h_dim=encoder.n_z)
        mmd_loss = mmd_loss / batch_size

        total_loss = recon_loss + mmd_loss
        total_loss.backward()

        enc_optim.step()
        dec_optim.step()

        step += 1

        if (step + 1) % 300 == 0:
            print("Epoch: [%d/%d], Step: [%d/%d], Reconstruction Loss: %.4f, MMD Loss %.4f" %
                  (epoch + 1, epochs, step + 1, len(train_loader), recon_loss.data.item(),
                   mmd_loss.item()))

    if (epoch + 1) % 1 == 0:
        batch_size = 104
        test_iter = iter(test_loader)
        test_data = next(test_iter)

        z_real = encoder(Variable(test_data[0]).cuda())
        reconst = decoder(z_real).cpu().view(batch_size, 1, 28, 28)
        sample = decoder(torch.randn_like(z_real)).cpu().view(batch_size, 1, 28, 28)

        if not os.path.isdir('./data/reconst_images'):
            os.makedirs('data/reconst_images')
            os.makedirs('data/reconst_images/Reconstruccion')
            os.makedirs('data/reconst_images/Muestreo')

        save_image(test_data[0].view(-1, 1, 28, 28), './data/reconst_images/wae_mmd_input.png')
        save_image(reconst.data, './data/reconst_images/Reconstruccion/wae_mmd_images_%d.png' % (epoch + 1))
        save_image(sample.data, './data/reconst_images/Muestreo/wae_mmd_samples_%d.png' % (epoch + 1))

