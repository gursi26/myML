#%%
import torch 
from torch import nn,optim
from torch.utils import data 
from torchvision import datasets,transforms
from torch.utils.data import DataLoader 
from torch.utils.tensorboard import SummaryWriter

#%%
class Discriminator(nn.Module):

    def __init__(self, img_dim):
        super(Discriminator,self).__init__()

        self.disc = nn.Sequential(
            nn.Linear(img_dim,128),
            nn.LeakyReLU(0.1),
            nn.Linear(128,1),
            nn.Sigmoid() # Ensure output is 0 or 1 (fake/real)
        )

    def forward(self,x):
        return self.disc(x)


class Generator(nn.Module):
    # z_dim is noise dimension, img_dim is output img dim
    def __init__(self, z_dim, img_dim):
        super(Generator,self).__init__()

        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256,img_dim),
            nn.Tanh()
        )

    def forward(self,x):
        return self.gen(x)


#%%
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
lr = 0.003
z_dim = 64
image_dim = 28 * 28 * 1
batch_size = 32
epochs = 10

#%%
disc = Discriminator(image_dim).to(dev)
gen = Generator(z_dim, image_dim).to(dev)
fixed_noise = torch.randn((batch_size, z_dim)).to(dev)
t = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

dataset = datasets.MNIST(root='./data', download=True, transform=t)
loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)

criterion = nn.BCELoss()

writer_fake = SummaryWriter(f'runs/GAN_MNIST/fake')
writer_real = SummaryWriter(f'runs/GAN_MNIST/real')
step = 0

#%%
for epoch in range(epochs):
    for batch_idx, (real, _) in enumerate(loader):

        real = real.view(-1,784).to(dev)
        batch_size = real.shape[0]

        # Train discriminator
        noise = torch.randn((batch_size, z_dim)).to(dev)
        fake = gen(noise)

        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))

        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        lossD = (lossD_fake + lossD_real) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        # Train generator
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0 : 
            print(f'Epoch : [{epoch/epochs}] Loss D : {lossD:.4f}, Loss G {lossG:.4f}')

#%%










