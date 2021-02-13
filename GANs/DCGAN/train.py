import torch 
from torch import nn,optim
import torchvision 
from torchvision import datasets,transforms
from torch.utils.data import DataLoader 
from torch.utils.tensorboard import SummaryWriter 
from model import Discriminator, Generator, initialize_weights

BATCH_SIZE = 128
EPOCHS = 5
LR = 0.0002
CHANNELS_IMG = 1
FEATURES_D = 64
Z_DIM = 100
FEATURES_G = 64
IMAGE_SIZE = 64
step = 0


dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

t = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)])
])

# data = datasets.ImageFolder('/users/gursi/desktop/celeb_dataset', transform=t)
# loader = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)

data = datasets.MNIST('./data', transform=t, download=True, train=True)
loader = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)

disc = Discriminator(CHANNELS_IMG, FEATURES_D)
disc.to(dev)
initialize_weights(disc)
opt_disc = optim.Adam(disc.parameters(), lr = LR, betas=(0.5, 0.999))

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_G)
gen.to(dev)
initialize_weights(gen)
opt_gen = optim.Adam(gen.parameters(), lr = LR, betas=(0.5, 0.999))

criterion = nn.BCELoss()

fixed_noise = torch.randn((32, Z_DIM, 1, 1)).to(dev)
writer_real = SummaryWriter(log_dir='runs/real')
writer_fake = SummaryWriter(log_dir='runs/fake')

gen.train()
disc.train()

for epoch in range(EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(dev)
        noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(dev)

        # Train disc 
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))

        fake = gen(noise)
        disc_fake = disc(fake).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        
        loss_D = (loss_disc_fake + loss_disc_real) / 2
        disc.zero_grad()
        loss_D.backward(retain_graph = True)
        opt_disc.step()

        # train gen
        output = disc(fake).reshape(-1)
        loss_G = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_G.backward()
        opt_gen.step()

        if batch_idx % 1 == 0:
            print(
                f"Epoch [{epoch}/{EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_D:.4f}, loss G: {loss_G:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1