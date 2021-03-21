import torch 

# Generator model
#-------------------------------------------------------------------------------------------------#

class ConvLayer(torch.nn.Module):

    def __init__(self, in_ch, out_ch, k=3, s=1, p=0):
        super(ConvLayer,self).__init__()
        reflection_pad = k // 2
        self.conv = torch.nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)
        self.ref_pad = torch.nn.ReflectionPad2d(reflection_pad)
        self.norm = torch.nn.InstanceNorm2d(out_ch, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        out = self.ref_pad(x)
        out = self.conv(out)
        out = self.norm(out)
        out = self.relu(out)
        return out 

class Upsample(torch.nn.Module):

    def __init__(self, in_ch, out_ch, k=3, s=2, p=0):
        super(Upsample,self).__init__()
        self.upsample = torch.nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)
        self.conv = torch.nn.Conv2d(out_ch, out_ch, kernel_size=2, stride=1, padding=0)
        self.norm = torch.nn.InstanceNorm2d(out_ch, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        out = self.upsample(x)
        out = self.conv(out)
        out = self.norm(out)
        out = self.relu(out)
        return out

class Generator(torch.nn.Module):

    def __init__(self):
        super(Generator,self).__init__()

        self.conv1 = ConvLayer(3,112)
        self.conv2 = ConvLayer(112,112)
        self.downsample1 = ConvLayer(112,112,s=2)
        
        self.conv3 = ConvLayer(112,224)
        self.conv4 = ConvLayer(224,224)
        self.downsample2 = ConvLayer(224,224,s=2)

        self.conv5 = ConvLayer(224,448)
        self.conv6 = ConvLayer(448,448)
        self.downsample3 = ConvLayer(448,448,s=2)

        self.conv7 = ConvLayer(448,448)
        self.conv8 = ConvLayer(448,448)

        self.upsample1 = Upsample(448,448)
        self.conv9 = ConvLayer(896,224)
        self.conv10 = ConvLayer(224,224)

        self.upsample2 = Upsample(224,224)
        self.conv11 = ConvLayer(448,112)
        self.conv12 = ConvLayer(112,112)

        self.upsample3 = Upsample(112,112)
        self.conv13 = ConvLayer(224,112)
        self.conv14 = ConvLayer(112,112)

        self.output = ConvLayer(112,3)

    def forward(self,x):

        out = self.conv1(x)
        skip1 = self.conv2(out)
        out = self.downsample1(skip1)

        out = self.conv3(out)
        skip2 = self.conv4(out)
        out = self.downsample2(skip2)

        out = self.conv5(out)
        skip3 = self.conv6(out)
        out = self.downsample3(skip3)

        out = self.conv7(out)
        out = self.conv8(out)

        out = self.upsample1(out)
        out = torch.cat([out,skip3], dim=1)
        out = self.conv9(out)
        out = self.conv10(out)

        out = self.upsample2(out)
        out = torch.cat([out,skip2], dim=1)
        out = self.conv11(out)
        out = self.conv12(out)

        out = self.upsample3(out)
        out = torch.cat([out,skip1], dim=1)
        out = self.conv13(out)
        out = self.conv14(out)

        out = self.output(out)
        return out


# Discriminator model
#-------------------------------------------------------------------------------------------------#

class DiscConv(torch.nn.Module):

    def __init__(self, in_c, out_c, ksize=3, s=2, p=0):
        super(DiscConv,self).__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_c, 
            out_channels=out_c, 
            kernel_size=ksize,
            stride=s,
            padding=p
        )
        self.bn = torch.nn.BatchNorm2d(out_c)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))



class Discriminator(torch.nn.Module):

    def __init__(self, input_size):
        super(Discriminator,self).__init__()
        self.conv1 = DiscConv(3,64)
        self.conv2 = DiscConv(64,128)
        self.conv3 = DiscConv(128,256)
        self.conv4 = DiscConv(256,64)

        self.output_size = self.calculate_output_size(input_size)

        self.fc1 = torch.nn.Linear(self.output_size,500)
        self.fc2 = torch.nn.Linear(500,100)
        self.fc3 = torch.nn.Linear(100,1)

        self.dropout = torch.nn.Dropout(p=0.3)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def calculate_output_size(self, input_size):
        ksize = 3
        stride = 2
        padding = 0 
        num_layers = 4
        final_channels = 64
        size = input_size

        for i in range(num_layers):
            size = int(((size - ksize + (2*padding))/stride) + 1)

        return int(size * size * final_channels)

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.dropout(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.dropout(out)
        
        out = torch.flatten(out, start_dim=1)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out


#-------------------------------------------------------------------------------------------------#


def test():
    model = Generator()
    noise = torch.zeros((5,3,256,256))
    print('Input shape : ', noise.shape)
    output = model.forward(noise)
    print('Generator output shape : ', output.shape)
    disc = Discriminator(256)
    disc_output = disc.forward(output)
    print('Discriminator output shape : ', disc_output.shape)
