"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn


class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
        @notice: avoid in-place ops.
        https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x)  # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp


class ApplyStyle(nn.Module):
    """
    @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
    """

    def __init__(self, latent_size, channels):
        super(ApplyStyle, self).__init__()
        self.linear = nn.Linear(latent_size, channels * 2)

    def forward(self, x, latent):
        style = self.linear(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]
        # x = x * (style[:, 0] + 1.) + style[:, 1]
        x = x * (style[:, 0] * 1 + 1.0) + style[:, 1] * 1
        return x


class ResnetBlock_Adain(nn.Module):
    def __init__(self, dim, latent_size, padding_type, activation=nn.ReLU(True)):
        super(ResnetBlock_Adain, self).__init__()

        p = 0
        conv1 = []
        if padding_type == "reflect":
            conv1 += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv1 += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv1 += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), InstanceNorm()]
        self.conv1 = nn.Sequential(*conv1)
        self.style1 = ApplyStyle(latent_size, dim)
        self.act1 = activation

        p = 0
        conv2 = []
        if padding_type == "reflect":
            conv2 += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv2 += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv2 += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), InstanceNorm()]
        self.conv2 = nn.Sequential(*conv2)
        self.style2 = ApplyStyle(latent_size, dim)

    def forward(self, x, dlatents_in_slice):
        y = self.conv1(x)
        y = self.style1(y, dlatents_in_slice)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.style2(y, dlatents_in_slice)
        out = x + y
        return out


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, deep=False, epsilon=None):
        super(Generator, self).__init__()
        activate_layer = nn.ReLU(True)
        norm_layer = nn.BatchNorm2d
        self.deep = deep

        self.first_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, kernel_size=7, padding=0),
            norm_layer(64),
            activate_layer,
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            norm_layer(128),
            activate_layer,
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            norm_layer(256),
            activate_layer,
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            norm_layer(512),
            activate_layer,
        )
        self.down4 = nn.Sequential(  # deep
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            norm_layer(512),
            activate_layer,
        )

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            activate_layer,
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            activate_layer,
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            activate_layer,
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            activate_layer,
        )
        self.last_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        )

        self.epsilon = epsilon

    def forward(self, input: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
        x = input  # 3 * 224 * 224

        x = self.first_layer(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        if self.deep:
            x = self.down4(x)
            x = self.up4(x)

        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        x = self.last_layer(x)
        x = (x + 1) / 2

        clamped_out = []
        for i in range(len(self.epsilon)):
            clamped_channel = torch.clamp(
                x[:, i, :, :],
                min=input[:, i, :, :] - self.epsilon[i],
                max=input[:, i, :, :] + self.epsilon[i],
            )
            clamped_out.append(clamped_channel)

        out = torch.stack(clamped_out, dim=1)
        return out  # modified face

    def decoder(self, input):
        x = input

        if self.deep:
            x = self.up4(x)

        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        x = self.last_layer(x)
        x = (x + 1) / 2

        return x


class Generator_Adain_Upsample(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        latent_size,
        n_blocks=6,
        deep=False,
        norm_layer=nn.BatchNorm2d,
        padding_type="reflect",
    ):
        assert n_blocks >= 0
        super(Generator_Adain_Upsample, self).__init__()
        activation = nn.ReLU(True)
        self.deep = deep

        self.first_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, kernel_size=7, padding=0),
            norm_layer(64),
            activation,
        )
        ### downsample
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            norm_layer(128),
            activation,
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            norm_layer(256),
            activation,
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            norm_layer(512),
            activation,
        )
        if self.deep:
            self.down4 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                norm_layer(512),
                activation,
            )

        ### resnet blocks
        BN = []
        for i in range(n_blocks):
            BN += [
                ResnetBlock_Adain(
                    512,
                    latent_size=latent_size,
                    padding_type=padding_type,
                    activation=activation,
                )
            ]
        self.BottleNeck = nn.Sequential(*BN)

        if self.deep:
            self.up4 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                activation,
            )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            activation,
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            activation,
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            activation,
        )
        self.last_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        )

    def forward(self, input, dlatents):
        x = input  # 3*224*224

        skip1 = self.first_layer(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        if self.deep:
            skip4 = self.down3(skip3)
            x = self.down4(skip4)
        else:
            x = self.down3(skip3)

        for i in range(len(self.BottleNeck)):
            x = self.BottleNeck[i](x, dlatents)

        if self.deep:
            x = self.up4(x)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        x = self.last_layer(x)
        x = (x + 1) / 2

        return x

    def encoder(self, input):
        x = input  # 3*224*224

        x = self.first_layer(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        if self.deep:
            x = self.down4(x)

        return x


class Defense_Discriminator(nn.Module):
    def __init__(self):
        super(Defense_Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=5, stride=2), nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=5, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.conv5 = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size=5, stride=2))
        self.conv6 = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size=4))

        self.tail = nn.Linear(1024, 1)

    def forward(self, input):
        x = input
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(input.shape[0], -1)
        output = self.tail(x)

        return output


class Discriminator(nn.Module):
    def __init__(self, input_nc, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(Discriminator, self).__init__()

        kw = 4
        padw = 1
        self.down1 = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=kw, stride=2, padding=padw),
            norm_layer(128),
            nn.LeakyReLU(0.2, True),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=kw, stride=2, padding=padw),
            norm_layer(256),
            nn.LeakyReLU(0.2, True),
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=kw, stride=2, padding=padw),
            norm_layer(512),
            nn.LeakyReLU(0.2, True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=kw, stride=1, padding=padw),
            norm_layer(512),
            nn.LeakyReLU(0.2, True),
        )

        if use_sigmoid:
            self.conv2 = nn.Sequential(
                nn.Conv2d(512, 1, kernel_size=kw, stride=1, padding=padw), nn.Sigmoid()
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(512, 1, kernel_size=kw, stride=1, padding=padw)
            )

    def forward(self, input):
        out = []
        x = self.down1(input)
        out.append(x)
        x = self.down2(x)
        out.append(x)
        x = self.down3(x)
        out.append(x)
        x = self.down4(x)
        out.append(x)
        x = self.conv1(x)
        out.append(x)
        x = self.conv2(x)
        out.append(x)

        return out
