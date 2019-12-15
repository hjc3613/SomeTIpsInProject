import torch
input = torch.randn(20, 4, 50, 60) #B, C, H, W
# batch normal是在每一个channel层中计算均值，方差，然后利用均值、方差修正该层的输入值
channel_1 = input[:, 0, :, :]
mean_1 = channel_1.mean()
std_1 = channel_1.std()
batched_1 = (channel_1 - mean_1)/std_1
