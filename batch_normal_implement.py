import torch
input = torch.randn(20, 4, 50, 60) #B, C, H, W
# batch normal， 对每一个channel，shape=(20, 50, 60)，计算each_channel.mean(), each_channel_std
mean = input.mean(dim=1)
std = input.std(dim=1)
# batch normal计算
output = (input-mean)/std
