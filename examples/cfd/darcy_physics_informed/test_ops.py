import torch

from ops import dx

# Initialize the array with zeros
array = torch.zeros(100, 100)

# Assign values based on the conditions
array[:, :50] = -0.5  # x from -1 to 0
array[:, 50:] = 0.5   # x from 0 to 1

array = array[None, None, ...]
print(array.shape)

dxarr = dx(array, dx=1/100, channel=0, dim=1, order=1, padding="zeros")
dyarr = dx(array, dx=1/100, channel=0, dim=0, order=1, padding="zeros")

print(dxarr.shape)


import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(25, 5))
im = ax[0].imshow(array[0, 0, ...].cpu().numpy())
plt.colorbar(im, ax=ax[0])
im = ax[1].imshow(dxarr[0, 0, ...].cpu().numpy())
plt.colorbar(im, ax=ax[1])
im = ax[2].imshow(dyarr[0, 0, ...].cpu().numpy())
plt.colorbar(im, ax=ax[2])

plt.savefig("testing.png")
