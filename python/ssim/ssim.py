import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error


def compute_ssim(baseImage, changedImage, max):
    baseMean = np.mean(baseImage)
    baseStdev = np.std(baseImage)

    changedMean = np.mean(changedImage)
    changedStdev = np.std(changedImage)

    baseVariance = baseStdev * baseStdev
    changedVariance = changedStdev * changedStdev

    #// Default values according to wikipedia
    k1 = 0.01
    k2 = 0.03

    c1 = (k1 * max) ** 2
    c2 = (k2 * max) ** 2
    c3 = c2 / 2

    luminance = (2 * (baseMean * changedMean) + c1) / (baseMean * baseMean + changedMean * changedMean + c1)

    contrast = (2 * baseVariance * changedVariance + c2) / (baseVariance * baseVariance + changedVariance * changedVariance + c2)

    crosscorrelation = 0

    baseFlat = baseImage.flatten()
    changedFlat = changedImage.flatten()

    for i in range(len(baseFlat)):
        crosscorrelation += (baseFlat[i] - baseMean) * (changedFlat[i] - changedMean)

    crosscorrelation = crosscorrelation / baseImage.size

    structure = (crosscorrelation + c3) / (baseStdev * changedStdev + c3)

    print(f"L:{luminance} C:{contrast} S:{structure} ")

    return (luminance * contrast * structure + 1) / 2




img = img_as_float(data.camera())
rows, cols = img.shape

noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
rng = np.random.default_rng()
noise[rng.random(size=noise.shape) > 0.5] *= -1

img_noise = img + noise
img_const = img + abs(noise)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4),
                         sharex=True, sharey=True)
ax = axes.ravel()

mse_none = mean_squared_error(img, img)
ssim_none = ssim(img, img, data_range=img.max() - img.min())

print(compute_ssim(img, img, 1), ssim(img, img, data_range=1))
print(compute_ssim(img, img_noise, 1))
print(compute_ssim(img, img_const, 1))

mse_noise = mean_squared_error(img, img_noise)
ssim_noise = ssim(img, img_noise,
                  data_range=img_noise.max() - img_noise.min())

mse_const = mean_squared_error(img, img_const)
ssim_const = ssim(img, img_const,
                  data_range=img_const.max() - img_const.min())

ax[0].imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[0].set_xlabel(f'MSE: {mse_none:.2f}, SSIM: {ssim_none:.2f}')
ax[0].set_title('Original image')

ax[1].imshow(img_noise, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[1].set_xlabel(f'MSE: {mse_noise:.2f}, SSIM: {ssim_noise:.2f}')
ax[1].set_title('Image with noise')

ax[2].imshow(img_const, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[2].set_xlabel(f'MSE: {mse_const:.2f}, SSIM: {ssim_const:.2f}')
ax[2].set_title('Image plus constant')

plt.tight_layout()
plt.show()
