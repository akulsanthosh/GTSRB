import torchvision.transforms as transforms
import torchvision.utils as utils
import os
from PIL import Image
import numpy as np

def image_loader(loader, base):
    for counter, [image, label] in enumerate(loader):
        unorm_image = unorm(image)
        save_data(unorm_image, label, base, counter)

def unorm(image):
    image = utils.make_grid(image)
    image = image / 2 + 0.5
    # print(image.shape)
    npimage = image.numpy()
    npimage = np.transpose(npimage, (1, 2, 0))
    print(npimage[0][0])
    return npimage

def save_data(image, label, base, counter):
    print(label.item())
    path = os.path.join(base, str(label.item()))
    os.makedirs(path, exist_ok = True)
    im = Image.fromarray((image * 255).astype(np.uint8))
    im.save(os.path.join(path,f"{counter}.png"))