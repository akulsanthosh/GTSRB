from torchvision import datasets, transforms

brightness = transforms.Compose([
        transforms.ColorJitter(brightness=3),
])

saturation = transforms.Compose([
    transforms.ColorJitter(saturation=5),
])

contrast = transforms.Compose([
    transforms.ColorJitter(contrast=5),
])

hue = transforms.Compose([
    transforms.ColorJitter(hue=0.4),
])

rotate = transforms.Compose([
    transforms.RandomRotation(15),
])

HVflip = transforms.Compose([
    transforms.RandomHorizontalFlip(1),
    transforms.RandomVerticalFlip(1),
])

Hflip = transforms.Compose([
    transforms.RandomHorizontalFlip(1),
])

Vflip = transforms.Compose([
    transforms.RandomVerticalFlip(1),
])

shear = transforms.Compose([
    transforms.RandomAffine(degrees = 15,shear=2),
])

translate = transforms.Compose([
    transforms.RandomAffine(degrees = 15,translate=(0.1,0.1)),
])

center = transforms.Compose([
    transforms.CenterCrop(32),
])

grayscale = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
])

transform_list = [None, brightness, saturation, contrast, hue, rotate, HVflip, Vflip, shear, translate, center, grayscale]
# transform_list = [None, brightness, saturation, contrast]