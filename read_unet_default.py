from unet import Unet

model = Unet()

print(model._defaults['model_path'].split('/')[1][:-4])