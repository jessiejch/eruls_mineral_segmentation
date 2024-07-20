import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from unet import Unet

from unet_vote import Unet_vote

base_path = "./img/special/"
image_path  = os.path.join(base_path, "image0362_007.jpg")
image       = Image.open(image_path)
# print(type(image))
unet_vote       = Unet_vote()
# image = unet_vote.get_result_nocolor(image)
image, model_id = unet_vote.get_result_colors(image)


image.show()
print(model_id)
