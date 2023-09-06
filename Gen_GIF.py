import imageio 
import os

#------------ GIF PARA LA WAE-MMD ------------#
genpath = "C:/Users/manum/Desktop/TFG/WAE-MMD/data/reconst_images/"

filenames = os.listdir(genpath + "Reconstruccion")

images = []

for filename in filenames:
    images.append(imageio.imread(genpath + "Reconstruccion/" + filename))
    
imageio.mimsave(genpath + "Reconstruccion/gif_reconstruccion.gif", images, duration = 0.5)


filenames = os.listdir(genpath + "Muestreo")

images = []

for filename in filenames:
    images.append(imageio.imread(genpath + "Muestreo/" + filename))
    
imageio.mimsave(genpath + "Muestreo/gif_muestreo.gif", images, duration = 0.5)

#------------ GIF PARA LA WAE-GAN ------------#
genpath = "C:/Users/manum/Desktop/TFG/WAE-GAN/data/reconst_images/"

filenames = os.listdir(genpath + "Reconstruccion")

images = []

for filename in filenames:
    images.append(imageio.imread(genpath + "Reconstruccion/" + filename))
    
imageio.mimsave(genpath + "Reconstruccion/gif_reconstruccion.gif", images, duration = 0.5)