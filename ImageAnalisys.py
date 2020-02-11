import cv2
import numpy as np
import sys
import math

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

from PIL import Image

import pdb

def cal_luminosity(file_name):
    
    cv_img = cv2.imread(file_name)#np.array con shape(filas, columnas, canales)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) #volvemos el valor del np.array a RGB para procesar
    cv_img = cv_img[40:407, 0:500]
    #cv_img = cv2.GaussianBlur(cv_img, (5,5), 0)
    #b, g, r = cv2.split(cv_img)#separamos los canales en gbr
    b = cv_img[:,:,0]#Estas lineas hacen lo mismo que la anterior, pero mas eficiente
    g = cv_img[:,:,1]
    r = cv_img[:,:,2] 
    
    #Este for hace lo mismo que cv_img2 = cv2.imread(file_name, 0)
    holder = np.ones(cv_img[:,:,0].shape)
    for i in range(b.shape[0]):#Filas
        for j in range(b.shape[1]):#Columnas
            lum=0.2126*r[i][j] + 0.7152*g[i][j] + 0.0722*b[i][j] #Photometric/digital ITU BT.709
            #lum=0.299*r[i][j] +0.587*g[i][j]+0.114*b[i][j] #Digital ITU BT.601 
                                                            #(gives more weight to the R and B components)
            holder[i][j] = lum
    
    #Normalizamos la matriz
    maxholder = np.amax(holder)
    for i in range(b.shape[0]):
        for j in range(b.shape[1]): 
            holder[i][j] = holder[i][j]/maxholder
   
    minholder = np.amin(holder)
    maxholder = np.amax(holder)    

    return(holder, maxholder, minholder)

image_name = sys.argv[1]

#creamos ambas matrices a partir de del calculo de luminosidad (grayscale)
cv_img1 = cal_luminosity(image_name)[0]
cv_img1max = cal_luminosity(image_name)[1]#Son de las imagenes no normalizadas
cv_img1min = cal_luminosity(image_name)[2]

bounds = np.linspace(cv_img1min, cv_img1max, 10, endpoint=True)

ax1 = plt.subplot(111)
pltimg = ax1.imshow(cv_img1, cmap = plt.cm.jet)
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(pltimg, cax = cax, ticks = bounds)

plt.show()
