import cv2
import numpy as np
import sys
import math

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar


import pdb


#El orden de la imagen resultante siempre es rgb
#Pero el orden al llamar el programa es rgb

def Croped(img):
    img = img[40:407, 0:500]
    return img

def cal_luminosity(cv_img):
    
    #cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) #volvemos el valor del np.array a RGB para procesar
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
    #pdb.set_trace()
    return holder

def Reconstructor(img1, img2, img3):
    img_r = np.zeros(shape=(img1.shape[0], img1.shape[1],3))
    #(BGR)
    for i in range(img1.shape[0]):#Filas
        for j in range(img1.shape[1]):#Columnas
            img_r[i][j][0] = img1[i][j] #B
            img_r[i][j][1] = img2[i][j] #G
            img_r[i][j][2] = img3[i][j] #R
            
    return img_r
            
            
def write_image(img):#Aumenta la intensidada de la imagen final nuevamente
    img = cv2.convertScaleAbs(img, alpha=(255.0))
    return img
            
files = [str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3])] 
        #[b,g,r]
        
        
name1 = files[0].replace('.tif', '')#b
name2 = files[1].replace('.tif', '')#g
name3 = files[2].replace('.tif', '')#r


name_final_file = 'Image_Reconstructed (r,g,b):_' + str(sys.argv[1]) + '_over_' + str(sys.argv[2]) + '_over_' + str(sys.argv[3])
final_name_file = name_final_file.replace('.tif', '') 
                                                                              
img11 = cv2.imread(files[0])#Lee la imagen a color para corte, este sera el azul
img11 = cv2.cvtColor(img11, cv2.COLOR_RGB2BGR)
img22 = cv2.imread(files[1])#Lee la imagen a color para fondo, este sera el verde
img22 = cv2.cvtColor(img22, cv2.COLOR_RGB2BGR)
img33 = cv2.imread(files[2])#Lee la imagen a color para fondo, este sera el rojo
img33 = cv2.cvtColor(img33, cv2.COLOR_RGB2BGR)


img11 = Croped(img11)#recortamos el encabezado
img22 = Croped(img22)#recortamos el encabezado
img33 = Croped(img33)#recortamos el encabezado

    
#pasamos las imagenes a intensidad
img1 = cal_luminosity(img11)#cambiamos a intensidad - Blue
img2 = cal_luminosity(img22)#cambiamos a intensidad - Green
img3 = cal_luminosity(img33)#cambiamos a intensidad - Red
               
img_result = Reconstructor(img3, img2, img1)#Reconstruimos la imagen bgr

#pdb.set_trace()
           
#cv2.imshow(name_final_file , img_r)#mostramos la imagen con borde negro
    #img_r.convertTo(img_r, CV_8UC3, 255.0)
cv2img = write_image(img_result)
cv2.imwrite(final_name_file  +'.jpg' , cv2img)

            
ax1 = plt.subplot(111)
divider = make_axes_locatable(ax1)
cax1 = divider.new_vertical(size="5%", pad=-0.25, pack_start=True)

pltimgg = ax1.imshow(img_result[:,:,0], cmap = 'Blues',  alpha=1)
colorbg = plt.colorbar(pltimgg, shrink=0.75)
colorbg.set_label('Carbon', size= 15)


pltimgr = ax1.imshow(img_result[:,:,1], cmap = 'Greens', alpha=0.2)
colorbr = plt.colorbar(pltimgr, shrink=0.75)
colorbr.set_label('Oxygen', size= 15)

pltimgb = ax1.imshow(img_result[:,:,2], cmap = 'Reds', alpha=0.5)
colorbb = plt.colorbar(pltimgb, shrink=0.75)
colorbb.set_label('Calcium', size= 15)

#ax1.set_title('Image: ' + final_name_file)
ax1.set_xlabel('pixel x position', size=15)
ax1.set_ylabel('pixel y position', size=15)



plt.show()#se muestra (b,g,r) siendo azul la base, verde en medio, y rojo superior
