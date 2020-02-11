import cv2
import numpy as np
import sys
import math

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

import pdb

drawing = False # true if mouse is pressed
ix = -1 #Creamos un punto inicial x,y
iy = -1,
dotslist = [] #Creamos una lista donde almacenaremos los puntos del contorno

def draw_dots(event,x,y,flags,param): #Crea los puntos de contorno
    global ix,iy,drawing, dotslist#Hacemos globales la variabbles dentro de la funcion

    if event == cv2.EVENT_LBUTTONDOWN:#creamos la accion que se realizara si damos click
        drawing = True #Drawinf se vuelve True
        ix = x #Tomamos el punto donde se dio click
        iy = y
        dot = [x,y]
        dotslist.append(dot)#Lo agregamos al dotslist

    elif event == cv2.EVENT_MOUSEMOVE:#Creamos la accion si el mouse se mueve
        if drawing == True: #drawing se vuelve true
            #cv2.circle(img,(x,y),1,(0,0,255),2)
            cv2.line(img1, (x,y), (x,y), (0,0,0), 2)#Dibujamos una linea de un solo pixel
            x = x
            y = y
            dot = [x,y]
            dotslist.append(dot)#Agregamos el punto a dotslist
            #print(dotslist) #Imprimimos el dotslist

    elif event == cv2.EVENT_LBUTTONUP:#Cremaos el evento si el boton se levanta
        drawing = False
        #cv2.circle(img,(x,y),1,(0,0,255),1)
        cv2.line(img1, (x,y), (x,y), (0,0,0), 2)#Dibujamos la ultima lina en el ultimo punto
      
    return dotslist#Retornamos el dotlist

def Mask(dotslist, img):#hacemos un corte de la imagen en linea recta de tal forma que tenga las 
                          #dimenciones maximas del poligono que creamos
    rect = cv2.boundingRect(dotslist)#Encontramos los limites maximos del
    (x,y,w,h) = rect#Tomamos las dimenciones maximas del dotlist y las guardamos para dimencionar la mascara
    croped = img[y:y+h, x:x+w].copy()#cortamos una seccion rectangular de la imagen
    mask = np.zeros(img.shape[:2], dtype = np.uint8)# creamos una mascara de ceros para poder hacer el corte irregular
    #cv2.drawContours(mask, [dotslist], -1, (255,255, 255), -1, cv2.LINE_AA)#dibujamos el contorno
    #dark_img = cv2.bitwise_and(img,img, mask=mask)#hacemos ceros todos los pixeles externos al contorno
    
    return mask

def Dark(dotslist, img, mask):
    
    cv2.drawContours(mask, [dotslist], -1, (255,255, 255), -1, cv2.LINE_AA)#dibujamos el contorno
    dark_img = cv2.bitwise_and(img,img, mask=mask)#hacemos ceros todos los pixeles externos al contorno
     
    return dark_img


def Croped(img):
    img = img[40:407, 0:500]
    return img

def cal_luminosity(cv_img):
    
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) #volvemos el valor del np.array a RGB para procesar
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

def Reconstructor(img1, img2):
    img_r = np.zeros(shape=(img1.shape[0], img1.shape[1],3))
    
    for i in range(img1.shape[0]):#Filas
        for j in range(img1.shape[1]):#Columnas
            img_r[i][j][1] = 125.0*img1[i][j]
            img_r[i][j][2] = 125.0*img2[i][j]

#pdb.set_trace()
    return img_r

def MinMax(image):
    minholder = np.amin(image)
    maxholder = np.amax(image)
    
    return [minholder, maxholder]   

#Inicio del programa    


files = [str(sys.argv[1]), str(sys.argv[2])] 

name_final_file = 'Image_Reconstructed:_' + str(sys.argv[1]) + '_over_' + str(sys.argv[2])


img1 = cv2.imread(files[0])#Lee la imagen a color para corte
img2 = cv2.imread(files[1])#Lee la imagen a color para fondo

img1 = Croped(img1)#recortamos el encabezado
img2 = Croped(img2)#recortamos el encabezado

cv2.namedWindow(files[0])#Cremaos la ventana para mistras a img
cv2.setMouseCallback(files[0],draw_dots) #llamamos al MouseCall para dibujar el contorno

while(1):
    #img1 = cal_luminosity(img1)[0]
    cv2.imshow(files[0], img1) #Mostramos a img en la ventana para dibujar el contono

  
    k = cv2.waitKey(1) & 0xFF
    
    if k == 32: #space
        img11 = img1.copy()#hacemos una copia de img1
        dotslist = np.asarray(dotslist)#convertimos img1 en un np.array()
        
        #pasamos las imagenes a intensidad
        img11 = cal_luminosity(img11)#cambiasmos a intensidad
        img2 = cal_luminosity(img2)#cambiasmos a intensidad
         
        mask = Mask(dotslist, img1)#creamos la mascara
        
        Dark_img = Dark(dotslist, img11, mask)#aplicamos la mascara a img11
        #Dark_img = cv2.addWeighted(Dark_img, 1y,)
        img_r = Reconstructor(Dark_img, img2)
        
        MinMaxImg11 = MinMax(img11)
        MinMaxImg2 = MinMax(img2)
        MinMaxImg_R = MinMax(img_r)
        
            
        
        cv2.imshow(name_final_file , img_r)#mostramos la imagen con borde negro
        #img_r.convertTo(img_r, CV_8UC3, 255.0)
        cv2.imwrite(name_final_file  +'.jpg' , img_r)
            
    if k == 27:#esc
        #pdb.set_trace()
        break# hace,el break para para el programa si se estripa esc
    



cv2.destroyAllWindows()#Destruimos todas las ventanas