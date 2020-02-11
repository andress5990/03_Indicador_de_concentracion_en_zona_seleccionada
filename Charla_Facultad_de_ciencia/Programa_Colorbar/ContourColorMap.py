import cv2
import numpy as np
import sys
import math

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

import pdb


drawing = False # true if mouse is pressed
ix = -1 #Creamos un punto inicial x,y
iy = -1,
dotslist = [] #Creamos una lista donde almacenaremos los puntos del contorno

# mouse callback function
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
            cv2.line(img, (x,y), (x,y), (255,255,255), 2)#Dibujamos una linea de un solo pixel
            x = x
            y = y
            dot = [x,y]
            dotslist.append(dot)#Agregamos el punto a dotslist
            #print(dotslist) #Imprimimos el dotslist

    elif event == cv2.EVENT_LBUTTONUP:#Cremaos el evento si el boton se levanta
        drawing = False
        #cv2.circle(img,(x,y),1,(0,0,255),1)
        cv2.line(img, (x,y), (x,y), (255,255,255), 2)#Dibujamos la ultima lina en el ultimo punto
      
    return dotslist#Retornamos el dotlist


def Croped(dotslist, img):#hacemos un corte de la imagen en linea recta de tal forma que tenga las 
                          #dimenciones maximas del poligono que creamos
    rect = cv2.boundingRect(dotslist)#Encontramos los limites maximos del
    (x,y,w,h) = rect#Tomamos las dimenciones maximas del dotlist y las guardamos para dimencionar la mascara
    croped = img[y:y+h, x:x+w].copy()#cortamos una seccion rectangular de la imagen
    dotslist2 = dotslist- dotslist.min(axis=0)#reajustamos el dotslist con el minimo 

    mask = np.zeros(croped.shape[:2], dtype = np.uint8)# creamos una mascara de ceros para poder hacer el corte irregular
    cv2.drawContours(mask, [dotslist2], -1, (255,255, 255), -1, cv2.LINE_AA)#dibujamos el contorno
    dts = cv2.bitwise_and(croped,croped, mask=mask)#hacemos ceros todos los pixeles externos al contorno
    

    return [dts, mask, croped]

#def histogram(img, mask):
#    hist = cv2.calcHist([img], [0], mask, [256], [0,256])
#    return hist

#def Listing(y):
#    y1 = []#Creamos una lista vacia
#    for i in range(len(y)):#llenamos la lista vacia con los datos de y, esto porque y es de la forma y = [[],[],[]], y necesitamos y = []
#        y1.append(y[i][0])  
#    
#    return y1

def cal_luminosity(img):
    
    #cv_img = cv2.GaussianBlur(cv_img, (5,5), 0)
    #b, g, r = cv2.split(cv_img)#separamos los canales en gbr
    b = img[:,:,0]#Estas lineas hacen lo mismo que la anterior, pero mas eficiente
    g = img[:,:,1]
    r = img[:,:,2] 
    
    #Este for hace lo mismo que cv_img2 = cv2.imread(file_name, 0)
    holder = np.ones(img[:,:,0].shape)
    for i in range(b.shape[0]):#Filas
        for j in range(b.shape[1]):#Columnas
            lum = 0.2126*r[i][j] + 0.7152*g[i][j] + 0.0722*b[i][j] #Photometric/digital ITU BT.709
            #lum=0.299*r[i][j] +0.587*g[i][j]+0.114*b[i][j] #Digital ITU BT.601 
                                                            #(gives more weight to the R and B components)
            holder[i][j] = lum#Esto va generando la imagen escala de intesidad
    
    #Normalizamos la matriz
    maxholder = np.amax(holder)
    for i in range(b.shape[0]):
        for j in range(b.shape[1]): 
            holder[i][j] = holder[i][j]/maxholder
   
    minholder = np.amin(holder)
    maxholder = np.amax(holder)    

    return [holder, maxholder, minholder]


file1 = str(sys.argv[1])

img = cv2.imread(file1, 1)#Lee la imagen a color
#img2 = cv2.imread(file1,cv2.IMREAD_GRAYSCALE)#Lee la imagen pero en intensidad (B and W)
img2 = cv2.imread(file1,1)#abre como cv2.IMREAD_COLOR, leyendo la imagen a color
cv2.namedWindow(file1)#Cremaos la ventana para mistras a img
cv2.setMouseCallback(file1,draw_dots) #llamamos al MouseCall para dibujar el contorno
#pdb.set_trace()

while(1):
    cv2.imshow(file1,img) #Mostramos a img en la ventana para dibujar el contono

    k = cv2.waitKey(1) & 0xFF
    
    if k == 32: #space
        dotslist = np.asarray(dotslist)#Convertimos el contorno en un array de numpy
        
        #Aplicamos el contorno a la image a partir de dtolist
        img_croped_BB = Croped(dotslist, img2)[0]#Recuperamos solo la region de interes (imagen cortada con bordes negros)
        img_croped_BB = cv2.cvtColor(img_croped_BB, cv2.COLOR_BGR2RGB) 
        mask = Croped(dotslist, img2)[1] #Recuperamos la mascara creada
        img_croped = Croped(dotslist, img2)[2]#Esta linea recupera la seccion de la image SIN CORTAR para poder hacer que la que si se corta tenga iguales dimensiones
        img_analisys = cal_luminosity(img_croped_BB)#Esto da como resultado una lista que tiene como primera entrada, la imagen en escala de intensidades
        #img_croped_BB2 = cv2.cvtColor(img_croped_BB, cv2.COLOR_BGR2RGB) 
   

        #pdb.set_trace()

        name1 = file1[0:-4]#definimos un nombre sin la extrension del archvo

        #Imagen e histograma
        fig1 = plt.figure()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
        #imagen
        ax1 = fig1.add_subplot(111)
        ax1.set_xlabel('x pixel position', fontsize= 12)
        ax1.set_ylabel('y pixel position', fontsize= 12)
        ax1.set_title('Image: ' + name1)
        
        bounds = np.linspace(0, 1, 10, endpoint=True)#0 y 1 son los limites inferior y superior de la image, estan asi porque la imagen esta ya normalizada 
        plt_img = ax1.imshow(img_analisys[0], cmap = plt.cm.jet)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plt_img, cax = cax, ticks = bounds)
        #cv2.imshow('croped', img2)
        
        fig2 = plt.figure()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
        #imagen
        ax2 = fig2.add_subplot(111)
        ax2.set_xlabel('x pixel position', fontsize= 12)
        ax2.set_ylabel('y pixel position', fontsize= 12)
        ax2.set_title('Image: ' + name1)
        ax2.imshow(img_croped_BB, cmap= 'Greys')
        ax2.legend(loc='best')
        
        plt.show()
       


    if k == 27:#esc
        #pdb.set_trace()
        break# hace,el break para para el programa si se estripa esc

cv2.destroyAllWindows()#Destruimos todas las ventanas