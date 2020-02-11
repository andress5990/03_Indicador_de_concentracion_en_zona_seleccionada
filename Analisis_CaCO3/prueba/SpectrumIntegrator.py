import cv2
import numpy as np
import sys
from scipy.optimize import curve_fit as fit
from scipy import exp 
from matplotlib import pyplot as plt

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

def histogram(img, mask):
    hist = cv2.calcHist([img], [0], mask, [256], [0,256])
    return hist

def Listing(y):
    y1 = []#Creamos una lista vacia
    for i in range(len(y)):#llenamos la lista vacia con los datos de y, esto porque y es de la forma y = [[],[],[]], y necesitamos y = []
        y1.append(y[i][0])  
    
    return y1


def Countsum(hist):
    #suma = 0
    #for l in hist:
    #    if l > 0:
    #        suma += l
    suma = sum(hist)
    return suma

def MeanAndSigma(x, y):
    
    mean = sum(x*y)/sum(y) #Calculamos el promedio pesado con los y
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))#calculamos la desviacion estandar
    #pdb.set_trace()

    return [mean, sigma]

def gaussModel(x, a, x0, sigma):
    
    f1 = -((x-x0)**2)/(2*(sigma**2))
    y = a*exp(f1)
    
    return y

def gauss(model, x, y, p0):
    popt, pcov = fit(model, x, y, p0)#popt son los parametros del fit, pcov es la curva del fit
    maxg = popt[0]
    meang = popt[1]
    sigmag = popt[2]

    return [maxg, meang, sigmag]

def maximum_value(hist, gauss):
    hist = np.asarray(hist)
    gauss = np.asarray(gauss)

    max_val = max(hist)
    max_ubication = hist.argmax()
    gmax_val = max(gauss)
    gmax_ubication = gauss.argmax()
                     

    return [max_val, max_ubication, gmax_val, gmax_ubication] 


files = [str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3])]

img = cv2.imread(files[0])#Leemos intensidad de la primera imagen r
img1 = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(files[1], cv2.IMREAD_GRAYSCALE)#Leemos intensidad de la segunda imagen g
img3 = cv2.imread(files[2], cv2.IMREAD_GRAYSCALE)#Leemos intensidad de la segunda imagen b



cv2.namedWindow(files[0])#Creamos la ventana para mostras a img1 como indicador para el contorno
cv2.setMouseCallback(files[0],draw_dots) #llamamos al MouseCall para dibujar el contorno en img1


while(1):
    cv2.imshow(files[0],img) #Mostramos a img1 en la ventana para dibujar el contono

    k = cv2.waitKey(1) & 0xFF
    
    if k == 32: #space
        dotslist = np.asarray(dotslist)#Convertimos el contorno en un array de numpy
        
        #Aplicamos el contorno a la image a partir de dtolist
        img1_croped_BB = Croped(dotslist, img1)[0]#Recuperamos solo la region de interes (imagen cortada con bordes negros)
        img2_croped_BB = Croped(dotslist, img2)[0]#Recuperamos solo la region de interes (imagen cortada con bordes negros)
        img3_croped_BB = Croped(dotslist, img3)[0]#Recuperamos solo la region de interes (imagen cortada con bordes negros)
        
        
        mask1 = Croped(dotslist, img1)[1] #Recuperamos la mascara creada en img1
        mask2 = Croped(dotslist, img2)[1]
        mask3 = Croped(dotslist, img3)[1]
        
        img1_croped = Croped(dotslist, img1)[2]#recuperamos la imagen cortada en rectangulo para analizar con la mascara     
        img2_croped = Croped(dotslist, img2)[2]
        img3_croped = Croped(dotslist, img3)[2]
        
        hist1 = histogram(img1_croped, mask1)#Calculamos el histograma usando la mascara #len(hist) = 256
        hist1 = Listing(hist1)
        hist1.pop(0)
        
        hist2 = histogram(img2_croped, mask2)#Calculamos el histograma usando la mascara #len(hist) = 256
        hist2 = Listing(hist2)
        hist2.pop(0)
        
        hist3 = histogram(img3_croped, mask3)#Calculamos el histograma usando la mascara #len(hist) = 256
        hist3 = Listing(hist3)
        hist3.pop(0)
        
        x1 = np.linspace(1.0, len(hist1), len(hist1))
        x2 = np.linspace(1.0, len(hist2), len(hist2))
        x3 = np.linspace(1.0, len(hist3), len(hist3))   
        
        [mean1, sigma1] = MeanAndSigma(x1, hist1)
        [maxg1, meang1, sigmag1] = gauss(gaussModel, x1, hist1, p0=[max(hist1), mean1, sigma1])#popt son los parametros del fit, pcov es la curva del fit
        fit1 = gaussModel(x1, maxg1, meang1, sigma1)
        
        [mean2, sigma2] = MeanAndSigma(x2, hist2)
        [maxg2, meang2, sigmag2] = gauss(gaussModel, x2, hist2, p0=[max(hist2), mean2, sigma2])#popt son los parametros del fit, pcov es la curva del fit
        fit2 = gaussModel(x2, maxg2, meang2, sigma2)
        
        [mean3, sigma3] = MeanAndSigma(x3, hist3)
        [maxg3, meang3, sigmag3] = gauss(gaussModel, x3, hist3, p0=[max(hist3), mean3, sigma3])#popt son los parametros del fit, pcov es la curva del fit
        fit3 = gaussModel(x3, maxg3, meang3, sigma3)
        #pdb.set_trace()

      
        #pdb.set_trace()
        Tcount1 = Countsum(hist1)
        print("Las cuentas totales distintas de cero para " + files[0] + " es " + str(Tcount1))
        Tcount2 = Countsum(hist2)
        print("Las cuentas totales distintas de cero para " + files[1] + " es " + str(Tcount2))
        Tcount3 = Countsum(hist3)
        print("Las cuentas totales distintas de cero para " + files[2] + " es " + str(Tcount3))
        
        name1 = files[0].replace('.tif', '')#b
        name2 = files[1].replace('.tif', '')#b
        name3 = files[2].replace('.tif', '')#b
        
        [hist_max_val1, hist_max_ubication1, gmax_val1, gmax_ubication1]= maximum_value(hist1, fit1)
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.set_xlabel('Intensity Values', fontsize= 12)
        ax1.set_ylabel('Counts', fontsize= 12)
        ax1.set_title('Histogram of: ' + name1)
        ax1.set_xticks(np.arange(0, len(x1), step=20))
        ax1.plot(x1, hist1, '-', color='red', markersize=6, label = 'histogram', linewidth=2)#ploteamos el histograma
        ax1.plot(x2, fit1, '-', color='teal', markersize=1, label = 'gaussian fit', linewidth=1)#ploteamos el histograma
        ax1.axvline(x = gmax_ubication1)#DIbujamos una linea vertical
        ax1.text(gmax_ubication1 + 25, (hist_max_val1)/2, 'max value: ' + str(gmax_ubication1))
        ax1.legend(loc='best')
        plt.savefig('Histogram of ' + name1)  
        
        [hist_max_val2, hist_max_ubication2, gmax_val2, gmax_ubication2]= maximum_value(hist2, fit2)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.set_xlabel('Intensity Values', fontsize= 12)
        ax2.set_ylabel('Counts', fontsize= 12)
        ax2.set_title('Histogram of: ' + name2)
        ax2.set_xticks(np.arange(0, len(x2), step=20))
        ax2.plot(x2, hist2, '-', color='red', markersize=6, label = 'histogram', linewidth=2)#ploteamos el histograma
        ax2.plot(x2, fit2, '-', color='teal', markersize=1, label = 'gaussian fit', linewidth=1)#ploteamos el histograma
        ax2.axvline(x = gmax_ubication2)#DIbujamos una linea vertical
        ax2.text(gmax_ubication2 + 25, (hist_max_val2)/2, 'max value: ' + str(gmax_ubication2))
        ax2.legend(loc='best')
        plt.savefig('Histogram of ' + name2)  


        [hist_max_val3, hist_max_ubication3, gmax_val3, gmax_ubication3]= maximum_value(hist3, fit3)
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.set_xlabel('Intensity Values', fontsize= 12)
        ax3.set_ylabel('Counts', fontsize= 12)
        ax3.set_title('Histogram of: ' + name3)
        ax3.set_xticks(np.arange(0, len(x3), step=20))
        ax3.plot(x3, hist3, '-', color='red', markersize=6, label = 'histogram', linewidth=2)#ploteamos el histograma
        ax3.plot(x3, fit3, '-', color='teal', markersize=1, label = 'gaussian fit', linewidth=1)#ploteamos el histograma
        ax3.axvline(x = gmax_ubication3)#DIbujamos una linea vertical
        ax3.text(gmax_ubication3 + 25, (hist_max_val3)/2, 'max value: ' + str(gmax_ubication3))
        ax3.legend(loc='best')
        plt.savefig('Histogram of ' + name3)

        plt.show()
    
        cv2.imshow('croped1' + files[0], img1_croped_BB)#Mostramos img2 con el contorno 
        cv2.imshow('croped2' + files[1], img2_croped_BB)#Mostramos img2 con el contorno 
        cv2.imshow('croped3' + files[2], img3_croped_BB)#Mostramos img2 con el contorno 
        

        cv2.imwrite("Corte_" + name1 +'.jpg', img1_croped_BB) 
        cv2.imwrite("Corte_" + name2 +'.jpg', img2_croped_BB)
        cv2.imwrite("Corte_" + name3 +'.jpg', img3_croped_BB)  


    if k == 27:#esc
        #pdb.set_trace()
        break# hace,el break para para el programa si se estripa esc

cv2.destroyAllWindows()#Destruimos todas las ventanas