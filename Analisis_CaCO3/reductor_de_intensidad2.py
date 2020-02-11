import cv2
import numpy as np
import sys
import pdb


d = [800,800]

img100 = np.ones(d)
img50 = np.zeros(d)
img33 = np.zeros(d)
img10 = np.zeros(d)
img0 = np.zeros(d)

for i in img100:
    i *=255.0

for i in img50:
    i *= 127.5
    
for i in img33:
    i *= 85

for i in img10:
    i *= 25.5
    


while(1):
    cv2.imshow('100',img100) #Mostramos a img1 en la ventana para dibujar el contono
    cv2.imshow('50',img50)
    cv2.imshow('33',img33)
    cv2.imshow('10',img10)
    cv2.imshow('0',img0)
    #pdb.set_trace()

    k = cv2.waitKey(1) & 0xFF
     
    if k == 32: #space
        cv2.imwrite('100.jpg', img100)
        cv2.imwrite('50.jpg', img50) 
        cv2.imwrite('33.jpg', img33) 
        cv2.imwrite('10.jpg', img10) 
        cv2.imwrite('0.jpg', img0)

    if k == 27:#esc
        #pdb.set_trace()
        break# hace,el break para para el programa si se estripa esc

cv2.destroyAllWindows()#Destruimos todas las ventanas