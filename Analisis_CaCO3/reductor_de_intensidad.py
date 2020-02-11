import cv2
import numpy as np
import sys

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

    #pdb.set_trace()
    return holder

def write_image(img, value):#Aumenta la intensidada de la imagen final nuevamente
    img = cv2.convertScaleAbs(img, alpha=(value))
    return img


file1 = str(sys.argv[1])
img1 = cv2.imread(file1, cv2.IMREAD_GRAYSCALE)

while(1):
    cv2.imshow(file1,img1) #Mostramos a img1 en la ventana para dibujar el contono
    #img100 = cal_luminosity(img1)
    img100 = write_image(img1, 255.0)
    img50 = write_image(img1, 127.5)
    img33 = write_image(img1, 85.0)
    img0 = write_image(img1, 0.0)
    cv2.imshow('grayscale100',img100)
    cv2.imshow('grayscale50',img50)
    cv2.imshow('grayscale33',img33)
    cv2.imshow('grayscale0',img0)
    


    k = cv2.waitKey(1) & 0xFF

    
    if k == 32: #space
        cv2.imwrite('100.jpg', img100)
        cv2.imwrite('50.jpg', img50) 
        cv2.imwrite('33.jpg', img33) 
        cv2.imwrite('0.jpg', img0)




    if k == 27:#esc
        #pdb.set_trace()
        break# hace,el break para para el programa si se estripa esc

cv2.destroyAllWindows()#Destruimos todas las ventanas