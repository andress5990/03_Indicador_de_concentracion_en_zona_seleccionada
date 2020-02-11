import cv2
import numpy as np
import sys
import math

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar


import pdb
            
file1 = str(sys.argv[1]) 

img1 = cv2.imread(file1, 1)#Lee la imagen a color
img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        
    

while(1):
    
    cv2.imshow(file1,img1) #Mostramos a img en la ventana para dibujar el contono

    k = cv2.waitKey(1) & 0xFF
    
    if k == 27:#esc
        #pdb.set_trace()
        break# hace,el break para para el programa si se estripa esc

cv2.destroyAllWindows()#Destruimos todas las ventanas