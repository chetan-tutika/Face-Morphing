from PIL import Image
import matplotlib.pyplot as plt
import scipy
import numpy as np 
import matplotlib.pyplot as plt




img = Image.open('damon.jpg')


# mywidth = 100


# wpercent = (mywidth/float(img.size[0]))
# hsize = int((float(img.size[1])*float(wpercent)))
# img = img.resize((mywidth,hsize), Image.ANTIALIAS)
# img.save('sourceClooney1.jpeg')

myh = 300


hpercent = (myh/float(img.size[1]))
wsize = int((float(img.size[0])*float(hpercent)))
img = img.resize((wsize,myh), Image.ANTIALIAS)
img.save('Damon.jpg')