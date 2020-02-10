import os.path
from PIL import Image

def scalling(path,height = 120,width = 120):
    images = os.listdir("C:\\Users\\PC\\Desktop\\MariGold\\")
    for image in images:
        imageFile = path+image
        im1 = Image.open(imageFile)
        im2  = im1.resize((height, width), Image.NEAREST)
        im2.save(path+'Scalled\\' + image)
        
scalling("C:\\Users\\PC\\Desktop\\MariGold\\",120,120)

simages = Image.open("C:\\Users\\PC\\Desktop\\MariGold\\Scalled\\img1.jpg")
