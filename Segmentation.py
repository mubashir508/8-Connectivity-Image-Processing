import cv2 as cv
import numpy as np
import random;

class Segmentation():
        
    def __init__(self,Image):

        
        self.Image = np.array(Image)
        self.totalObjects = 0

        self.labels = []

    def show(self,prompt, img):
        cv.namedWindow(prompt, cv.WINDOW_NORMAL)
        cv.imshow(prompt, img)
        cv.waitKey(0)
        cv.destroyWindow(prompt)

    def showChannel(image, idx, prompt,self):
        temp = np.array(image)
        temp[:, :, (idx+1) % 3] = 0
        temp[:, :, (idx+2) % 3] = 0

        self.show(prompt=prompt, img=temp)

    def Segment(self):
        # converting image into binary
        for i in range(self.Image.shape[0]):
            for j in range(self.Image.shape[1]):
                # checking the if the pixel is dark or not
                if(self.Image[i][j]!=255):
                    # if pixel is dark
                    #checking its left pixel 
                    if(j>0 and self.Image[i][j-1]!=255):
                        # assigning the pixel label of previous its left neighbour
                        self.Image[i][j] = self.Image[i][j-1];
                    elif(i>0 and self.Image[i-1][j]!=255):
                        self.Image[i][j] = self.Image[i-1][j];
                    
                    #diagonal top left
                    elif(i>0 and j>0 and self.Image[i-1][j-1]!=255):
                        self.Image[i][j] = self.Image[i-1][j-1];
                    #diagonal top right
                    elif(i>0 and j!=self.Image.shape[1] and self.Image[i-1][j+1]!=255):
                        self.Image[i][j] = self.Image[i-1][j+1];                   
                    else:
                        newLabel = 1;
                        if len((self.labels))>0:
                            newLabel = self.labels[-1]+1
                            
                        self.Image[i][j] = newLabel;
                        self.labels.append(newLabel);


        self.CheckEqualvilance();

        return self.Image
    
    def CheckEqualvilance(self):
         
         for i in range(self.Image.shape[0]):
            for j in range(self.Image.shape[1]):
                # checking the if the pixel has been assigned a pixel or not
                if(self.Image[i][j]!=255 ):
                    pixel = self.Image[i][j];
                    equalArray  = [];

                    # if pixel is has been assigned some label
                    #checking its left pixel 
                    if(j>0 and self.Image[i][j-1]!=255):
                        # assigning the pixel label of previous its left neighbour
                        if(self.Image[i][j-1]!=pixel):
                            equalArray.append(self.Image[i][j-1])
                            self.Image = np.where(self.Image==self.Image[i][j-1],pixel,self.Image)
                    if(i>0 and self.Image[i-1][j]!=255):
                        if(self.Image[i-1][j]!=pixel):
                            equalArray.append(self.Image[i-1][j])
                            self.Image = np.where(self.Image==self.Image[i-1][j],pixel,self.Image)

                    #diagonal top left

                    if(i>0 and j>0 and self.Image[i-1][j-1]!=255):
                        if(self.Image[i-1][j-1]!=pixel):
                            equalArray.append(self.Image[i-1][j-1])
                            self.Image = np.where(self.Image==self.Image[i-1][j-1],pixel,self.Image)

                    #diagonal top right
                    if(i>0 and j!=self.Image.shape[1] and self.Image[i-1][j+1]!=255):
                        if(self.Image[i-1][j+1]!=pixel):
                            equalArray.append(self.Image[i-1][j+1])
                            self.Image = np.where(self.Image==self.Image[i-1][j+1],pixel,self.Image)

                

            
    
originalImage = cv.imread("./image.bmp");

grayImg = cv.cvtColor(originalImage,cv.COLOR_BGR2GRAY)


coinBin = np.where(grayImg>100,0,255).astype(np.uint8);

labels = []
fancy = {};


def unique(pixel):
    if(pixel !=255):
        labels.append(pixel)

findLabels = np.vectorize(unique)




Segmentor = Segmentation(coinBin);
segmentedImage = Segmentor.Segment();
findLabels(segmentedImage)

labels = set(labels)

for label in labels:
    fancy[label] = random.randint(0,255),random.randint(0,255),random.randint(0,255);
    random.seed(label)

#converting the image to original rgb channels and assigning colorful labels
for i in range(originalImage.shape[0]):
    for j in range(originalImage.shape[1]):
        if(segmentedImage[i][j] !=255):
            originalImage[i][j] = fancy[segmentedImage[i][j]]


cv.imshow(f"Segmented with total objects {len(labels)} ",originalImage)
cv.waitKey(0)

            




        



