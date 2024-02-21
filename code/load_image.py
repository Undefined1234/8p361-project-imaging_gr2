import matplotlib.pyplot as plt
import math

def load_images(filenames, categories):
    """ This function displays images from the train dataset using matplotlib.""" 
    nr_figs = len(filenames)
    fig,ax = plt.subplots(nrows=math.ceil(nr_figs/2), ncols=2, figsize=(6,nr_figs*2))
    
    row = 0
    col = 0
    for name,cat in zip(filenames,categories):
        path = f"../data/train+val/train/{cat}/{name}.jpg"      
        img = plt.imread(f"{path}")
        ax[row,col].imshow(img)
        ax[row,col].set_title(f"Class {cat}")
        ax[row,col].xaxis.set_tick_params(labelbottom=False)
        ax[row,col].yaxis.set_tick_params(labelbottom=False)
        
        col += 1
        if col == 2:
            col = 0
            row +=1


filenames = ["0000d563d5cfafc4e68acb7c9829258a298d9b6a",
             "0000da768d06b879e5754c43e2298ce48726f722",
             "000d4bcc9d239e8304890ffd764794e93504e475",
             "000af35befdd9ab2e24fac80fb6508dfd1edd172"]
categories = ["0", "1", "0", "1"]

load_images(filenames, categories)


