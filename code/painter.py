
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Specified all color codes for all 20 types of cuisines
color_parlette = {"#000000", # blak
                "#FFFF00", #yellow
                "#1CE6FF", #cyan
                "#FF34FF", #pink 
                "#FF4A46", #red
                "#FFC300",  # green forest
                "#006FA6", # blue ocean
                "#A30059",# purple
                "#FFDBE5",  #light pink
                "#7A4900",  # gold or brown 
                "#0000A6", # blue electric 
                "#63FFAC", # green phospho
                "#B79762", #brown
                "#EEC3FF", #  
                "#8FB0FF", # light blue 
                "#997D87", #violet
                "#5A0007", 
                "#809693", 
                "#FEFFE6", #ligt yellow
                "#1B4400"}

def show_values_on_hbars(splot, ax, size=0.4):
    """
        plot numerical values on a seaborn vertical plot
    
    Arguments:
        splot : a seaborn plot
        size (float): the size of the text
    
    Return:
        None
    """
    # Loop through each patch in the plot

    for patch in splot.patches:
        # Get the dimension to locate the text
        pt_x = patch.get_x() + patch.get_width() + float(size)
        pt_y = patch.get_y() + patch.get_height()
        # Get the value of the bar
        value = int(patch.get_width())
        # Add the text
        ax.text(pt_x, pt_y, value, ha="left") 

def plot2PCA(pcaresult, title, data, size = (15,10)):
    """
        this wraps all the code for plotting 2 principal components on a 2D plane with the specified color code

    Arguments:
        pcaresult (array): the 2 principal components returned by PCA(n_components=2).fit_transform()
        title (str): the name of the plot
        data(dataframe): the original dataset for extracting cuisine types

    Return:
        None
    """
    # Plot the two principal components
    # Configure the plot details
    fig = plt.figure(figsize = size)
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title(title)

    # Get the task name
    cuisines = data.cuisine.unique()
    # Create a dataframe of the pca result for plot
    components = pd.DataFrame(pcaresult, columns = ['PC1', 'PC2'])
    # Plot data points corresponding to each task in the components
    for cuisine, color in zip(cuisines,color_parlette):
        idx = data['cuisine'] == cuisine
        ax.scatter(components.loc[idx,'PC1'],components.loc[idx,'PC2'],c = color,alpha=0.5,cmap='hsv', s = 40)
    ax.legend(cuisines)
    plt.show()

def plotCM(cm, cm_title, size = (22,18)):
    """
	   this function plots confusion matrix on a colored heatmap.
    
    Arguements:
		cm(ndarry): an arrray returned by the confusion_matrix() in sklearn package
		labelencoder(object): an object instantiated from the LabelEncoder() class
		size (tuple): the figure size
		cm_title (str): the title of the plot
    
    Return:
		None
	"""
    # Get the number of row of the matrix
    list_of_num = cm.shape[0]
    # Create a dataframe
    df_cm= pd.DataFrame(cm, index = le.inverse_transform(np.arrange(num))  ,columns = le.inverse_transform(np.arrange(num)))
    # Set the size for the figure
    plt.figure(figsize = size)
    # Create the heatmap
    ax = sns.heatmap(df, annot=True, fmt="d",cmap="YlGnBu")
    # Set the labels
    ax.set(ylabel='True Tasks', xlabel='clustering labels', title=cm_title)
    plt.show()


def plotBar(data, xlab, ylab, title, size = (20,25), scale = False, font_scale=3, show_font_on_har= False, col_palette = "BuGn_r"):
    """
        a functions that plot bar chart using seaborn package

    Arguements:
        data(dataframe): the dataframe we use to plot the bar chart
        xlab (str): the string to indicate which variable should be plotted along on the x-axis
        ylab (str): the string to indicate which variable should be plotted along on the y-axis
        title (str): a title of the plot
        size (tuple): a tuple to detemine the height and width of the plot
        scale (bool): determine whether font size should be changed
        font_scale (int): font size
        show_font_on_har (bool): determine whether font should be shown next to the horizontal bar
        col_palette (str): a color code
        
    Return:
        None
    """
    fig, ax = plt.subplots(figsize=size)
    if scale == True:
        sns.set(font_scale=font_scale) 
    s = sns.barplot(x=xlab, y=ylab, palette=col_palette, data=data)
    s.set_title(title)
    if show_font_on_har == True:
        show_values_on_hbars(s, ax,0.3)
