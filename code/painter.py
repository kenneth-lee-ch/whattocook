
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# Define global variables
# Specified all color codes for all 20 types of cuisines
color_parlette = {u'irish':"#000000", # blak
                u'mexican':"#FFFF00", #yellow
                u'chinese':"#1CE6FF", #cyan
                u'filipino': "#FF34FF", #pink 
                u'vietnamese':"#FF4A46", #red
                u'spanish':"#FFC300",  # green forest
                u'japanese':"#006FA6", # blue ocean
                u'moroccan':"#A30059",# purple
                u'french':"#FFDBE5",  #light pink
                u'greek': "#7A4900",  # gold or brown 
                u'indian':"#0000A6", # blue electric 
                u'jamaican':"#63FFAC", # green phospho
                u'british': "#B79762", #brown
                u'brazilian': "#EEC3FF", #  
                u'russian':"#8FB0FF", # light blue 
                u'cajun_creole':"#997D87", #violet
                u'thai':"#5A0007", 
                u'southern_us':"#809693", 
                u'korean':"#FEFFE6", #ligt yellow
                u'italian':"#1B4400"}

# Create a list for fixing the legend with the corresponding color all the time
lgend = list()
for l, c in color_parlette.items():
    lgend.append(mpatches.Patch(color=c, label=l))

# Extract all colors values from the dictionary
color_vector = color_parlette.values()

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

def plot2Components(result, title, xlab, ylab, data, size = (15,10)):
    """
        this wraps all the code for plotting 2 principal components on a 2D plane with the specified color code

    Arguments:
        result (array): the 2 components returned either by PCA.fit_transform() or TSNE.fit_transform()
        title (str): the name of the plot
        xlab (str): the label for the x-axis
        ylab (str): the label for the y-axis
        data(dataframe): the original dataset for extracting cuisine types

    Return:
        None
    """
    # Plot the two principal components
    # Configure the plot details
    fig = plt.figure(figsize = size)
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
   # Get the task name
    cuisines = data.cuisine.unique()
    # Create a dataframe of the pca result for plot
    components = pd.DataFrame(result, columns = ['C1', 'C2'])
    # Plot data points corresponding to each task in the components
    for cuisine, color in zip(cuisines,color_vector):
        idx = data['cuisine'] == cuisine
        ax.scatter(components.loc[idx,'C1'],components.loc[idx,'C2'],c = color,alpha=0.5,cmap='hsv', s = 40)
    plt.legend(handles=lgend)
    plt.show()

def plotCM(cm, le, cm_title, size = (22,18)):
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
    dim = cm.shape[0]
    # Create a dataframe
    df_cm= pd.DataFrame(cm, index = le.inverse_transform(np.arange(dim))  ,columns = le.inverse_transform(np.arange(dim)))
    # Set the size for the figure
    plt.figure(figsize = size)
    # Create the heatmap
    ax = sns.heatmap(df_cm, annot=True, fmt="d",cmap="YlGnBu")
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

def plotF1(y_score, Y_test):
    """
        plotting the f1 micro curve given the accuracy and test dataset in multiclass classification

    Arguements:
        y_score: confidence scores for samples. The confidence score for a sample is the signed distance of that sample to the hyperplane.
        Y_test: test sample 
    
    Return:
        None
    """
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    # Get number of classes
    n_classes = y_score.shape[1]
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
        y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score,average="micro")

    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))

    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))
