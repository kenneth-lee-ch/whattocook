from nltk.stem import WordNetLemmatizer
from nltk import WordPunctTokenizer
import re
import unicodedata
import numpy as np

# Instantiate an tokenizer object and an lemmatizer object
wpt = WordPunctTokenizer()
lemmatizer = WordNetLemmatizer()

def clean(text):
    """
        this function applies different function to manipulate a strnig 

    Arguemnts:
        text(str): a string
    Return:
        text(str): a "cleaned" string
  """
    # Converts them into lower case
    text = text.lower()
    # grouping together the different inflected forms of a word
    text = lemmatizer.lemmatize(text,pos='n')
    # de-accent the words
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # Remove everything else besides letter 
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    return text 

def text_preprocess(data):
    """
        preprocess the ingredients names and preserves the structure of the original data

    Arguemnts:
        data (dataframe): a dataframe that contains all the ingredients in the data
    Return:
        cleaned_corpus(list): a list of lists of ingredients that has been preprocessed.
    """
    cleaned_corpus = list()
    # Loop through every list of ingredients
    for ls in data["ingredients"]:
        clean_ls = list()
    # Loop through every ingredient in the list
        for ingred_names in ls:
            dummy_ls = []
            # remove this description from the ingredients
            ingred_names =  re.sub(r'\(.*oz.\)','', ingred_names)
            # If there are more than one word in that string, we split them 
            if " " in ingred_names:
                list_of_names = ingred_names.split(" ")
                for ingredient in  list_of_names:
                    cleaned_txt = clean(ingredient)
                    dummy_ls.append(cleaned_txt)
                # Join the string together and append that to a list
                clean_ls.append(" ".join(dummy_ls))
        # If not, we can clean it directly
            else:
                cleaned_txt = clean(ingred_names)
                # Append the cleaned name to a list
                clean_ls.append(cleaned_txt)
        cleaned_corpus.append(clean_ls)
    return cleaned_corpus    


def convertWord2VecForTrainTestSplit(word2vec_model, corpus):
    """
        this function converts the word embedding trained by word2vec to a 39774 by D matrix for training classifiers. 
        D is the number of dimensions specified in the parameters for training word2vec model. 
    
    Arguments:
        word2vec_model: a word embedding trained by word2vec model 
        doc: a list of lists, which contain ingredient names in the data.
    Return:
        a 39774 by D np array 
    """
    weight_matrix = [] # Create an empty list
    for ingredients in corpus:
        # Sum the vectors and divded them by the number of ingredients
        ingredient_vectors = np.sum(word2vec_model.wv[ingredients], axis = 0)/len(ingredients)
        # Add the weights to a list
        weight_matrix.append(ingredient_vectors)
    # Convert the list to a nparray
    return np.asarray(weight_matrix)