import pandas as pd
import numpy as np
from collections import Counter
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import random



class ModelEvaluator:
    """
    This class calculates the % of the actual ingredient being in the top n recommended items 
    based on the test set. A list of 50 items absent in the given recipe are randomly drawn and 
    ranked together with the 1 actual ingredient.
    """
    
    def __init__(self, all_items, n):
        self.all = all_items
        self.n = n
    
    def get_missing_items(self, recipe, sample_size, seed = 208):
        """
        This returns a list of ingredients that are not in the given recipe.
        
        Input: a new recipe, i.e. list of ingredients; number of absent ingredients to draw; seed
        Output: a list of ingredients that do not exist in the given recipe
        """
        
        # Below obtains the absent items
        missing_items = set(self.all)-set(recipe)
        
        # Draw a random sample of size 'sample_size'
        random.seed(seed)
        rec_sample = random.sample(missing_items, sample_size)
        
        return set(rec_sample)

    def _if_in_topn_(self, actual_item, recommended_items, n):
        """
        Check to see if the item is among the top 'n' items in the ranked list of items
        
        Input: an item of interest (string); a list of ranked items; n in the 'top n' criterion
        Output: 1, if the item is among the first n items in the list; 0, otherwise
        """
        return int(actual_item in recommended_items[:n])
    
    def evaluate_model(self, model, X_te):
        """
        This computes % of the true ingredient (last item in the recipe) being among the first n ingredients
        ranked by the paticular model of interest in the test set.
        
        Input: a model as defined in RecSys.py; test set upon which the 'top n' is calculated, list of lists
        Output: % of times the model succcesfully ranking the actual ingredient in the first n ingredients
        """
        
        hit_rate = 0
        
        # For each recipe in the test set
        for i in range(0,len(X_te)): 
            # We first store 50 randomly drawn items that are not in the recipe
            candidates = self.get_missing_items(X_te[i], 50)
            
            # Rank the last ingredient in the recipe together with the 50 random ingredients 
            recommendation = model.recommend_items(X_te[i], candidates)
            hit_rate += self._if_in_topn_(X_te[i][-1], recommendation, self.n)
            
            # To keep track, we print the progress every 500 iterations
            if i % 500 == 0:
                print('Iteration:', i) 
        
        return hit_rate/len(X_te)



def corpus_to_matrix(corpus, all_ingredients = None):
    """
    This function takes in the recipe corpus and expands it into a matrix of 0's and 1's 
    to facilitate the calculation of cosine similiarity.
    
    Input: Corpus, i.e. a list of lists; list of all possible ingredients, if specified
    Output: A matrix of size len(corpus) by (# of unique ingredients in the corpus)
    """
    
    if all_ingredients == None:
        all_ingredients = list(itertools.chain.from_iterable(corpus))
    
    # Obtain the list of unique ingredients in the 
    unique_ingredients = np.unique(all_ingredients)
    
    # Generate a placeholder (i.e. matrix of 0's) for the expanded recipe matrix; largely sparse
    recipe = np.zeros([len(corpus),len(unique_ingredients)])
    
    # For each row in the recipe corpus 
    for i in range(0, len(corpus)):
        
        # For each entry in the row
        for j in range(0, len(unique_ingredients)):
            
            # We update the entry with 1, if the corresponding ingredient is in the recipe corpus
            if unique_ingredients[j] in corpus[i]:
                recipe[i][j] = 1
                
    return recipe       



class popularity:
    """
    This class takes in the training corpus and a recipe, and ranks the candidate 
    ingredients based on their appearance in the training corpus.
    We use the raw corpus for this class.
    """
    
    def __init__(self, all_items):
        self.all = all_items
        
        # Summarize all items into a dictionary of occurences
        self.popularity = Counter(all_items)

    def recommend_items(self, recipe, candidates):
        """
        This method ranks the last ingredient in the recipe joined with other candidate ingredients, based
        on the popuarity model; that is, we rank the ingredients by their appearance in the full list of 
        ingredients. 
        
        Input: a recipe, i.e. list of strings; candidate ingredients, also a list of strings
        Output: a ranked list of ingredients including the last item of the recipe and the candidate 
        ingredients, i.e. list of strings, from most popular to least popular ingredients
        """
        
        # Combine the last item of the given recipe with the candidate ingredients, and make a new list
        candidates = list(candidates)
        candidates.append(recipe[-1])
        
        count = []
        
        # For each item in the new list of ingredients
        for i in candidates:
            
            # We obtain their number of occurrences
            count.append(self.popularity[i])
        
        # Sort the ingredients from least popular to most popular
        rec = [candidates[i] for i in np.argsort(count)]        
        rec.reverse() # descending order
        
        return rec
    


class collaborative:
    """
    This class computes the cosine similarity (i.e. Pearson correlation) between the new recipe and the 
    training recipes. And then, it ranks the items in the candidate list along with the last item in 
    the new recipe, based on their occurrences in the 1000 training recipes that are the most similar 
    to the new recipe (excluding the last ingredient when computing cosine similarities).
    """
    
    def __init__(self, all_items, corpus):
        self.recipe = corpus_to_matrix(corpus) # Convert the corpus to expanded recipe matrix
        self.all = all_items
        self.unique = np.unique(all_items) # Store all the unique ingredients
        
    def recommend_items(self, recipe, candidates):
        """
        This method ranks the last ingredient in the given recipe combined with the candidate ingredients, 
        following the idea of collaborative filtering model. 
        
        Input: a recipe, i.e. list of strings; candidate ingredients, also a list of strings
        Output: a ranked list of ingredients including the last item of the recipe and the candidate 
        ingredients, i.e. list of strings, from most popular to least popular ingredients
        """
        
        # Combine the last ingredient in the given recipe with the candidate ingredients
        candidates = list(candidates)
        candidates.append(recipe[-1])
        
        # Expand the given recipe into a row vector, excluding the last item in the recipe
        new_recipe = corpus_to_matrix([recipe[-1]], self.all)
        
        # Compute the cosine similarity between the new recipe and all recipes in the expanded recipe matrix
        sim = cosine_similarity(self.recipe, new_recipe)
        order = np.argsort(sim, axis=0).tolist()  # obtain the order of the similarity scores
        order.reverse() # descending order
        nearest_1000 = []
        
        # Obtain the 1000 most 'similar' recipes based on the cosine similiarity scores
        for i in range(0,1000):
            nearest_1000.append((self.recipe[order[i]]).tolist())
        nearest_1000 = np.array(nearest_1000)
        
        # Compute the occurrences of each ingredient on the 1000 most 'similar' recipes
        popularity = np.sum(nearest_1000, axis=0)[0]
        
        count_1000 = []
        
        # For each item in the list of ingredients to be ranked
        for i in candidates:
            
            # We store their occurrences
            count_1000.append(popularity[np.where(self.unique == i)[0]])
        
        # Sort the ingredients from least popular to most popular
        rec = [candidates[int(i)] for i in np.argsort(count_1000, axis=0)]
        rec.reverse() # descending order
        
        return rec