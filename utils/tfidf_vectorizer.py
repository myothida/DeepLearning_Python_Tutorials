import pandas as pd
import math
from collections import Counter

class Tfidf_Vectorizer :
    def __init__(self):
        self.idf_dict = {}
        self.vocabulary = {}

    # Calculate Term Frequency (TF) for a document
    def _compute_tf(self, doc):
        tf_dict = {}
        total_terms = len(doc)
        term_count = Counter(doc)  # Count the occurrences of each term in the document
        
        for term, count in term_count.items():
            tf_dict[term] = count / total_terms  # Compute the TF
        
        return tf_dict

    # Calculate Inverse Document Frequency (IDF) for the entire corpus using Scikit-learn's formula
    def _compute_idf(self, corpus):
        doc_count = len(corpus)  # Total number of documents in the corpus               
        doc_freq_term = {} #number of documents in the document set that contain the term t
        all_unique_terms = set()


        for doc in corpus:
            unique_terms = set(doc)  # Get unique terms in this document            
            all_unique_terms.update(unique_terms)
       
        for term in all_unique_terms:
            for doc in corpus:
                if term in doc and term not in doc_freq_term:
                    doc_freq_term[term] =1
                elif term in doc:
                    doc_freq_term[term] +=1
 
        # Compute IDF for each term using Scikit-learn's formula with smoothing  -  idf(t) = log [ (1 + n) / (1 + df(t)) ] + 1
        for term, doc_freq in doc_freq_term.items():
            self.idf_dict[term]= math.log((doc_count + 1)/(1+doc_freq))+1

       
    # Transform a document into its TF-IDF representation
    def transform(self, doc):
        tf = self._compute_tf(doc)
        tfidf = {}

        # Compute the TF-IDF for each term in the document
        for term, tf_value in tf.items():
            if term in self.idf_dict:
                tfidf[term] = tf_value * self.idf_dict[term]  # Multiply TF by IDF
            else:
                tfidf[term] = 0  # If the term is not in the IDF dict, its IDF will be zero
        
        return tfidf

    # Fit the model on the entire corpus and create the IDF values
    def fit(self, corpus):
        vocabulary = set()
        for doc in corpus:
            vocabulary.update(doc)  # Add all unique words from each document to the vocabulary
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(vocabulary))}

        # Compute the IDF values for the entire corpus using Scikit-learn's formula
        self._compute_idf(corpus)

    # Transform the entire corpus into TF-IDF representation
    def fit_transform(self, corpus):
        self.fit(corpus)
        return [self.transform(doc) for doc in corpus]