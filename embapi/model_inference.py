import gensim
from pathlib import Path
import numpy as np
from nltk.tokenize import word_tokenize


def load_model(modelPath):
    ''' load pretrained model
    params 
    * path: path to trained model file
    return: 
    * model
    '''
    trained_model = gensim.models.KeyedVectors.load_word2vec_format(
        modelPath , binary=True)

    return trained_model

def get_word_embeddings(model, words):
    ''' get embeddings for a list of words
    params:
    * model: model
    * words: list of words
    
    return: 
    * tuple of size 2. first element is the array of 
      embedding-vectors, second element is the list of 
      out of vocabulary-words
    '''
    embeddingDimensions = model.vector_size
    embeddings = np.zeros((len(words), embeddingDimensions))

    ooV = []
    for i,w in enumerate(words):
        try:
            embeddings[i,] = model.word_vec(w)
        except KeyError: # handle out of vocabulary words
            ooV.append(w)
    
    return embeddings, ooV

def get_sentence_embedding(model, sentence, pooling='max'):
    ''' get the embeddings for a sentence
    params:
    * model: model
    * sentence: sentence (string)
    * pooling: method to aggregate the word-level embeddings
      "max" or "mean"
    return: 
    * tuple of size 2. first element is a array containing 
      the sentence-embedding, second element are the 
      out-of-vocabulary words
    '''
    words = word_tokenize(sentence, language='german')
    wordEmbeddings, ooV = get_word_embeddings(model, words)

    if pooling == 'max':
        sentEmbedding = (wordEmbeddings+1).max(axis=0)
    elif pooling == 'mean':
        sentEmbedding = (wordEmbeddings+1).mean(axis=0)
    else:
        raise ValueError('invalid pooling method')

    return sentEmbedding, ooV
