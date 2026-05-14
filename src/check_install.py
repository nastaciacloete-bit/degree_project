# imports
import numpy as np #for numerical computing
import pandas as pd #for working with tabular data (CSV files)
import sklearn #main machine learning library used
import matplotlib #for plotting charts and visualisations
import nltk #natural language processing library
import spacy #NLP library often used for tokenisation and language processing
import torch #deep learning library
import transformers #hugging face transformers, often used for advanced NLP models
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer #used for VADER sentiment analyser class, lexicon/rule-based sentiment analysis tool
from tqdm import tqdm #progress bars during loops

# proof of imports
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)

# check sentiment analysis 
sia = SentimentIntensityAnalyzer() #creates an istance of the VADER sentiment analyser
print(sia.polarity_scores("This project setup is great!")) #runs quick sentiment check on sample sentence
