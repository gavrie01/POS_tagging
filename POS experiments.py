import string
import os
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import Image
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import word_tokenize, pos_tag, ne_chunk # parts of speech
nltk.download('maxent_ne_chunker')
nltk.download('words')
import spacy
#-------------------------
generated_text = []
maximum_generated_text_length = 10
# Initialize the dictionary for Markov chain
mc_dict = {}
#download english model
# execute 'python -m spacy download en_core_web_sm' or via PIP
nlp = spacy.load("en_core_web_sm") 
# --------------------------------------------
# clear console on start
clear = lambda: os.system('cls')  # on Windows System
os.system('clear')  # on Linux System
clear()
# ---------------------------------------------

# Load the text from a file
with open('data/Dorian_Grey.txt', encoding='UTF-8') as file:
    text = file.read()

#-------------------------

# Remove punctuation and convert to lowercase
exclude = set(string.punctuation)
text = ''.join(ch for ch in text if ch not in exclude).lower().strip()
text = ''.join(s for s in text if s not in ('“', '”', '’'))
text = ''.join([i for i in text if not i.isdigit()])
# Tokenize the text into words
words = word_tokenize(text)
#-------------------------------


# breakdown to parts of speech
tagged_text = pos_tag(words)

#write into file to explore better
with open('data/POS_mark_up.txt', 'w', encoding='UTF-8') as f:
    for word, pos in tagged_text:
        f.write(f"{word}: {pos}\n")
# separate by part of speech: Noun
with open('data/POS_Noun.txt', 'w', encoding='UTF-8') as f:
    for word, pos in tagged_text:
        if pos in ['NN', 'NNS', 'NNP', 'NNPS']:
            f.write(f"{word}: {pos}\n")
#nouns = [word for word, pos in tagged_text if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
#print("Nouns:", nouns)

## separate by part of speech: Verb
with open('data/POS_Verb.txt', 'w', encoding='UTF-8') as f:
    for word, pos in tagged_text:
        if pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            f.write(f"{word}: {pos}\n")
# separate by part of speech: Adverb
with open('data/POS_Adverb.txt', 'w', encoding='UTF-8') as f:
    for word, pos in tagged_text:
        if pos in ['RB', 'RBR', 'RBS']:
            f.write(f"{word}: {pos}\n")

# separate by part of speech: Adjective
with open('data/POS_Adjective.txt', 'w', encoding='UTF-8') as f:
    for word, pos in tagged_text:
        if pos in ['JJ', 'JJR', 'JJS']:
            f.write(f"{word}: {pos}\n")


# separate by part of speech: Determiner
with open('data/POS_Determiner.txt', 'w', encoding='UTF-8') as f:
    for word, pos in tagged_text:
        if pos in ['DT', 'PDT']:
            f.write(f"{word}: {pos}\n")

# separate by part of speech: Rest
with open('data/POS_Rest.txt', 'w', encoding='UTF-8') as f:
    for word, pos in tagged_text:
        if pos in ['CC', 'CD', 'IN', 'MD', 'RP', 'TO', 'UH', 'WDT', 'WP', 'WP$', 'WRB']:
            f.write(f"{word}: {pos}\n")
 

 # NER------------------
named_entities = ne_chunk(tagged_text)
proper_nouns = [subtree.leaves()[0][0] for subtree in named_entities if type(subtree) == nltk.Tree]
with open('data/NER.txt', 'w', encoding='utf-8') as f:
    for subtree in named_entities:
        if isinstance(subtree, nltk.Tree):
            if subtree.label() == 'PERSON':
                name = ' '.join([word for word, pos in subtree.leaves()])
                f.write(name + '\n')
#-----------------------