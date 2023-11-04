import string
import os
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import random
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import word_tokenize, pos_tag, ne_chunk # parts of speech
nltk.download('maxent_ne_chunker')
nltk.download('words')
import spacy  #NER
#-------------------------
# Initialize the dictionary for Markov chain
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
#---------------------------------


# Tokenize the text into words
words = word_tokenize(text)
#-------------------------------
# Process the text with spaCy
doc = nlp(text)

# NER detection
with open('data/NER.txt', 'w', encoding='utf-8') as f:
#ent.label_ in ["PERSON", "ORG", "DATE", "TIME", "LANGUAGE", "EVENT", "NORP", "ORDINAL", "CARDINAL"]]
    for ent in doc.ents:
        f.write(f'{ent.text} : {ent.label_}\n')
        

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
 