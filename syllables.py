import string
import os
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import nltk
import pyphen  # breakdown words into syllables
import spacy

# -------------------------
# clear console on start
clear = lambda: os.system('cls')  # on Windows System
os.system('clear')  # on Linux System
clear()
# ---------------------------------------------

# Load the text from a file
with open('data/Dorian_Grey.txt', encoding='UTF-8') as file:
    text = file.read()

# -------------------------
# Remove punctuation and convert to lowercase
exclude = set(string.punctuation)
text = ''.join(ch for ch in text if ch not in exclude).lower().strip()
text = ''.join(s for s in text if s not in ('“', '”', '’'))
text = ''.join([i for i in text if not i.isdigit()])
# ---------------------------------

# Tokenize the text into words
words = word_tokenize(text)

# Load a specific language (e.g., English)
dic = pyphen.Pyphen(lang='en')

# Dictionary to store words and their syllables
word_syllables = {}

# Break each word into syllables and store in the dictionary
for word in words:
    syllables_for_word = dic.inserted(word).split('-')
    word_syllables[word] = syllables_for_word

# Print the dictionary
for word, syllables in word_syllables.items():
    print(f"{word}: {'-'.join(syllables)}")
