# Word2Vec

"""
Problem with Bag Of Words and TF-IDF Problems:
=>Both BOW and TF-IDF approach semantic information is not stored.TF-IDF gives importance to uncomman words.
=>There is definitely chance of overfitting.

Soltion- Word2Vec:
=>In this specific model,each word is represented as a vector of 32 or more dimension instead of a single number.
=>Here the semantic information and relation between different words is also preserved.

Steps to perform Word2Vec:
=>Tokenization of sentences.
=>Create Histograms.
=>Take most frequent words.
=>Create a matrix with all the unique words. It also represent the occurence of relation between the words.

>Install gensim library for Word2Vec:
  Command: pip install gensim
"""
import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import re

paragraph="""I have three visions for India. In 3000 years of our history people from all over the world have come and invaded us, captured our lands, conquered our minds. From Alexander onwards the Greeks, the Turks, the Moguls, the Portuguese, the British, the French, the Dutch, all of them came and looted us, took over what was ours. Yet we have not done this to any other nation. We have not conquered anyone. We have not grabbed their land, their culture and their history and tried to enforce our way of life on them. Why? Because we respect the freedom of others. That is why my FIRST VISION is that of FREEDOM. I believe that India got its first vision of this in 1857, when we started the war of Independence. It is this freedom that we must protect and nurture and build on. If we are not free, no one will respect us.We have 10 percent growth rate in most areas. Our poverty levels are falling. Our achievements are being globally recognised today. Yet we lack the self-confidence to see ourselves as a developed nation, self-reliant and self-assured. Isn’t this incorrect? MY SECOND VISION for India is DEVELOPMENT. For fifty years we have been a developing nation. It is time we see ourselves as a developed nation. We are among top five nations in the world in terms of GDP.I have a THIRD VISION. India must stand up to the world. Because I believe that unless India stands up to the world, no one will respect us. Only strength respects strength. We must be strong not only as a military power but also as an economic power. Both must go hand-in-hand. My good fortune was to have worked with three great minds. Dr.Vikram Sarabhai, of the Dept. of Space, Professor Satish Dhawan, who succeeded him and Dr. Brahm Prakash, father of nuclear material. I was lucky to have worked with all three of them closely and consider this the great opportunity of my life."""

## Preprocessing the data
text=re.sub(r'\[[0-9]*\]',' ',paragraph)
text=re.sub(r'\s+',' ',text)
text=text.lower()
text=re.sub(r'\d',' ',text)
text=re.sub(r'\s',' ',text)

sentences=nltk.sent_tokenize(text)
sentences=[nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i]=[word for word in sentences[i] if not word in stopwords.words('english')]

## Training the Word2Vec model
model=Word2Vec(sentences,min_count=1)
words=model.wv.vocab
print(words)

## Finding Word Vectors
vector=model.wv['war']
print(vector)

## Most similar words
similar=model.wv.most_similar('war')
print(similar)
