import tensorflow
import tensorflow_datasets as tfds
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

'''
This will get the dataset of IMDB reviews from the tensorflow library.
The data is present in the imdb and info consists of name, version, description, etc.
'''

import numpy as np

train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

# str(s.tonumpy()) is needed in Python3 instead of just s.numpy()
for s,l in train_data:
  training_sentences.append(s.numpy().decode('utf8'))
  training_labels.append(l.numpy())
  
for s,l in test_data:
  testing_sentences.append(s.numpy().decode('utf8'))
  testing_labels.append(l.numpy())
  
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

'''
The imdb dataset consists of 25000 training data and 25000 testing data.
The for loop will extract the data from the dataset and will store in as arrays as reviews and their labels. The labels and reviews are store in as the numpy arrays.
'''

vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type='post'
oov_tok = "<OOV>"

'''
The vocab_size indicates maximum number of words to keep based on the word frequency. Thats is keep first 10000 words with maximum frequency.
The embedding_dim determines the different dimensions in which the ouput will be reprsented.
The max_length represent the maximum size a sentence should be of. If it is <120 oov_tok is padded at the end and if the length of input sentence is >120 then the 1st 120 words are accepted else are truncated from the end and this is represented by trunc_type. By default the value of the trunc_type is pre i.e. value is padded or truncated from the front side.
'''

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences,maxlen=max_length)

'''
word_index consists of the words with there values for example:
{'<OOV>': 1, 'the': 2, 'and': 3, 'a': 4, 'of': 5, 'to': 6, 'is': 7, 'br': 8, 'in': 9, 'it': 10}.

Sequences consists of the sentences in the form of the words.For example for the 1st review the sequence will be:
[12, 14, 33, 425, 392, 18, 90, 28, 1, 9, 32, 1366, 3585, 40, 486, 1, 197, 24, 85, 154, 19, 12, 213, 329, 28, 66, 247, 215, 9, 477, 58, 66, 85, 114, 98, 22, 5675, 12, 1322, 643, 767, 12, 18, 7, 33, 400, 8170, 176, 2455, 416, 2, 89, 1231, 137, 69, 146, 52, 2, 1, 7577, 69, 229, 66, 2933, 16, 1, 2904, 1, 1, 1479, 4940, 3, 39, 3900, 117, 1584, 17, 3585, 14, 162, 19, 4, 1231, 917, 7917, 9, 4, 18, 13, 14, 4139, 5, 99, 145, 1214, 11, 242, 683, 13, 48, 24, 100, 38, 12, 7181, 5515, 38, 1366, 1, 50, 401, 11, 98, 1197, 867, 141, 10].

padded consists of the sequences with padding performed on it to make the sentences of same size (here 120 words).
'''

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(padded[3]))
print(training_sentences[3])

'''
The reverse_word_index provide the sentences back from the numbers which we gave in word_index and convert the sequences or padded back to the correct sentences or reviews.
'''

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size,
			      embedding_dim,
			      input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


'''
In this the model is having some different then the regular ones.
Here we are using tf.keras.layers.Embedding for word embbeding.In the model the flatten layers will have output_shape of (None, 1920) which can make the calculation take more time.Instead of this we can use tf.keras.layers.GlobalAveragePooling layer which will have and output_shape of (None,64).
'''

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size,
			      embedding_dim,
    			      input_length=max_length),
    tf.keras.layers.GlobalAveragePooling(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


num_epochs = 10
model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))

'''
After training the accuracy is 100% and the validation accuracy is ~83% which means there is a problem of overfitting.
'''

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)

'''
The above code will give us the weights of the 1st layer. The output of above code will be the shape of the weights of the 1st layer i.e. (10000, 16).
'''


import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

'''
To visualize the embedding of the words get the meta and the vecs file from the above code as .tsv file which means tab seperated values.
Get the files and go to projector.tensorflow.org and load the files.
1st upload the vecs.tsv file and then the meta.tsv file. vecs.tsv file consist of the words and the meta.tsv file consists of the weights.

Click on the Sphereize data which will show the clusters formed by the words and visualize the whole data in the form of the sphere i.e. 3d model.

On the right hand side you can visualize search the word and it will show the nearest points if the word and the clusters are highlighted.
'''

'''
To make the prediction:
'''

sentence = "I really think this is amazing. honest."
sequence = tokenizer.texts_to_sequences([sentence])
model.predict(pad_sequences(sequence,maxlen=max_length, truncating=trunc_type))
