import pandas as pd
import numpy as np
#import nltk
import glob
import io

import random
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
import re

plt.ioff() #http://matplotlib.org/faq/usage_faq.html (interactive mode)
#Load data
contents = []
overalls = []
for filename in glob.iglob('./Review_Texts/*.dat', ):
#for filename in glob.iglob('./test_hotel/*.dat', ):
    with io.open(filename, 'r', encoding="utf8") as input_file:
        lines = input_file.readlines()
        
        for line in lines:
            #print(line)
            if '<Content>' in line:
                contents.append(line[9:].strip('\n'))
            if '<Overall>' in line:
                overalls.append(line[9:].strip('\n'))


#MAKE a Dataframe
df = pd.DataFrame(list(map(list, zip(contents, overalls))))
columns = ['review', 'score']
df.columns = columns
df.review = df.review.str.lower()
df.review = df.review.apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

#Assume that negative reviews have 0-1 score and positive have 4-5 score
def filter(user):
    score = int(user)
    if (score <2):     
        return 'neg'
    elif (score >=2 and score <4):
        return 'neutral'
    else:
        return 'pos'

df['category'] = df['score'].apply(filter)


# Limiting # of positive reviews to 17K
short_pos = df.review[df['category'] == 'pos']
short_pos = short_pos[:17000]
print("\nLENGTH OF POSITIVE REVIEWS: ")
print(len(short_pos))
short_neg = df.review[df['category'] == 'neg']
print("\nLENGTH OF NEGATIVE REVIEWS: ")
print(len(short_neg))
print("\n")

#CREATE A NEW DF FOR NEGATIVE AND POSITIVE REVIEWS with SCORE COLUMN
neg_df = pd.DataFrame(short_neg)
neg_df['label'] = 0

pos_df = pd.DataFrame(short_pos)
pos_df['label'] = 1

from sklearn.utils import shuffle
new_df = pd.concat([neg_df, pos_df])

new_df = shuffle(new_df)

# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
X = new_df['review'].values
y = new_df['label'].values

max_words = 300
top_words = 5000

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Summarize number of classes
print("Classes: ")
print(np.unique(y))

#import re
#m = re.search(r'\d+', X_train)
#numeric = m.group()
#int(numeric)
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
	
def build_model(top_words, max_words):
	model = Sequential()
	model.add(Embedding(top_words, 32, input_length=max_words))
	model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	model.add(Dense(250, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	return model

model = build_model(top_words, max_words)

# serialize model to JSON
model_json = model.to_json()
with open("ConvNet.json", "w") as json_file:
    json_file.write(model_json)

filepath="weightsCONVNET.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

callbacks_list = [checkpoint]

history = model.fit(X_train, y_train, validation_split=0.1, nb_epoch=2, batch_size=128, verbose=2, callbacks=callbacks_list)

scores = model.predict_proba(X_test)

from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, scores)
roc_auc = auc(fpr, tpr)
np.savetxt('testCONVNET.out', (fpr, tpr, thresholds))

fig = plt.figure()
ax = plt.subplot(111)
plt.title('ROC')
ax.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
ax.legend(loc='lower right')
ax.plot([0,1],[0,1],'r--')
ax.set_xlim([-0.1,1.1])
ax.set_ylim([-0.1,1.1])
ax.set_ylabel('True Positive Rate')
ax.set_xlabel('False Positive Rate')
plt.show()
fig.savefig('ROC_AUC CONVNET.png')

print(history.history.keys())

f = open("history CONVNET.txt","w")
f.write( str(history) )
f.close()
# summarize history for loss

print(history.history.keys())
ax2 = plt.subplot(121)
plt.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
plt.title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig.savefig('Learning curve  CONVNET.png')
print("END !!!")