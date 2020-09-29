import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from textblob import Word
import re
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier

data = pd.read_csv('text_emotion.csv') # Data.world

# Drop the author Column as it's not useful
data = data.drop('author', axis=1)

# Dropping rows with other emotion labels
data = data.drop(data[data.sentiment == 'anger'].index)
data = data.drop(data[data.sentiment == 'boredom'].index)
data = data.drop(data[data.sentiment == 'enthusiasm'].index)
data = data.drop(data[data.sentiment == 'empty'].index)
data = data.drop(data[data.sentiment == 'fun'].index)
data = data.drop(data[data.sentiment == 'relief'].index)
data = data.drop(data[data.sentiment == 'surprise'].index)
data = data.drop(data[data.sentiment == 'love'].index)
data = data.drop(data[data.sentiment == 'hate'].index)
data = data.drop(data[data.sentiment == 'neutral'].index)
data = data.drop(data[data.sentiment == 'worry'].index)

# Making all letters lowercase
data['content'] = data['content'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# Removing Punctuation, Symbols
data['content'] = data['content'].str.replace('[^\w\s]',' ')

# Removing Stop Words using NLTK
stop = stopwords.words('english')
data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# Lemmatisation
data['content'] = data['content'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

#Correcting Letter Repetitions
def de_repeat(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

data['content'] = data['content'].apply(lambda x: " ".join(de_repeat(x) for x in x.split()))

# Code to find the top 10,000 rarest words appearing in the data
freq = pd.Series(' '.join(data['content']).split()).value_counts()[-10000:]

# Removing all those rarely appearing words from the data
freq = list(freq.index)
data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

# Encoding output labels 'sadness' as '1' & 'happiness' as '0'
lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(data.sentiment.values)

# Splitting into training and testing data in 90:10 ratio
X_train, X_val, y_train, y_val = train_test_split(data.content.values, y, stratify=y, random_state=42, test_size=0.1, shuffle=True)

# Extracting Count Vectors Parameters
count_vect = CountVectorizer(analyzer='word')
count_vect.fit(data['content'])
X_train_count =  count_vect.transform(X_train)
X_val_count =  count_vect.transform(X_val)

# Model : Linear SVM
lsvm = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None)
lsvm.fit(X_train_count, y_train)
y_pred = lsvm.predict(X_val_count)

# Find the Accuracy of the Training Model
Temp = accuracy_score(y_pred, y_val)
Accuracy = str(Temp)
print('LSVM with Count Vectors accuracy result : ' + Accuracy)

# Read the text file and seperate each sentence with ";" symbol 
# Untuk bagian ini kami belum membuat pengambilan data secara langsung melainkan menginput nama filenya karena masih dalam bentuk prototype
subtitle = pd.read_csv('CyberTruck-Phone-Impressions-Ridiculous.txt', sep = ";", header = None)

# Transpose the matrix from column to row and row to column
subtitle = subtitle.T

# Drop all "NULL" data from the dataset
subtitles = subtitle.dropna()

# Doing some preprocessing on the subtitles
subtitles[0] = subtitles[0].str.replace('[^\w\s]',' ')
stop = stopwords.words('english')
subtitles[0] = subtitles[0].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
subtitles[0] = subtitles[0].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

# Extracting Count Vectors feature from the subtitles
subtitle_count = count_vect.transform(subtitles[0])

# Predicting the emotion of the subtitle using trained linear SVM
subtitle_pred = lsvm.predict(subtitle_count)
print(subtitle_pred)

# Count all the happiness and sadness number from "subtitle_pred"
happinessCount = np.count_nonzero(subtitle_pred == 0)
sadnessCount = np.count_nonzero(subtitle_pred == 1)

# Get the percentage for both happiness and sadness
happinessPercent = happinessCount / (happinessCount + sadnessCount) * 100
sadnessPercent = sadnessCount / (happinessCount + sadnessCount) * 100

# Make the percentage into a string for printing
hpPercentage = str(happinessPercent)
sdPercentage = str(sadnessPercent)

# Print both happiness and sadness results
print("Happiness Percent = " + hpPercentage + "%")
print("Sadness Percent = " + sdPercentage + "%")