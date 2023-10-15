########### Import ###########
import pandas as pd
import seaborn as sn
import matplotlib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
%matplotlib inline
color = sn.color_palette()
pd.options.display.max_columns = None
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import warnings

########### Options ###########
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
warnings.simplefilter(action='ignore', category=Warning)


########### Simple Exploratory Data Analysis ###########
df = pd.read_csv("emails.csv")

def check_df(dataframe, target_col):
    print("########## HEAD ##########")
    print(dataframe.head())
    print("########## TARGET VALUE COUNTS ##########")
    print(dataframe[target_col].value_counts())
    print("########## MISSING VALUES ##########")
    print(dataframe.isnull().sum())
    print("########## DUPLICATED ##########")
    print(dataframe.duplicated().sum())
check_df(df, "spam")
"""
########## HEAD ##########
                                                text  spam
0  Subject: naturally irresistible your corporate...     1
1  Subject: the stock trading gunslinger  fanny i...     1
2  Subject: unbelievable new homes made easy  im ...     1
3  Subject: 4 color printing special  request add...     1
4  Subject: do not have money , get software cds ...     1
########## TARGET VALUE COUNTS ##########
0    4360
1    1368
########## MISSING VALUES ##########
text    0
spam    0
########## DUPLICATED ##########
33
"""
# drop duplicate
df.drop_duplicates(inplace=True)



########### Data Preprocessing ###########
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

def clean_text(text):
    text=text.lower() # Convert text to lowercase
    text=re.sub('[^a-z]',' ',text) # Replace non-alphabetic characters with space
    text=re.sub('subject',' ',text) # Replace the word "subject" with space
    text=word_tokenize(text) # Tokenize the words
    text=[word for word in text if len(word)>1] # Remove single-letter words
    return ' '.join(text)

df['text']=df['text'].apply(clean_text)

df.head()
"""
                                                text  spam
0  naturally irresistible your corporate identity...     1
1  the stock trading gunslinger fanny is merrill ...     1
2  unbelievable new homes made easy im wanting to...     1
3  color printing special request additional info...     1
4  do not have money get software cds from here s...     1
"""

sentences = df["text"].tolist()
labels = df["spam"].tolist()

cleaned_sentences = []
stop_words = set(stopwords.words("english"))
for sentence in sentences:
    words = sentence.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    cleaned_sentence = " ".join(filtered_words)
    cleaned_sentences.append(cleaned_sentence)
# Removed stop_words from text
cleaned_df = pd.DataFrame({'text': cleaned_sentences, 'spam': labels})


cv = CountVectorizer()
X = cv.fit_transform(cleaned_df['text']).toarray()
y = cleaned_df['spam']
# Spliting the data
X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.2, random_state=0)


# Number Transactions
datasets = [X_train, y_train, X_test, y_test]
names = ["X_train", "y_train", "X_test", "y_test"]
for data, name in zip(datasets, names):
    print(f"Number transactions {name} dataset: {data.shape}")
"""
Number transactions X_train dataset:  (4556, 33538)
Number transactions y_train dataset:  (4556,)
Number transactions X_test dataset:  (1139, 33538)
Number transactions y_test dataset:  (1139,)
"""



########### Model ###########
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train,y_train)

y_pred = model.predict(X_test) # predictions

print(f"""Train Score : {model.score(X_train, y_train)}
Test Score : {model.score(X_test, y_test)}""")
"""
Train Score : 0.9967076382791923
Test Score : 0.9894644424934153
"""

confusion_matrix(y_test, y_pred)
"""
[854,  11],
[  1, 273]]
"""

print(classification_report(y_test, y_pred))
"""
              precision    recall  f1-score   support
           0       1.00      0.99      0.99       865
           1       0.96      1.00      0.98       274
    accuracy                           0.99      1139
   macro avg       0.98      0.99      0.99      1139
weighted avg       0.99      0.99      0.99      1139
"""













