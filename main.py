# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
import re
import spacy
from sklearn.feature_extraction import DictVectorizer
from difflib import SequenceMatcher

def readdishdb():
    i = 1
    dishdb = {}
    f = open("dishdb", "r")
    for dish in f:
        dish = dish.rstrip()
        dishdb[dish] = i
        i += 1
    return dishdb

def readrestdb():
    i = 1
    restdb = {}
    f = open("restdb", "r")
    for rest in f:
        rest = rest.rstrip()
        restdb[rest] = i
        i += 1
    return restdb

TAG_RE = re.compile(r'<[^>]+>')
BR_RE = re.compile(r'(<br/>)+')
NUM_RE = re.compile(r'[0-9]+')
RETURN_RE = re.compile(r'[\n]+')
WHITE_RE = re.compile(r'^\s+')
DASH_RE = re.compile(r'-')
NOT_RE = re.compile(r"n't")
def remove_tags(text):
    text = RETURN_RE.sub("", text)
    text = text.replace(" n't", " not")
    text = text.replace("(", " ")
    text = text.replace("/", " ")
    text = DASH_RE.sub('', text)
    text = WHITE_RE.sub("", text)
    text = BR_RE.sub("\n", text)
    text = NUM_RE.sub('', text)
    text = TAG_RE.sub(' ', text)
    return text

nlps = spacy.load("en_core_web_sm")

def makesent(review, nlps):
    doc = nlps(remove_tags(review.lower()))
    wordlist = []
    for token in doc:
        if token.text == "-":
            wordlist.append(",")
            continue
        if token.pos_ == "NFP PUNCT" or token.pos_ == "PUNCT":
            if token.text == ",":
                wordlist.append(",")
            else:
                wordlist.append(".")
            continue
        if not token.text == "it":
            wordlist.append(token.text)

    sent = " ".join(wordlist)
    sent = sent.replace(' .', ".")
    doc = nlps(sent)
    retsent = []
    for sentline in doc.sents:
        retsent.append(sentline.text.replace(".", ""))

    return retsent

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

nlp = spacy.load("model-best")

def findfood(nlp, text):
    doc = nlp(text)
    foodlist = []
    for ent in doc.ents:
        if ent.label_ == "FOOD":
            doc1 = nlp(ent.text)
            for ent1 in doc1.ents:
                if ent1.label_ == "FOOD":
                    foodlist.append(ent.text)
                    break
    return foodlist

def matchmenulist(foodlist, menulist):
    bestmatch = foodlist[0]
    bestscore = 0
    for food in foodlist:
        for X in menulist:
            Y = food

            cosine = similar(X, Y)
            if cosine > 0.5:
                if cosine > bestscore:
                    bestmatch = X
                    bestscore = cosine

    return (bestmatch, bestscore)

dishdb = readdishdb()
restdb = readrestdb()





lemmatizer = WordNetLemmatizer()

stop_words = stopwords.words('english')

data = pd.read_csv('Restaurant_Reviews.csv')


words = []
import re
from nltk.tag import pos_tag

def cleanwords(words):
    tokens = []
    for token, tag in pos_tag(words):
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)
        token = re.sub('[.,)"]', '', token)
        token = re.sub("n't", 'not', token)
        token = re.sub("'", '', token)
        tokens.append(token)

    return tokens

words = []
for i in data['Review']:
#line = i.split()
    i = i.lower()
    for word in cleanwords(word_tokenize(i)):
        if len(word) <= 2:
            continue
        if word not in words:
            words.append(word)

for i in words:
    data[i] = 0

counter = 0
for i in data['Review']:
    i = i.lower()
    for j in cleanwords(word_tokenize(i)):
        if len(j) <= 2:
            continue
        data.at[counter,j] += 1
    counter += 1




review = data['Review']
y = data['Liked']

X = data.drop(columns=['Review','Liked'])



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)

model = SVC()

param_grid = {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001], 'kernel':['linear','rbf']}

grid = GridSearchCV(model,param_grid,refit = True, verbose=2)

grid.fit(X_train,y_train)



pred = grid.predict(X_test)

accuracy = accuracy_score(y_test,pred)

print(accuracy)

'''
coefss = []
for i in range(len(grid.coef_[0])):
    coefss.append(grid.coef_[0][i])


coef_dict = {}
for coef, feat in zip(coefss,X.columns):
    coef_dict[feat] = coef
'''

'''
sort_orders = sorted(coef_dict.items(), key=lambda x: x[1], reverse=True)

for i in sort_orders:
    print(i[0], i[1])
'''

dishrate = {}
dishfile = open("dishratings.csv", "w")
dishfile.write('"RestID","DishID","Rating"' + "\n")
for rest in restdb:
    menulist = []
    menufile = open(rest + ".menu", "r")
    for menu in menufile:
        menu = menu.rstrip()
        menulist.append(menu.lower())
    reviewfile = rest + ".reviews"
    restid = restdb[rest]
    f = open(reviewfile, "r")
    for l in f:
        fields = l.split("|")
        if len(fields[1]) == 0:
            continue
        for review in makesent(fields[1], nlps):
            counter = 0
            datai = {}

            for word in cleanwords(word_tokenize(review)):
                if len(word) <= 2:
                    continue
                try:
                    datai[word] += 1
                except:
                    datai[word] = 1

            inputdata = []
            inputdata.append({})
            for word in words:
                try:
                    inputdata[0][word] = datai[word]
                except:
                    inputdata[0][word] = 0

            v = DictVectorizer(sparse=False)
            X = v.fit_transform(inputdata)
            f = findfood(nlp, review)
            if len(f) == 0:
                continue
            predicted = grid.predict(X)
            m = matchmenulist(f, menulist)
            if m[1] > 0.7:
                if predicted == 1:
                    try:
                        dishrate[m[0]] += 1
                    except:
                        dishrate[m[0]] = 1
                else:
                    try:
                        dishrate[m[0]] -= 1
                    except:
                        dishrate[m[0]] = -1

    for dish in dishrate:
        print(str(restid) + "," + str(dishdb[dish]) + "," + str(dishrate[dish]) + "\n")
        dishfile.write(str(restid) + "," + str(dishdb[dish]) + "," + str(dishrate[dish]) + "\n")

dishfile.close()

