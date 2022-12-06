import spacy
from sklearn.feature_extraction import DictVectorizer
import rowordnet as rwn

vec = DictVectorizer()
wn = rwn.RoWordNet()
nlp = spacy.load("ro_core_news_lg")
sentence = nlp("Azi nu se mai aude nimic despre Şoltoianu, deşi avocaţii săi anunţaseră că vor face apel la CEDO.")
words = [w.text for w in sentence]
for i in range(len(words)):
    start = 0 if i<=3 else i-3
    end = len(words)-1 if i>=len(words)-1 else i+4
    context = words[start:end]
    print("The word: ",words[i]," and its context:",context)
    print("Embedding:")
    print(nlp(" ".join(context)).vector)
    print("\n\n")


