import spacy
from sklearn.feature_extraction import DictVectorizer
import rowordnet as rwn

vec = DictVectorizer()
wn = rwn.RoWordNet()
nlp = spacy.load("ro_core_news_lg")
raw_text = "Ar trebui să se producă duminică, 3 iunie. Repetate, pentru a determina, din al doilea tur, " \
           "cine va gestiona Primăria Chişinău timp de un an, până la următoarele alegeri locale generale. La fel ca " \
           "şi în turul 1, puţini mai vorbesc astăzi despre alegeri. Nu se prea aud mesaje de mobilizare a " \
           "alegătorilor, la fel ca şi în turul 1, iar cele care apar, pe alocuri, sunt mobilizatoare doar pentru un " \
           "anume partid, pentru un electorat anume. La fel ca şi în turul 1, nimeni nu a ieşit să spună că aceste " \
           "alegeri ar fi extrem de importante, că în lipsa lor municipalitatea ar pierde bani, proiecte, " \
           "locuri de muncă, chişinăuienii fiind tentaţi în continuare să-şi părăsească familiile, plecând peste " \
           "hotare, în căutarea unei vieţi mai bune. Chiar şi cel care a generat aceste alegeri noi, deşi instalase, " \
           "ceva mai devreme, un primar interimar care ar fi putut deţine şi în următoarele luni interimatul, " \
           "nu a suflat un cuvânt, nici până şi nici după 20 mai, despre importanţa mobilizării electoratului din " \
           "municipiul Chişinău pentru aceste alegeri noi. După 20 mai, tăcerea celor care ar trebui să mobilizeze " \
           "alegătorii pare şi mai „argumentată”, fiind eclipsată de omorul, în plină noapte, a unui tânăr, " \
           "în centrul Chişinăului. Nu e primul omor de acest fel, comis de persoane care practică lupte sportive. " \
           "S-ar putea să nu fie şi ultimul, atâta timp cât guvernarea e protejată de astfel de sportivi luptători. " \
           "În 2011, dublul campion mondial al kickboxing, Ion Şoltoianu, a împuşcat mortal, la o terasă din centrul " \
           "Chişinăului, un fost poliţist, pe Ion Stratulat. Ulterior, Şoltoianu a fost condamnat la 12 ani de " \
           "privaţiune de libertate pentru omor, şantaj şi deţinere ilegală de arme. Azi nu se mai aude nimic despre " \
           "Şoltoianu, deşi avocaţii săi anunţaseră că vor face apel la CEDO. "
doc = nlp(raw_text)
doc2 = nlp("Şoltoianu, deşi avocaţii săi anunţaseră că vor face apel la CEDO.")


def extract_hypernymes(word):
    hyper = []
    synset_ids = wn.synsets(literal=word.lemma_, pos=rwn.synset.Synset.Pos.NOUN)
    if len(synset_ids) > 0:
        c = wn.synset_to_hypernym_root(synset_ids[0])
        for c1 in c:
            hyper.append(wn(c1).literals[0])
    return hyper


def extract_feat_for_word(word, rel):
    hyper = []
    if word.pos_ == "NOUN":
        hyper = extract_hypernymes(word)

    return {
        "text": word.text,
        "pos": word.pos_,
        "morpho": word.morph,
        "lemma": word.lemma_,
        "dep": word.dep_,
        "hyper": hyper,
        "position_rel": rel
    }


for word in doc2:
    vect = []
    if word.head != word:
        vect.append(extract_feat_for_word(word.head, "head"))
        for siblings in word.head.children:
            vect.append(extract_feat_for_word(siblings, "0" if word != siblings else "main"))
    else:
        vect.append(extract_feat_for_word(word, "main"))
    for chil in word.children:
        vect.append(extract_feat_for_word(chil, "children"))
    print(word, ": ")
    print("Features:")
    print(vect)
    print("Features as numeric data:")
    print(vec.fit_transform(vect).toarray())

