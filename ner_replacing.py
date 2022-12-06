import spacy
from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer()
nlp = spacy.load("ro_core_news_lg")
nlp.add_pipe("merge_entities")
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


def replace_ner(text):
    doc = nlp(text)
    out = ""
    for tok in doc:
        text = tok.text
        if tok.ent_type_:
            text = tok.ent_type_
        out += text + tok.whitespace_
    return out


print(replace_ner(raw_text))
