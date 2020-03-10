import vparser
import numpy as np
import matplotlib.pyplot as plt

ignore = " 1234567890\t\n()[]!.,:;-_#$%&/={}'"

corpus_eng = "../corpus/english_botany.txt"
corpus_ger = "../corpus/german_botany.txt"
corpus_lat = "../corpus/latin_botany.txt"
corpus_prt = "../corpus/portuguese_botany.txt"
corpus_trk = "../corpus/turkish_botany.txt"

def rank_letter_freq(text):
    text = text.lower()
    dic = {}
    total = 0
    for letter in text:
        if not letter in ignore:
            if not letter in list(dic):
                dic[letter] = 1
            else:
                dic[letter] += 1
            total += 1
    res = []
    for letter in list(dic):
        res += [dic[letter]/total]
    res.sort(reverse = True)
    return res

def rank_word_sizes(text):
    sizes = [0]*30
    total = 0
    for c in ignore:
        text = text.replace(c, " ")
    text = text.split(" ")
    for word in text:
        if (len(word) < 30) and (len(word)>0):
            sizes[len(word)] += 1
            total += 1
    for i in range(len(sizes)):
        sizes[i] = sizes[i] / total
    return sizes


def get_corpus_text(file):
    f = open(file, "r", errors="ignore")
    t = f.read()
    f.close()
    return t

def graph_top_letter_frequencies():
    y = rank_letter_freq(vparser.get_all_text())
    x = list(range(len(y)))
    plt.plot(x, y, label="voynich")
    
    y = rank_letter_freq(get_corpus_text(corpus_eng))
    x = list(range(len(y)))
    plt.plot(x, y, label="english")    
    
    y = rank_letter_freq(get_corpus_text(corpus_ger))
    x = list(range(len(y)))
    plt.plot(x, y, label="german")        
    
    y = rank_letter_freq(get_corpus_text(corpus_lat))
    x = list(range(len(y)))
    plt.plot(x, y, label="latin")    
    
    y = rank_letter_freq(get_corpus_text(corpus_prt))
    x = list(range(len(y)))
    plt.plot(x, y, label="portuguese")        
    
    y = rank_letter_freq(get_corpus_text(corpus_trk))
    x = list(range(len(y)))
    plt.plot(x, y, label="turkish")           
    
    plt.legend(loc="upper right")
    plt.show()    
    
    
def graph_word_sizes():
    y = rank_word_sizes(vparser.get_all_text())
    x = list(range(len(y)))
    plt.plot(x, y, label="voynich")
    
    y = rank_word_sizes(get_corpus_text(corpus_eng))
    x = list(range(len(y)))
    plt.plot(x, y, label="english")    
    
    y = rank_word_sizes(get_corpus_text(corpus_ger))
    x = list(range(len(y)))
    plt.plot(x, y, label="german")        
    
    y = rank_word_sizes(get_corpus_text(corpus_lat))
    x = list(range(len(y)))
    plt.plot(x, y, label="latin")    
    
    y = rank_word_sizes(get_corpus_text(corpus_prt))
    x = list(range(len(y)))
    plt.plot(x, y, label="portuguese")        
    
    y = rank_word_sizes(get_corpus_text(corpus_trk))
    x = list(range(len(y)))
    plt.plot(x, y, label="turkish")           
    
    plt.legend(loc="upper right")
    plt.show()    