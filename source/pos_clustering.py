import vparser
import math
import random
import copy
import nltk
import csv
import numpy as np
import matplotlib.pyplot as plt

pt_bible = "../corpus/portuguese_bible.txt"
en_bible = "../corpus/english_bible.txt"

def get_text(file):
    try:
        f = open(file, "r")
        text = f.read()
        f.close()
    except:
        f.close()
        f = open(file, "r", encoding='utf-8')
        text = f.read()
        f.close()        
    return text

def generate_bigram_maps(lists):
    maps = {}
    total = len(lists)
    progress = 0
    lastprog = 0
    for sentence in lists:
        progress += 1
        if ((progress / total) * 100) > (lastprog + 5):
            lastprog = ((progress / total) * 100)
            print ((progress / total) * 100, "%")
        previous = "$START$"
        for index, word in enumerate(sentence):
            try:
                nextword = sentence[index+1]
            except:
                nextword = "$END$"
            try:
                wordinfo = maps[word]
            except:
                wordinfo = [{},{},1]
                maps[word] = wordinfo
            try:
                wordinfo[0][previous] = wordinfo[0][previous] + 1
            except:
                wordinfo[0][previous] = 1
            try:
                wordinfo[1][nextword] = wordinfo[1][nextword] + 1
            except:
                wordinfo[1][nextword] = 1            
            wordinfo[2] += 1
            previous = word
    for word in list(maps):
        wordinfo = maps[word]
        count = wordinfo[2]
        for prev in list(wordinfo[0]):
            wordinfo[0][prev] = wordinfo[0][prev] / count
        for nxt in list(wordinfo[1]):
            wordinfo[1][nxt] = wordinfo[1][nxt] / count        
    return maps
    
def cosine_word_similarity(word1info, word2info):
    return (cosine_bag_similarity(word1info[0], word2info[0]) + cosine_bag_similarity(word1info[1], word2info[1])) / 2
    
def cosine_bag_similarity(bag1, bag2):
    allwords = list(set(bag1) | set(bag2))
    a = 0
    for word in allwords:
        try:
            b1 = bag1[word]
        except:
            b1 = 0
        try:
            b2 = bag2[word]
        except:
            b2 = 0
        a += b1 * b2
    c1 = 0
    for word in list(bag1):
        c1 += bag1[word]**2
    c2 = 0
    for word in list(bag2):
        c2 += bag2[word]**2
    return a / (math.sqrt(c1) * math.sqrt(c2))

def find_most_similar(maps, showprogress=False, antilarge=True, smoothing = True):
    best = -1
    bw1 = None
    bw2 = None
    total = len(list(maps))
    progress = 0
    lastprog = 0    
    words = list(maps)
    for id1 in range(len(words)):
        word1 = words[id1]
        if (showprogress):
            progress += 1
            if ((progress / total) * 100) > (lastprog + 0.1):
                lastprog = ((progress / total) * 100)
                print ((progress / total) * 100, "%")        
        for id2 in range(id1+1, len(words)):
            word2 = words[id2]
            if (word1 != word2):
                sim = cosine_word_similarity(maps[word1], maps[word2])
                if (smoothing):
                    sim = sim + 0.1
                if (antilarge):
                    sim = sim / ((maps[word1][2] + maps[word2][2])**2)                
                if (sim > best):
                    best = sim
                    bw1 = word1
                    bw2 = word2
                    if (not smoothing) and (sim > 0.9999): #already found perfect pair
                        return bw1, bw2, best
    return bw1, bw2, best
                    
def find_fast_order_similar(maps, tests, antilarge=False):
    best = -1
    bw2 = None
    words = list(maps)
    word1 = min(words, key=lambda w : maps[w][2])
    wordcount = len(words)
    if (wordcount == 1):
        return "FINISH"    
    if (wordcount-1 < tests):
        for word2 in words:
            if (word2 != word1):
                sim = cosine_word_similarity(maps[word1], maps[word2])
                if (antilarge):
                    sim = sim / ((maps[word1][2] + maps[word2][2]))    
                if (sim > best):
                    best = sim
                    bw2 = word2      
                    if (not smoothing) and (sim > 0.9999): #already found perfect pair
                        return word1, bw2, best                
    else:
        for i in range(tests):
            word2 = random.choice(words)
            while word2 == word1:
                word2 = random.choice(words)
            sim = cosine_word_similarity(maps[word1], maps[word2])
            if (antilarge):
                sim = sim / ((maps[word1][2] + maps[word2][2]))    
            if (sim > best):
                best = sim
                bw2 = word2        
                if (not smoothing) and (sim > 0.9999): #already found perfect pair
                    return word1, bw2, best            
    return  word1, bw2, best
            
        
def find_fast_similar(maps, tests=10000, antilarge=True, smoothing = True):
    best = -1
    bw1 = None
    bw2 = None
    words = list(maps)
    wordcount = len(words)
    if (wordcount == 1):
        return "FINISH"
    if(wordcount <= math.sqrt(2*tests)):
        bw1, bw2, best = find_most_similar(maps, False, antilarge, smoothing)
    else:
        for i in range(tests):
            word1 = ""
            word2 = ""
            while word2 == word1:
                word1 = random.choice(words)
                word2 = random.choice(words)
            sim = cosine_word_similarity(maps[word1], maps[word2])
            if (smoothing):
                sim = sim + 0.1
            if (antilarge):
                sim = sim / ((maps[word1][2] + maps[word2][2]))
            if (sim > best):
                best = sim
                bw1 = word1
                bw2 = word2   
                if (not smoothing) and (sim > 0.9999): #already found perfect pair
                    return bw1, bw2, best                
    return bw1, bw2, best

    
def join_wordinfos(word1info, word2info):
    prevbag = join_bag(word1info[0], word2info[0], word1info[2], word2info[2])
    nextbag = join_bag(word1info[1], word2info[1], word1info[2], word2info[2])
    count = word1info[2] + word2info[2]
    return [prevbag, nextbag, count]
    
def join_bag(bag1, bag2, freq1, freq2):
    bag = {}
    allwords = list(set(bag1) | set(bag2))
    for word in allwords:
        try:
            b1 = bag1[word]
        except:
            b1 = 0
        try:
            b2 = bag2[word]
        except:
            b2 = 0    
        bag[word] = ((b1*freq1) + (b2*freq2))/(freq1 + freq2)
    return bag
        
def replace_all_instances_in_maps(maps, original, new):
    for word in list(maps):
        wordinfo = maps[word]
        prevs = wordinfo[0]
        nexts = wordinfo[1]
        count = wordinfo[2]
        if original in list(prevs):
            freq = prevs[original]
            if new in list(prevs):
                prevs[new] += freq
            else:
                prevs[new] = freq
            del prevs[original]
        if original in list(nexts):
            freq = nexts[original]
            if new in list(nexts):
                nexts[new] += freq
            else:
                nexts[new] = freq
            del nexts[original]        
        
def cluster(maps, classes, tests=10000, order=True, antilarge=False, smoothing=False):
    words = list(maps)
    classcount = len(words)    
    dictionary = {}
    newclass = 0
    for word in words:
        dictionary[word] = [word]
    
    while (classcount > classes):
        if order:
            word1, word2, score = find_fast_order_similar(maps, tests, antilarge)
        else:
            word1, word2, score = find_fast_similar(maps, tests, antilarge, smoothing)
        classname = "CLASS" + str(newclass)
        newclass += 1
        classcount -= 1
        #print(word1, word2)
        print(word1, dictionary[word1])
        print(word2, dictionary[word2])
        print(word1, "+", word2, "=", classname, "     ( score", score,")     ( currently ", classcount, "classes )")
        
        wordcontent = list(set(dictionary[word1]) | set(dictionary[word2]))
        del dictionary[word1]
        del dictionary[word2]
        dictionary[classname] = wordcontent
        
        newwordinfo = join_wordinfos(maps[word1], maps[word2])
        del maps[word1]
        del maps[word2]
        maps[classname] = newwordinfo
        replace_all_instances_in_maps(maps, word1, classname)
        replace_all_instances_in_maps(maps, word2, classname)
        
        print(classname, dictionary[classname])
        print()
        
    return maps, dictionary

def read_dictresults(filename):
    f=open(filename, "r")
    t = f.read()
    f.close()
    d = eval(t)
    for c in list(d):
        print(d[c])
        print(len(d[c]))
        print(c)
        print() 
        
def nice_results(dictionary, originalmap, newmap, tokenmap=None):
    result = ""
    for wordclass in list(dictionary):
        words = dictionary[wordclass]
        result = result + "\n\n"
        result += wordclass + " contains " + str(len(words)) + " words and occurs a total of " +str(newmap[wordclass][2]) + " times"
        if (tokenmap != None):
            classes = {}
            total = 0
            for word in words:
                try:
                    pos = tokenmap[word]
                except:
                    pos = "none"
                try:
                    classes[pos] = classes[pos] + originalmap[word][2]
                except:
                    classes[pos] = originalmap[word][2]
                total += originalmap[word][2]
            sortedclasses = sorted(classes, key=lambda c: classes[c], reverse=True) 
            result = result + "\n"
            for pos in sortedclasses:
                result = result + pos + "=" + str((classes[pos]/total)*100)[:5] + "%, "
        sortedwords = sorted(words, key=lambda w: originalmap[w][2], reverse=True) 
        quantity = min(len(sortedwords), 6*10)
        count = 0
        linecount = 10
        while count<quantity:
            if linecount >= 10:
                result = result + "\n    "
                linecount = 0
            result = result + sortedwords[count] + ", "
            count += 1
            linecount += 1
    return result

def make_token_map(textstring):
    tokens = nltk.word_tokenize(textstring)
    classifications = nltk.pos_tag(tokens)
    tokenmap = {}
    for pair in classifications:
        word = pair[0].lower()
        try:
            wordcounts = tokenmap[word]
        except:
            wordcounts = {}
            tokenmap[word] = wordcounts
        classification = pair[1]
        try:
            wordcounts[classification] = wordcounts[classification] + 1
        except:
            wordcounts[classification] = 1
    besttokens = {}
    for word in list(tokenmap):
        besttokens[word] = max(list(tokenmap[word]), key=lambda c: tokenmap[word][c])
    return besttokens

def make_tokenized_file(tokenmap, dictionary, originalmap, csvfilename):
    tokens = []
    for word in list(tokenmap):
        token = tokenmap[word]
        if not token in tokens:
            tokens += [token]
    table = {}
    for token in tokens:
        table[token] = []
    wordclasses = list(dictionary)
    granularfocus = {}
    for wordclass in wordclasses:
        words = dictionary[wordclass]
        classes = {}
        total = 0
        for word in words:
            try:
                pos = tokenmap[word]
            except:
                pos = "none"
            try:
                classes[pos] = classes[pos] + originalmap[word][2]
            except:
                classes[pos] = originalmap[word][2]
            total += originalmap[word][2]        
        for clas in list(classes):
            classes[clas] = (classes[clas]/total)*100
        granularfocus[wordclass] = focus_metric([classes[clas] for clas in list(classes)])*100
        for token in tokens:
            if token in list(classes):
                table[token] += [classes[token]]
            else:
                table[token] += [0]
    tokens = sorted(tokens)
    result = [["Class"]+wordclasses]
    for token in tokens:
        if max(table[token]) > 0:
            result = result + [[token]+table[token]]
            
    result += [[]] #empty line
    result += [["Focus"]+[granularfocus[clas] for clas in wordclasses]]    
    result += [[]] #empty line
    
    compacts = {}
    for miniclass in ["Conjunction", "Preposition", "Determiner", "GrammarOther", "Adjective", "Noun", "Pronoun", "Adverb", "Verb", "Other"]:
        compacts[miniclass] = [0] * len(wordclasses)
    for token in tokens:
        if token in ["CC"]:
            joinwith = "Conjunction"
        elif token in ["IN", "TO"]:
            joinwith = "Preposition"
        elif token in ["DT", "WDT"]:
            joinwith = "Determiner"
        elif token in ["EX", "MD", "PDT", "POS", "RP"]:
            joinwith = "GrammarOther"
        elif token in ["JJ", "JJR", "JJS"]:
            joinwith = "Adjective"
        elif token in ["NN", "NNS", "NNP", "NNPS"]:
            joinwith = "Noun"
        elif token in ["PRP", "PRP$", "WP", "WP$"]:
            joinwith = "Pronoun"
        elif token in ["RB", "RBR", "RBS", "WRB"]:
            joinwith = "Adverb"
        elif token in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
            joinwith = "Verb"
        elif token in ["CD", "FW", "LS", "UH", "none"]:
            joinwith = "Other"
        else:
            print ("Unknown token:", token)
            joinwith = "Other"
        for i in range(len(table[token])):
            compacts[joinwith][i] += table[token][i]
    miniclasses = sorted(list(compacts))
    for miniclass in miniclasses:
        if max(compacts[miniclass]) > 0:
            result = result + [[miniclass]+compacts[miniclass]]
    
    reorder = {}  
    for wordclass in wordclasses:
        reorder[wordclass] = []
    for miniclass in miniclasses:
        if max(compacts[miniclass]) > 0:
            for wordclass in wordclasses:
                reorder[wordclass] += [compacts[miniclass][wordclasses.index(wordclass)]]
    compactfocus = {}
    for wordclass in wordclasses:
        compactfocus[wordclass] = focus_metric(reorder[wordclass])*100
                
    result += [[]] #empty line
    result += [["Focus"]+[compactfocus[clas] for clas in wordclasses]]    
    result += [[]] #empty line
                
    f = open(csvfilename, 'w', newline='')
    with f:
        writer = csv.writer(f)
        for row in result:
            writer.writerow(row)    
    f.close()
    
    granfocs = [granularfocus[clas] for clas in wordclasses]
    compfocs = [compactfocus[clas] for clas in wordclasses]
    
    return sum(granfocs)/len(granfocs) , sum(compfocs)/len(compactfocus)
    
def focus_metric(values):
    a = 0
    total = 0
    b = 1/len(values)
    for v in values:
        a += v**2
        total += v
    try:
        return math.sqrt(((a/(total**2)) - b)/(1 - b))
    except:
        return 0

def main(filename, tests, classes, order, antilarge, smoothing, tokenize):
    textstring = get_text(filename)
    maps = generate_bigram_maps(vparser.text_to_word_lists(textstring))
    originalmaps = copy.deepcopy(maps)
    outfilename = "../outputs/" + (filename.split("/")[-1]).split(".")[0] + "_" + str(tests) + "_" + str(classes)
    if order:
        outfilename += "_order"
    if antilarge:
        outfilename += "_antilarge"
    if smoothing:
        outfilename += "_smoothing"
    if tokenize:
        outfilename += "_tokenize"
    csvfilename = outfilename + ".csv"
    outfilename += ".txt"
    print("Preparing",outfilename) 
    maps, dictionary = cluster(maps, classes, tests, order, antilarge, smoothing)
    if tokenize:
        print("Tokenizing...")
        tokenmap = make_token_map(textstring)
        granfocs, compfocs = make_tokenized_file(tokenmap, dictionary, originalmaps, csvfilename)
    else:
        tokenmap = None
    results = nice_results(dictionary, originalmaps, maps, tokenmap)
    print(results)
    f = open(outfilename, "w", encoding="utf-8")
    f.write(results + "\n\n\n" + str(dictionary) + "\n\n\n" + str(maps))
    f.close()
    print("exported to,",outfilename)
    if tokenize:
        return granfocs, compfocs    

if __name__ == "__main__":
    filename = input("Please input source text filename: ")
    tests = eval(input("Please input how many samples per join: "))
    classes = eval(input("Please input the number of final classes: "))
    order = eval(input("Use ordered collapsing? True, False: "))
    antilarge = eval(input("Use antilarge? True, False: "))
    if order:
        smoothing = False
    else:
        smoothing = eval(input("Use smooting? True, False: "))
    tokenize = eval(input("Verify with tokenizer? True, False: "))
    print(main(filename, tests, classes, order, antilarge, smoothing, tokenize))
    
def graphs():    
    filename = "../corpus/english_alice.txt"
    classes = 10
    antilarge = False
    order = True
    smoothing = False
    tokenize = True
    x1 = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    y11 = []
    y12 = []
    for tests in x1:
        r1, r2 = main(filename, tests, classes, order, antilarge, smoothing, tokenize)
        y11 += [r1]
        y12 += [r2]
        
    tests = 5000
    x2 = [100, 70, 50, 40, 30, 25, 20, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    y21 = []
    y22 = []
    for classes in x2:
        r1, r2 = main(filename, tests, classes, order, antilarge, smoothing, tokenize)
        y21 += [r1]
        y22 += [r2]    
        
    plt.subplot(2, 1, 1)
    plt.plot([math.log(x, 10) for x in x1], y11)
    plt.plot([math.log(x, 10) for x in x1], y12)
    plt.xlabel('Samples (log10)')
    plt.ylabel('Focus')
    
    plt.subplot(2, 1, 2)
    plt.plot(x2, y21)
    plt.plot(x2, y22)
    plt.xlabel('Classes')
    plt.ylabel('Focus')
    
    plt.show()

    
#maps, dictionary = cluster(generate_bigram_maps(vparser.text_to_word_lists(get_text(en_bible))), 20, 1000)
#print(dictionary)