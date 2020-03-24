takashi = "../transcriptions/takahashi.txt"

ignore = "\n1234567890()[],-_#$%&/={}'|/\\" + '"' + chr(8220) + chr(8221) + chr(8212) + chr(191) + chr(161) 
breaks = "\t.!?:;" + chr(1567)

class Page:
    #identifier (id string)
    #lines (list of Line entities)
    #content (text content)
    #raw (raw content)
    #I = illustration type  (T,H,A,Z,B,C,P,S) Text, Herbal, Astronomical, Zodiac, Biological, Cosmological, Pharmaceutical or Stars.
    #Q = Quire              (A-T)
    #P = page in quire      (A-X)
    #L = Currier's language (A,B)
    #H = Currier's hand     (1,2,3,4,5,X,Y)
    
    def __init__(self, pagelines):
        self.raw = pagelines
        header = pagelines[0]
        pagelines = pagelines[1:]
        self.identifier = header.split(">")[0][1:]
        
        self.I = header[header.index("$I=")+3] 
        self.Q = header[header.index("$Q=")+3] 
        self.P = header[header.index("$P=")+3] 
        try:
            self.L = header[header.index("$L=")+3] 
        except:
            self.L = "unknown"
        try:
            self.H = header[header.index("$H=")+3] 
        except:
            self.H = "unknown"
        
        self.lines = []
        self.content = ""
        
        for l in pagelines:
            line = Line(l)
            self.lines = self.lines + [line]
            self.content = self.content + line.content + "\n"
        
class Line:
    #identifier (id string)
    #info (info code string)
    #has_interruption (<-> in raw content)
    #has_paragraph_break (<$> in raw content)
    #raw (raw content string)
    #content (cleaned content)
    def __init__(self, line):
        self.raw = line
        header = line.split(">")[0][1:]
        #print(header)
        header = header.split(",")
        self.identifier = header[0]
        self.info = header[1]
        header = line.split(">")[0]
        line = line[len(header)+1:]
        if "<->" in line:
            self.has_interruption = True
        else:
            self.has_interruption = False
        if "<$>" in line:
            self.has_paragraph_break = True
        else:
            self.has_paragraph_break = False        
        self.content = line.replace(" ", "").replace("\t", "").replace("\n", "").replace(".", " ").replace("<->", " ").replace("<$>", " ")
        if "#" in self.content:
            print(self.content)
            print("WARNING: # found in:",self.raw)
        if (">" in self.content) or ("<" in self.content):
            print(self.content)
            print("WARNING: <> found in:",self.raw)        
    
def get_all_pages():        #get list of all Page instances
    f = open(takashi, "r")
    lines = f.readlines()
    f.close()
    pages = []
    thispage = []
    for l in lines:
        if (l[0] == "#") or (len(l)<4) or ("<fRos>" in l):
            continue #ignore comment line
        if ("$Q=" in l) and ("$P=" in l) and ("$F=" in l) and ("$B=" in l): #couple of checks for line header
            if thispage != []: #process last page
                pages = pages + [Page(thispage)]
                thispage = []
            thispage = thispage + [l]
        else:
            thispage = thispage + [l]
    if thispage != []: #process last page
        pages = pages + [Page(thispage)]    
    return pages
        
def get_all_text():        #get complete cleaned book string
    pages = get_all_pages()
    content = ""
    for p in pages:
        content = content + p.content
    #print(content)
    return content

def get_filtered_pages(filter_function):   #get list of Page instances that verify filter_function
    pages = get_all_pages()
    filtered = []
    for p in pages:
        if (filter_function(p)):
            filtered += [p]
    return filtered
    
def get_filtered_text(filter_function):         #get string of content of Page instances that verify filter_function
    pages = get_filtered_pages(filter_function)
    content = ""
    for p in pages:
        content = content + p.content
    #print(content)
    return content    

def get_language_pages(language):            # get list of pages with language (A, B, unknown)
    return get_filtered_pages(lambda x: x.L == language)
    
def get_language_text(language):             # get text of pages with language (A, B, unknown)
    return get_filtered_text(lambda x: x.L == language)

def text_to_word_lists(text):                # get text string, return lists
    text = text.lower()
    for ign in ignore:
        text = text.replace(ign, " ")
    for bre in breaks:
        text = text.replace(bre, ".")
    sentences = text.split(".")
    result = []
    for sent in sentences:
        pre_sentence = sent.split(" ")
        sentence = [word for word in pre_sentence if word != ""]
        if sentence != []:
            result += [sentence]
    return result