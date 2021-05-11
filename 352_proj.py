import sys
import numpy as np
import torch
import torch.nn as nn
import csv

from torch.nn.modules.activation import LogSoftmax

data = []
tokens = {}
vocab = {} #top 5000 tokens
unigrams = {}
bigrams ={}
trigrams = {}


#sys.stdout = open('a3_Jiang_111736065_OUTPUT.txt', 'w')f
filters = ['.',',', "`", '-', '``','(', ':', ')', '--', "'", "''"]
reducers = ['machine', 'machines', 'language', 'languages', ]
def getData(file):
    global data
    global tokens
    global vocab
    f = open(file)
    f = csv.reader(f, delimiter = '\t')
    for row in f:
        context = row[2].split(' ')
        data.append('<s>')
        if '<s>' not in tokens:
            tokens['<s>'] = 1
        else:
            tokens['<s>'] += 1
        for i in context:
            word = i.split('/')[0].lower()
            if '<head>' in word:
                word = word.replace('<head>', '')
            data.append(word)
            if word not in filters:
                if word not in tokens.keys():
                    tokens[word] = 1
                else:
                    tokens[word] += 1
        data.append('</s>')
        if '</s>' not in tokens:
            tokens['</s>'] = 1
        else:
            tokens['</s>'] += 1

    vocab = dict(sorted(tokens.items(), key=lambda x: x[1], reverse = True)[:5000])


def writedata():
    
    fd = open('parsed_data.txt', 'w+')
    fd.write((' '.join(data)).replace('</s>', '\n'))
    fd.close()
    og = sys.stdout
    sys.stdout = open('token_counts.txt', 'w+')
    print(tokens)
    sys.stdout = open("vocab.txt", 'w+')
    print(vocab)
    sys.stdout= og


def getUnigram(data):
    global unigrams
    global vocab
    for word in data:
        if word in vocab.keys():
            if word not in unigrams:
                unigrams[word] = 1
            else:
                unigrams[word] += 1
        else:
            if '<OOV>' not in unigrams.keys():
                unigrams['<OOV>'] = 1
            else:
                unigrams['<OOV>'] += 1

def getBigrams(data):
    global vocab 
    global bigrams
    for i in range(len(data) - 1):
        currentword = data[i]
        nextword = data[i+1]
        if currentword in vocab.keys():
            if nextword in vocab.keys():
                if currentword not in bigrams.keys():
                    bigrams[currentword] = {}
                    bigrams[currentword][nextword] = 1
                else:
                    if nextword not in bigrams[currentword].keys():
                        bigrams[currentword][nextword] = 1
                    else:
                        bigrams[currentword][nextword] += 1
            else:
                if currentword not in bigrams.keys():
                    bigrams[currentword] = {}
                    bigrams[currentword]['<OOV>'] =1
                else:
                    if '<OOV>' not in bigrams[currentword].keys():
                        bigrams[currentword]['<OOV>'] = 1
                    else:
                        bigrams[currentword]['<OOV>'] += 1
        else:
            if '<OOV>' not in bigrams.keys():
                bigrams['<OOV>'] = {}
                if nextword in vocab.keys():
                    bigrams['<OOV>'][nextword] = 1
                else:
                    bigrams['<OOV>']['<OOV>'] = 1
            else:
                if nextword in vocab.keys():
                    if nextword not in bigrams['<OOV>'].keys():
                        bigrams['<OOV>'][nextword] = 1
                    else:
                        bigrams['<OOV>'][nextword] += 1
                else:
                    if '<OOV>' not in bigrams['<OOV>'].keys():
                        bigrams['<OOV>']['<OOV>'] = 1
                    else:
                        bigrams['<OOV>']['<OOV>'] += 1

def getTrigrams(data):
    global bigrams
    global trigrams
    global vocab
    for i in bigrams.keys():
        trigrams[i] = {}
        for j in bigrams[i].keys():
            trigrams[i][j] = {}
    for i in range(len(data) - 2):
        currentword = data[i]
        nextword = data[i+1]
        thirdword = data[i+2]
        if currentword == '<s>' or currentword == '</s>' or nextword == '<s>' or nextword == '</s>' or thirdword == '<s>' or thirdword == '</s>': 
            continue
        if currentword not in vocab.keys():
            currentword = '<OOV>'
        if nextword not in vocab.keys():
            nextword = '<OOV>'
        if thirdword in vocab.keys():
            if thirdword not in trigrams[currentword][nextword].keys():
                trigrams[currentword][nextword][thirdword] = 1
            else:
                trigrams[currentword][nextword][thirdword] += 1
        else:
            if '<OOV>' not in trigrams[currentword][nextword].keys():
                trigrams[currentword][nextword]['<OOV>'] = 1
            else:
                trigrams[currentword][nextword]['<OOV>'] += 1
   # print(trigrams['specific']['formal']['languages'])
    #print(trigrams['to']['process'])
    #print(trigrams['specific']['formal'])


def addOne(w1, w2 = None):
    global bigrams
    global unigrams
    global trigrams
    output = {}
    poss = bigrams[w1]
  # print(poss)
    v = len(vocab.keys())
    output = {}
    for word in vocab.keys():
        if word not in poss.keys():
            poss[word] = 0
    for word in poss.keys():
        if word in reducers:
            poss[word] *= 0.1
        output[word] = (poss[word] + 1) / (unigrams[w1] + v)
    if w2 != None:
        out2 = {}
        for word in vocab.keys():
            if word not in trigrams[w2].keys():
                trigrams[w2][word] = {}                
        x = trigrams[w2][w1]
        if len(x) == 0:
            return output
        for word in x:
            if word in reducers:
                x[word]*= 0.1
          #  print(word)
            if word in output:
                bi = output[word]
          #  print(bi)
            tri = (x[word] + 1)/ (bigrams[w2][w1] + v)
           # print(tri)
            out2[word] = (bi + tri)/2
        return out2
    return output

def generateLang(tokenList, length):
    if tokenList[-1] == '<s>' and len(tokenList) == 1:
        out = addOne('<s>')
    else:
        if tokenList[-1] == '</s>':
            return " ".join(tokenList)
        out = addOne(tokenList[-1], tokenList[-2])
        
    output = tokenList
    counter = 0
    
    top = dict(sorted(out.items(), key=lambda x: x[1], reverse=True)[:10])
    rand = np.random.randint(len(top.keys()))
    token = list(top.keys())[rand]
    if token not in ['<s>', '</s>', '<OOV>']:
        output.append(token)
    

    while counter < length:
        out = addOne(output[-1])


  

        top = dict(sorted(out.items(), key=lambda x: x[1], reverse=True)[:10])
        if '<OOV>' in top:
            top.pop('<OOV>')
        if '<s>' in top:
            top.pop('<s>')
        if '</s>' in top:
            top.pop('</s>')


        rand = np.random.randint(len(top.keys()))
        token = list(top.keys())[rand]
        
        if token == '<s>' or token == '<OOV>' or token == '</s>':
            continue
        #print(f'token: {token} with chosen with probability: {out[token]}')
        output.append(token)
        counter += 1
    output.append('<end>')

    return " ".join(output)
if __name__ == "__main__":
    trainingData = sys.argv[1]
    getData(trainingData)
    writedata()
    #sys.exit()
    getUnigram(data)
    getBigrams(data)
    getTrigrams(data)
  
    print('Choose a keyword to generate a random sentence:')
    query = input('neuropsychology | economics | culture | engineering | information | greek | machine | manufacturing | technology:\n')
    size = input('Input a sentence length: ')
    query = ['<s>', query]
    res = generateLang(query, int(size)); 

    print(res)


