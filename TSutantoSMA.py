# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 11:25:43 2018
MIT License with Acknowledgement
@author: Taufik Sutanto
Simple Social Media Analytics ver 0.11.1
https://taufiksutanto.blogspot.com/2018/01/easiest-social-media-analytics.html
"""
from pattern.web import Twitter, URL
from nltk.tokenize import TweetTokenizer; Tokenizer = TweetTokenizer(reduce_len=True)
from tqdm import tqdm
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from bs4 import BeautifulSoup as bs
from sklearn.decomposition import LatentDirichletAllocation as LDA
import re, networkx as nx, matplotlib.pyplot as plt, operator, numpy as np,community

def crawl(topic, N=100, Nbatch = 25):
    t = Twitter() # language='en','id'
    M = N//Nbatch #integer
    i, Tweets, keepCrawling = None, [], True
    for j in tqdm(range(M)):
        if keepCrawling:
            for tweet in t.search(topic, start=i, count=Nbatch):
                try:
                    Tweets.append(tweet)
                    i = tweet.id
                except:
                    print("Twitter Limit reached")
                    keepCrawling = False # Second Break (outer loop)
                    break
        else:
            break
    print('Making sure we get the full tweets, please wait ...')
    for i, tweet in enumerate(tqdm(Tweets)):
        try:
            webPage = URL(tweet.url).download()
            soup = bs(webPage,'html.parser')
            full_tweet = soup.find_all('p',class_='TweetTextSize')[0] #modify this to get all replies
            full_tweet = bs(str(full_tweet),'html.parser').text
            Tweets[i]['fullTxt'] = full_tweet
        except:
            Tweets[i]['fullTxt'] = tweet.txt
    print('Done!... Total terdapat {0} tweet'.format(len(Tweets)))
    return Tweets        

def strip_non_ascii(string,symbols):
    ''' Returns the string without non ASCII characters''' #isascii = lambda s: len(s) == len(s.encode())
    stripped = (c for c in string if 0 < ord(c) < 127 and c not in symbols)
    return ''.join(stripped)

def cleanTweets(Tweets):
    factory = StopWordRemoverFactory(); stopwords = set(factory.get_stop_words()+['rt','pic','com','yg','ga'])
    factory = StemmerFactory(); stemmer = factory.create_stemmer()
    for i,tweet in enumerate(tqdm(Tweets)):
        txt = tweet['fullTxt'] # if you want to ignore retweets  ==> if not re.match(r'^RT.*', txt):
        txt = txt.lower() # Lowercase
        txt = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',txt)# clean urls
        txt = Tokenizer.tokenize(txt)
        symbols = set(['@']) # Add more if you want
        txt = [strip_non_ascii(t,symbols) for t in txt] #remove all non ASCII characters
        txt = ' '.join([t for t in txt if len(t)>1])
        Tweets[i]['cleanTxt'] = txt # this is not a good Python practice, only for learning.
        txt = stemmer.stem(txt).split()
        Tweets[i]['nlp'] = ' '.join([t for t in txt if t not in stopwords])    
    return Tweets

def translate(txt,language='en'): # txt is a TextBlob object
    try:
        return txt.translate(to=language)
    except:
        return txt

def sentiment(Tweets): #need a clean tweets
    print("Calculating Sentiment and Subjectivity Score: ... ")
    T = [translate(TextBlob(tweet['cleanTxt'])) for tweet in tqdm(Tweets)]
    Sen = [tweet.sentiment.polarity for tweet in tqdm(T)]
    Sub = [float(tweet.sentiment.subjectivity) for tweet in tqdm(T)]
    Se, Su = [], []
    for score_se, score_su in zip(Sen,Sub):
        if score_se>0.1:
            Se.append('pos')
        elif score_se<-0.05: #I prefer this
            Se.append('neg')
        else:
            Se.append('net')
        if score_su>0.5:
            Su.append('Subjektif')
        else:
            Su.append('Objektif')
    label_se = ['Positif','Negatif', 'Netral']
    score_se = [len([True for t in Se if t=='pos']),len([True for t in Se if t=='neg']),len([True for t in Se if t=='net'])]    
    label_su = ['Subjektif','Objektif']
    score_su = [len([True for t in Su if t=='Subjektif']),len([True for t in Su if t=='Objektif'])]
    PieChart(score_se,label_se); PieChart(score_su,label_su)
    Sen = [(s,t['fullTxt']) for s,t in zip(Sen,Tweets)]
    Sen.sort(key=lambda tup: tup[0])
    Sub = [(s,t['fullTxt']) for s,t in zip(Sub,Tweets)]
    Sub.sort(key=lambda tup: tup[0])
    return (Sen, Sub)

def printSA(SA, N = 2, emo = 'positif'):
    Sen, Sub = SA
    e = emo.lower().strip()
    if e=='positif' or e=='positive':
        tweets = Sen[-N:]
    elif e=='negatif' or e=='negative':
        tweets = Sen[:N]
    elif e=='netral' or e=='neutral':
        net = [(abs(score),t) for score,t in Sen if abs(score)<0.01]
        net.sort(key=lambda tup: tup[0])
        tweets = net[:N]
    elif e=='subjektif' or e=='subjective':
        tweets = Sub[-N:]
    elif e=='objektif' or e=='objective':
        tweets = Sub[:N]
    else:
        print('Wrong function input parameter = "{0}"'.format(emo)); tweets=[]
    print('"{0}" Tweets = '.format(emo))
    for t in tweets:
        print(t)

def wordClouds(Tweets):
    txt = [t['nlp'] for t in Tweets]; txt = ' '.join(txt)
    wc = WordCloud(background_color="white")
    wordcloud = wc.generate(txt)
    plt.figure(num=1, facecolor='w', edgecolor='k')
    plt.imshow(wordcloud, cmap=plt.cm.jet, interpolation='nearest', aspect='auto'); plt.xticks(()); plt.yticks(())
    plt.show()

def PieChart(score,labels):
    fig1 = plt.figure(); fig1.add_subplot(111)
    plt.pie(score, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal');plt.show()
    return None

def drawGraph(G, Label = False):
    fig3 = plt.figure(); fig3.add_subplot(111)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G,pos, alpha=0.2,node_color='blue',node_size=600)
    if Label:
        nx.draw_networkx_labels(G,pos)
    nx.draw_networkx_edges(G,pos,width=4); plt.show()

def Graph(Tweets, Label = True): # Need the Tweets Before cleaning
    print("Please wait, building Graph .... ")
    G=nx.Graph()
    for tweet in tqdm(Tweets):
        G.add_node(tweet.author)
        mentionS =  re.findall("@([a-zA-Z0-9]{1,15})", tweet['fullTxt'])
        for mention in mentionS:
            if "." not in mention: #skipping emails
                usr = mention.replace("@",'').strip()
                G.add_node(usr); G.add_edge(tweet.author,usr)
    Nn=G.number_of_nodes();Ne=G.number_of_edges()
    print('Finished. There are %d nodes and %d edges in the Graph.' %(Nn,Ne))
    if Label:
        drawGraph(G, Label = True)
    else:
        drawGraph(G)
    return G

def Centrality(G, N=10):
    phi = 1.618033988749895 # largest eigenvalue of adj matrix
    ranking = nx.katz_centrality_numpy(G,1/phi)
    important_nodes = sorted(ranking.items(), key=operator.itemgetter(1))[::-1]#[0:Nimportant]
    Mstd = 1 # 1 standard Deviation CI
    data = np.array([n[1] for n in important_nodes])
    out = len(data[abs(data - np.mean(data)) > Mstd * np.std(data)]) # outlier within m stDev interval
    if out>N:
        dnodes = [n[0] for n in important_nodes[:N]]
        print('Influencial Users: {0}'.format(str(dnodes)))
    else:
        dnodes = [n[0] for n in important_nodes[:out]]
        print('Influencial Users: {0}'.format(str(important_nodes[:out]))) 
    Gt = G.subgraph(dnodes)
    drawGraph(Gt, Label = True)
    return Gt

def Community(G):
    part = community.best_partition(G)
    values = [part.get(node) for node in G.nodes()]
    mod, k = community.modularity(part,G), len(set(part.values()))
    print("Number of Communities = %d\nNetwork modularity = %.2f" %(k,mod)) # https://en.wikipedia.org/wiki/Modularity_%28networks%29
    fig2 = plt.figure(); fig2.add_subplot(111)
    nx.draw_shell(G, cmap = plt.get_cmap('gist_ncar'), node_color = values, node_size=30, with_labels=False)
    plt.show
    return values

def print_Topics(model, feature_names, Top_Topics, n_top_words):
    for topic_idx, topic in enumerate(model.components_[:Top_Topics]):
        print("Topic #%d:" %(topic_idx+1))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

def getTopics(Tweets,n_topics=5, Top_Words=7):
    Txt = [t['nlp'] for t in Tweets] # cleaned: stopwords, stemming
    tf_vectorizer = CountVectorizer(strip_accents = 'unicode', token_pattern = r'\b[a-zA-Z]{3,}\b', max_df = 0.95, min_df = 2)
    dtm_tf = tf_vectorizer.fit_transform(Txt)
    tf_terms = tf_vectorizer.get_feature_names()
    lda_tf = LDA(n_components=n_topics, learning_method='online', random_state=0).fit(dtm_tf)   
    vsm_topics = lda_tf.transform(dtm_tf); doc_topic =  [a.argmax()+1 for a in tqdm(vsm_topics)] # topic of docs
    print('In total there are {0} major topics, distributed as follows'.format(len(set(doc_topic))))
    fig4 = plt.figure(); fig4.add_subplot(111)
    plt.hist(np.array(doc_topic), alpha=0.5); plt.show()
    print('Printing top {0} Topics, with top {1} Words:'.format(n_topics, Top_Words))
    print_Topics(lda_tf, tf_terms, n_topics, Top_Words)
    return lda_tf, dtm_tf, tf_vectorizer
    