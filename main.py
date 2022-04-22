from kmeanP import *
from kmeanPP import *

with open("docword.nips.txt", "r") as f:
    doc_count = f.readline()
    diff_words = f.readline()
    word_count = f.readline()
    tweets = [[x+1] for x in range(int(doc_count))]
    for line in f:
        file1 = re.split(r' ', str(line[:-1]))
        tweets[int(file1[0])-1].append(file1[1])
f.close()

seeds = sorted(rand(10,int(doc_count)))
Kmean = kmeansP(seeds,tweets)
Kmean.converge()
print("nips: for K =", 10,"cost is",Kmean.get_totalDis())
nip_clusters = Kmean.get_clusters()

with open("docword.kos.txt", "r") as f:
    doc_count = f.readline()
    diff_words = f.readline()
    word_count = f.readline()
    tweets = [[x+1] for x in range(int(doc_count))]
    for line in f:
        file1 = re.split(r' ', str(line[:-1]))
        tweets[int(file1[0])-1].append(file1[1])
f.close()

seeds = sorted(rand(12,int(doc_count)))
Kmean = kmeansP(seeds,tweets)
Kmean.converge()
print("kos: for K =", 12,"cost is",Kmean.get_totalDis())
kos_clusters = Kmean.get_clusters()



file = open("docword.enron.txt", 'r')
doc_count = int(file.readline().strip())
vocab_count = int(file.readline().strip())
entries = int(file.readline().strip())
lines = file.readlines()
sparse = {}
for i in range(1,doc_count+1):
    sparse[i] = [0]*(vocab_count+1)
tweets={}
i = 0
for line in lines:
    a = list(map(int, line.strip().split()))
    i+=1
    if len(a)>1:
        doc_id, word_id, freq = a
        if doc_id not in tweets.keys():
            tweets[doc_id] = set([word_id])
        else:
            tweets[doc_id].add(word_id)
        sparse[doc_id][word_id]=1
file.close()

Kmean = KmeanPP(list(np.random.choice(range(1,doc_count+1), 15, replace=False)),tweets,sparse)
Kmean.fit()
print("enron: for K =", 12,"cost is",Kmean.get_totalDis())

enron_cluster = Kmean.get_totalerror()