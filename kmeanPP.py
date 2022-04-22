import copy
import numpy as np

class KmeanPP():
    def __init__(self,seeds,tweets,sparse,max_itr = 10):
        self.seeds = seeds #inital centroids
        self.tweets = tweets #docs as set of words
        self.sparse = sparse #sparse of docs
        self.max_itr = max_itr
        self.K = len(seeds) #K value
        self.doc_count = len(self.tweets)  #number of documents
        self.vocab_count = len(self.sparse[1])-1  #total number of different words
        
        self.clusters = {} # cluster to docID
        self.rev_clusters = {} # reverse index, docID to cluster
        self.centroids = {} #centroids
        
        
    def jaccard(self,setA,setB):
        ''' Takes
        setA : set of words in doc A
        setB: set of words in doc B
        
        return: jaccard index as well as jaccard distance (= 1-jaccard index)'''
        
        index = float(len(setA.intersection(setB)))/float(len(setA.union(setB)))
        return index, 1-index
        
        
    def updateclusterize(self):
        ''' Updating clusters using latest updated centroids(Seeds) '''
        
        clusters = {} #to store new clusters, cluster to docID
        rev_clusters = {} #to store new reverse index, docID to cluster
        
        for i in range(1,self.doc_count+1):
            rev_clusters[i] = -1
        for i in range(self.K):
            clusters[i] = set([self.seeds[i]])
            rev_clusters[self.seeds[i]] = i
            
        for i in range(1,self.doc_count+1):
            if i in self.seeds:
                continue
            new_cluster = -1
            ind = -1
            for j in range(self.K):  #finding centroid in current centroids(seeds) having max jaccard index with docID = i
                index, dist = self.jaccard(self.tweets[i],self.tweets[self.seeds[j]])
                if index > ind:
                    ind = index
                    new_cluster = j
            clusters[new_cluster].add(i)
            rev_clusters[i] = new_cluster
        return clusters,rev_clusters
            
    def meandoc(self,j,mean_set):
        '''Takes
        jth cluster and mean set of jth cluster
        
        return: docID having max jaccard index with mean set'''
        
        min_ind = -1
        new_cent = -1
        for i in self.clusters[j]:
            index, dist = self.jaccard(mean_set, self.tweets[i])
            if index > min_ind:
                min_ind = index
                new_cent = i
        return new_cent
        
        
    def calcNewCentroids(self):
        ''' calculating new centroids using mean method
        '''
        new_centroids = []
        for j in range(self.K):
            mean = [0]*(len(self.sparse[1]))
            count = 0
            for m in self.clusters[j]:    
                mean = [mean[k] + self.sparse[m][k] for k in range(len(self.sparse[1]))] #calculating frequency of words in a cluster
                count += 1
            mean = [1 if (i/count)>0.5 else 0 for i in mean] 
            mean_set = set()
            for i in range(1,len(mean)):
                if mean[i]==1:
                    mean_set.add(i)
            meandoc = self.meandoc(j,mean_set)
            new_centroids.append(meandoc)
        self.seeds = copy.deepcopy(new_centroids) #deepcopy is used to copy nested dictionary
        return new_centroids
    
    
    def fit(self):
        seed = copy.deepcopy(self.seeds)
        for i in range(self.max_itr):
            clusters,rev_clusters = self.updateclusterize()
            self.clusters = copy.deepcopy(clusters)
            self.rev_clusters = copy.deepcopy(rev_clusters)
            new_seed = self.calcNewCentroids()
            if(set(seed) == set(new_seed)):
                break
            seed = new_seed
        return seed
    
    
    def get_clusters(self):
        return self.clusters
    
    
    def get_revclusters(self):
        return self.rev_clusters
    
    
    def get_centroids(self):
        return self.seeds
    
    
    def get_totalerror(self):
        ''' calculating total distance of each doc with corresponding centroids '''
        
        dist = 0
        centroids = self.get_centroids()
        for k in self.clusters:
            ID = centroids[k]
            for ID2 in self.clusters[k]:
                bag1 = set(self.tweets[ID])
                bag2 = set(self.tweets[ID2])
                dist += self.jaccard(bag1, bag2)[0]
        return dist