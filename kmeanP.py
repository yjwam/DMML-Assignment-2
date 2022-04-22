import copy
import random as np
class kmeansP():
    def __init__(self,seeds,tweets,max_itr = 10):
        self.seeds = seeds
        self.tweets = tweets
        self.max_iterations = max_itr
        self.k = len(seeds)
        
        self.clusters = {} # cluster to tweetID
        self.rev_clusters = {} # reverse index, tweetID to cluster
        self.jaccardMatrix = {} # stores pairwise jaccard distance in a matrix
        self.centroids = {}
        
        self.initializeClusters()
        self.initializeMatrix()
        
    def jaccardDistance(self, setA, setB):
        # Calcualtes the Jaccard Distance of two sets
        try:
            return 1 - float(len(setA.intersection(setB))) / float(len(setA.union(setB)))
        except TypeError:
            print('none')
            
    def initializeMatrix(self):
        for ID1 in range(len(self.tweets)):
            self.jaccardMatrix[ID1] = {}
            bag1 = set(self.tweets[ID1][1:])
            for ID2 in range(len(self.tweets)):
                if ID2 not in self.jaccardMatrix:
                    self.jaccardMatrix[ID2] = {}
                bag2 = set(self.tweets[ID2][1:])
                distance = self.jaccardDistance(bag1, bag2)
                self.jaccardMatrix[ID1][ID2] = distance
                self.jaccardMatrix[ID2][ID1] = distance
                
    def initializeClusters(self):
        # Initialize tweets to no cluster
        for ID in range(len(self.tweets)):
            self.rev_clusters[ID] = -1

        # Initialize clusters with seeds
        for k in range(self.k):
            self.clusters[k] = set([self.seeds[k]])
            self.rev_clusters[self.seeds[k]] = k
            
    def calcNewClusters(self):
        # Initialize new cluster
        new_clusters = {}
        new_rev_cluster = {}
        for k in range(self.k):
            new_clusters[k] = set()

        for ID in range(len(self.tweets)):
            min_dist = float("inf")
            min_cluster = self.rev_clusters[ID]

            # Calculate min average distance to each cluster
            for k in self.clusters:
                dist = 0
                count = 0
                for ID2 in self.clusters[k]:
                    dist += self.jaccardMatrix[ID][ID2]
                    count += 1
                if count > 0:
                    avg_dist = dist/float(count)
                    if min_dist > avg_dist:
                        min_dist = avg_dist
                        min_cluster = k
                        
            new_clusters[min_cluster].add(ID)
            new_rev_cluster[ID] = min_cluster
        return new_clusters, new_rev_cluster

    def converge(self):
        # Initialize previous cluster to compare changes with new clustering
        new_clusters, new_rev_clusters = self.calcNewClusters()
        self.clusters = copy.deepcopy(new_clusters)
        self.rev_clusters = copy.deepcopy(new_rev_clusters)

        # Converges until old and new iterations are the same
        iterations = 1
        while iterations < self.max_iterations:
            new_clusters, new_rev_clusters = self.calcNewClusters()
            iterations += 1
            if self.rev_clusters != new_rev_clusters:
                self.clusters = copy.deepcopy(new_clusters)
                self.rev_clusters = copy.deepcopy(new_rev_clusters)
            else:
                #print iterations
                return None
            
    def get_clusters(self):
        clust = {}
        for k in range(self.k):
            clust[k] = set()
        for k in self.clusters:
            for ID in self.clusters[k]:
                clust[k].add(self.tweets[ID][0])
        return clust
                

    def get_jaccardMatrix(self):
        New = {}
        for ID1 in self.jaccardMatrix:
            New[self.tweets[ID1][0]] = {}
            for ID2 in self.jaccardMatrix[ID1]:
                if self.tweets[ID2][0] not in New:
                    New[self.tweets[ID2][0]] = {}
                dist = self.jaccardMatrix[ID1][ID2]
                New[self.tweets[ID1][0]][self.tweets[ID2][0]] = dist
                New[self.tweets[ID2][0]][self.tweets[ID1][0]] = dist   
                
        return New
    
    def get_centroids(self):
        for k in self.clusters:
            min_dist = float("inf")
            for ID in self.clusters[k]:
                dist = 0
                for ID2 in self.clusters[k]:
                    bag1 = set(self.tweets[ID][1:])
                    bag2 = set(self.tweets[ID2][1:])
                    dist += self.jaccardDistance(bag1, bag2)
                if min_dist > dist:
                    min_dist = dist
                    cent = ID
            self.centroids[k]=cent
        return self.centroids
    
    def get_totalDis(self):
        dist = 0
        centroids = self.get_centroids()
        for k in centroids:
            ID = centroids[k]
            for ID2 in self.clusters[k]:
                bag1 = set(self.tweets[ID][1:])
                bag2 = set(self.tweets[ID2][1:])
                dist += self.jaccardDistance(bag1, bag2)
        return dist