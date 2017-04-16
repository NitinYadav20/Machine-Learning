import os
import random
import math
import operator

def load_data():   #load the documents from the directory
    dirname = "data"
    News_Document = {}
    News_Labels = {}
    for fn in sorted(os.listdir(dirname)):
        if fn.endswith("news_articles.mtx"):
            with open(os.path.join(dirname, fn), 'r') as f:
                News_Document = f.readlines()  #reading the files
        if fn.endswith("news_articles.labels"):
            with open(os.path.join(dirname, fn), 'r') as f:
                News_Labels = f.readlines()
    return News_Document, News_Labels

def termfreq(News_Document,News_Labels):

    width,height = 4882,1840
    documents = [[0 for x in range(width)] for y in range(height)] 
    for x in range(len(News_Document)): #we are telling the fucntion to leave the first two lines of the document
        if(x not in (0,1)):
            count=0
            for y in (News_Document[x].split()):
                if(count==0):
                    doc_id = int(y)
                if(count==1):
                    term = int(y)
                if(count==2):
                    documents[doc_id-1][term-1]= int(y)
                    documents[doc_id-1][-1]=doc_id-1
                count +=1


    width = 1839

    lables=[0 for x in range(width)]
    for x in range(len(News_Labels)):
        count=0
        for y in (News_Labels[x].split(",")):
            if(count==0):
                doc_id= int(y)
            if(count==1):
                lables[doc_id-1]=y.rstrip()
            count+=1

    return documents,lables

def load_dataset(split,documents=[],train_set=[],test_set=[]):  #function to split data set into training set and test set based on split value
    for x in range(len(documents)-1):
        if random.random() < split:
            train_set.append(documents[x])
        else:
            test_set.append(documents[x])
    return train_set, test_set
##http://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
def cosine_distance(instance_one, instance_two, length): #to find similarity between documents
    sumxx,sumxy,sumyy = 0, 0, 0
    for i in range(length):
        x=instance_one[i]; y=instance_two[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return (1- (sumxy/math.sqrt(sumxx*sumyy)))
    
def getNeighbors(train_set, testInstance, k): #returns ks most similar neighbours from training set for a test instance
    distances = []
    length = len(testInstance)-1
    for x in range(len(train_set)):
        dist = cosine_distance(testInstance, train_set[x], length)
        distances.append((train_set[x],x, dist))

    distances.sort(key=operator.itemgetter(2))

    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][1])

        return neighbors
#referenced from :http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
def getResponse(neighbors,labels): #returns the majority voted response from a number of neighbours
    classVotes = {}
    for x in neighbors:
        response = labels[x]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1

    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)

    return sortedVotes[0][0]

#referenced from :http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
#returns sums the total correct predictions and returns the accuracy as a percentage of correct classifications
def getAccuracy(test_set, predictions, labels):
    correct = 0
    for x in range(len(test_set)):
        if labels[test_set[x][-1]] == predictions[x]:
            correct +=1 

    return (correct/float(len(test_set))) * 100.0
#main body where variables and lists are intialized.
def main(splitvalue,kvalue):
    train_set=[]
    test_set=[]
    documents=[]
    labels=[]
    split = splitvalue

    print("\nLOADING DATASET....")
    News_Document, News_Labels = load_data()
    print("{:d} News Doc loaded".format(len(News_Document)-2))
    print("{:d} News Labels loaded".format(len(News_Labels)))



    documents, labels = termfreq(News_Document,News_Labels)

    train_set, test_set =load_dataset(split,documents,train_set,test_set)

    print("\nInitiating Predictions")
    predictions=[]
    k = kvalue
    print('SPLIT VALUE : ',split)
    print('K VALUE : ', k)
    for x in range(len(test_set)):
            neighbors = getNeighbors(train_set, test_set[x], k)
            result = getResponse(neighbors,labels)
            predictions.append(result)
    accuracy = getAccuracy(test_set, predictions, labels)
    print('ACCURACY: ' + repr(accuracy) + '%')

for k in range (1,11):
    main(0.7,k)

