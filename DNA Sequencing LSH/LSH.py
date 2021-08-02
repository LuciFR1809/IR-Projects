"""
LSH Based Text Retrieval System

Retrieving Documents based on the given Genome transcript, using a sample dataset of 4380 
strings of human genome transcripts. LSH has been used to identify the most similar documents 

"""

import numpy as np
import json
#import matplotlib.pyplot as plt
import timeit
import random

"""
preprocessing()

Pre-processes the dataset and retrieves all the genome sequences
"""
def preprocessing():    
    file = open("human_data.txt",'r')
    humandata = file.read()
    humandata = humandata.split()
    count=0
    gene=[]
    classes=[]
    for _ in range(0,7):
      classes.append([])
    for lines in humandata[2:]:
      # print(lines)
      if count%2 == 0:
        gene.append(lines)
      else:
        # print(lines)
        y = int(lines)
        classes[y].append(count/2)
      count=count+1
    #print(count/2)
    return gene

"""
shingling(gene,shingle_index, shingle_size=5)

Creates a boolean matrix indicating the presence or absence of a shingle inside a document.This 
matrix is then stored inside a text file named doc_shingle.txt

### VARIABLES
    #doc_shingle : boolean matrix
    #shingle_size=5 by default
    #shingle_index={} : dictionary maps shingle to index
"""
def shingling(gene,shingle_index, shingle_size=5):
    #alternative
    start = timeit.default_timer()
    count=0
    #doc_shingle = np.zeros((1,4380)) 
    doc_shingle= np.array([]).reshape(0,len(gene))
    #shingle_index is storing the shingle number
    for doc in gene: 
      for i in range(0,int(len(doc)-shingle_size)):
        a = doc[i:i+shingle_size]
        if 'N' in a:
          continue
        if a not in shingle_index:
          shingle_index[a] = len(shingle_index)
          doc_shingle = np.vstack ((doc_shingle,np.zeros((1,len(gene))))) # corresponding shingle 4380 row
          doc_shingle[shingle_index[a]][count] = 1
        else:
          doc_shingle[shingle_index[a]][count] = 1
      count+=1
    np.savetxt("doc_shingle.txt",doc_shingle)
    end = timeit.default_timer()
    print("Time taken for Shingling : ", end-start)
    # with open("shingles.txt","w") as file:
    #   json.dump(shingle_index,file)

"""
first_nonzero(arr, axis, invalid_val=-1)

Finds the first index with non-zero matrix value. It is used to find the signature matrix
"""
def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

"""
minHash(doc_shingle)

Finds the signature matrix for all the given documents

###VARIABLES
    #min_hash : signature matrix for 100 hash functions 
    
"""

def minHash(doc_shingle):
    start = timeit.default_timer()
    min_hash = np.zeros((1,doc_shingle.shape[1]))
    for _ in range(0,100):
      np.random.shuffle(doc_shingle)
      # list.append(doc_shingle)
      min_hash = np.vstack ((min_hash, first_nonzero(doc_shingle,0)))
    min_hash = np.delete(min_hash,0,axis=0)
    end = timeit.default_timer()
    print("Time taken to create Min Hash : ", end-start)
    return min_hash
    # print(min_hash)

"""
query_hashing(query,shingle_index,shingle_size)

Finds the boolean matrix corresponding to the given query

###VARIABLES
    #query_shingles : boolean matrix for query
"""

def query_hashing(query,shingle_index,shingle_size):
    query_shingles=np.zeros((len(shingle_index),1))
    for i in range(0,len(query)-shingle_size):
      a = query[i:i+shingle_size]
      if a in shingle_index:
        query_shingles[int(shingle_index[a])][0]=1
    return query_shingles

"""
Bucketing(a,b,min_hash,r)

Creates the buckets for LSH step

###VARIABLES
    #buckets : A dictionary containing documents with same hash for the same bands.
"""

def Bucketing(a,b,min_hash,r):
    buckets={}
    doc_size=min_hash.shape[1]-1
    for docs in range(0,doc_size):
      temp_sum=0
      for rows in range(0,r):
        temp_sum+=min_hash[rows][docs]
      temp_sum=(a*temp_sum+b)%doc_size
      if temp_sum in buckets:
        buckets[temp_sum].add(docs)
      else:
        buckets[temp_sum]=set()
        buckets[temp_sum].add(docs)
    return buckets

"""
JaccardIndex(min_hash)

Calculates the similarity between query and documents and returns an array indicating the
similarity measure of each document corresponding to the query

###VARIABLES
    #answer : set of similar documents
    #buckets : A dictionary containing documents with same hash for the same bands given by Bucketing function
"""
def JaccardIndex(min_hash):
  c = min_hash.shape[1]
  row=20
  band=5
  #print(type(band))
  freq=np.zeros(c-1)
  answer=[]
  #query_col=min_hash[:,c-1]
  for bands in range(0,band):
    a=random.random()
    b=random.random()
    temp_sum=0
    for rows in range(bands,bands+row):
      temp_sum+=min_hash[rows][c-1]
    temp_sum=(a*temp_sum+b)%(c-1)
    buckets=Bucketing(a,b,min_hash,row)
    if temp_sum in buckets:
      for doc in buckets[temp_sum]:
        s=similarity(min_hash[:,doc],min_hash[:,c-1])
        if  s >= 0.6:
          answer.append(doc)
  freq=(freq/band)
  return set(answer)

def similarity(colA, colB):
  intersect=0
  for i in range(0,100):
    if colA[i] == colB[i]:
      intersect+=1
  return intersect/100



"""
query_processing(query,shingle_index,shingle_size)

Takes in the query, processes it and prints the most similar document corresponding to that query
"""

def query_processing(query,shingle_index,shingle_size):
  start=timeit.default_timer()
  doc_shingle=np.loadtxt("doc_shingle.txt")
  query_shingle=query_hashing(query,shingle_index,shingle_size)
  # print(query_shingle.shape[0],query_shingle.shape[1],doc_shingle.shape[0],doc_shingle.shape[1])
  doc_shingle=np.concatenate((doc_shingle,query_shingle),axis=1)
  min_hash=minHash(doc_shingle)
  print("Most relevant Document indices are ",JaccardIndex(min_hash))
  end = timeit.default_timer()
  print("Time taken to retrieve answer : ", end-start)

"""
Driver code
"""
if __name__ == "__main__":
    gene=preprocessing()
    shingle_size=int(input("Enter shingle size: "))
    shingle_index={}
    shingling(gene,shingle_index,shingle_size)
    while True:
        query = input("Press Quit to exit. Enter Query : ")
        if query == "Quit":
            break
        else:
            query_processing(query,shingle_index,shingle_size)