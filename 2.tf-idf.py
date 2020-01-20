from sklearn.feature_extraction.text import TfidfVectorizer
from mosestokenizer import MosesDetokenizer
from nltk.corpus import stopwords
from math import sqrt
from decimal import Decimal
#open to file in this directory
f1= open('/home/shuchita-rahman/Documents/thesis/f1', 'r')
f2= open('/home/shuchita-rahman/Documents/thesis/f2', 'r')


#put file data in data variable
mainData = f1.read()
checkSimilarityData = f2.read()


#remove stop word from this data(am, an etc.)
stop = set(stopwords.words('english'))
mainData = [word for word in mainData.split() if word not in stop]
checkSimilarityData= [word for word in checkSimilarityData.split() if word not in stop]
#print(mainData)
#print(checkSimilarityData)
#data are tokenized to convert it to vector we must detokenize data or make it a string

detokenize = MosesDetokenizer('en')
mainData = detokenize(mainData)
checkSimilarityData = detokenize(checkSimilarityData)
#printing data whinch are now string
#print(mainData)
#print(checkSimilarityData)

#changing sentence to vector
tfidf = TfidfVectorizer()
response = tfidf.fit_transform([mainData, checkSimilarityData])

#feature_names = tfidf.get_feature_names()
#print(response)
#print(feature_names)

#for every word how much similar they are 


#for col in response.nonzero()[1]:
 #    print(feature_names[col], ' - ', response[0, col],'-',response[1, col])
#seperate standard answer and sample answer     
x=[response.nonzero()[1]][0]
y=[response.nonzero()[0]][0]

standard=[]
sample=[]

j=0
k=0
l=0
for i in y:
    if i == 0:
        standard.insert(j,response[0,x[k]])
    
    elif i== 1:
        sample.insert(l,response[0,x[k]])
    k=k+1    
         
    
 # Euclidean distance:
def euclidean_distance(standard,sample):
  return sqrt (sum(pow(a-b,2) for a,b in zip(standard,sample)))      
    
a=euclidean_distance(standard,sample)

#b = (a*100) / 10
print('\n\n  Euclidean: ',a)   

#mahanttan distance
def manhattan_distance(standard,sample):
    return sum(abs(a-b) for a,b in zip(standard,sample))
b=manhattan_distance(standard,sample) 
#b = (b*100) / 10
print('\n  Manhattan:  ', b)   

#minkowski distance
def nth_root(value,n_root):
    root_value=1/float(n_root)
    return round(Decimal(value)** Decimal(root_value),3)
def minkowski_distance(standard,sample):
    return nth_root(sum(pow(abs(a-b),3) for a,b in zip(standard,sample)),3)
b=minkowski_distance(standard,sample) 
#b = (b*100) / 10
print('\n  Minkowski:  ',b)   


#cosine distance
def square_rooted(x):
    return round(sqrt(sum(a*a for a in x)),3)
def cosine_distance(standard,sample):
    numerator=sum(a*b for a,b in zip(standard,sample))
    denumerator=square_rooted(standard)*square_rooted(sample)
    return round(numerator/float(denumerator),3)
b=cosine_distance(standard,sample) 
#b = (b*100) / 10
print('\n  Cosine   :  ', b)
#jaccard distance
def jaccard_distance(stndard,sample):
    intersection=len(set.intersection(*[set(standard),set(sample)]))
    union=len(set.union(*[set(standard),set(sample)]))
    return intersection/float(union)
b=jaccard_distance(standard,sample) 
#b = (b*100) / 10
print('\n  Jaccard  :  ', b)   
  
