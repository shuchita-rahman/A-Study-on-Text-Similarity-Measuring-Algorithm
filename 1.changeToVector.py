from sklearn.feature_extraction.text import TfidfVectorizer
from mosestokenizer import MosesDetokenizer
from nltk.corpus import stopwords

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

#data are tokenized to convert it to vector we must detokenize data or make it a string

detokenize = MosesDetokenizer('en')
mainData = detokenize(mainData)
checkSimilarityData = detokenize(checkSimilarityData)

#printing data whinch are now string
print(mainData)
print(checkSimilarityData)

#changing sentence to vector
tfidf = TfidfVectorizer()
fitDataOne = tfidf.fit_transform([mainData])
feature_names = tfidf.get_feature_names()

#for every word how much similar they are 
for col in fitDataOne.nonzero()[1]:
    print(feature_names[col], ' - ', fitDataOne[0, col])

fitDataTwo = tfidf.fit_transform([checkSimilarityData])
feature_names = tfidf.get_feature_names()

#feature_names = tfidf.get_feature_names(fitDataTwo)    
print("\n-------------\n")
for col in fitDataTwo.nonzero()[1]:
    print(feature_names[col], ' - ', fitDataTwo[0, col])    
