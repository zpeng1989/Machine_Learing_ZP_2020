import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

text_data = np.array(['I love Brazil, Brazil', 'Brazil is best', 'Germany beats both'])

count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)

features = bag_of_words.toarray()

print(features)

target = np.array([0,0,1])

classifer = MultinomialNB(class_prior = [0.25, 0.5])

model = classifer.fit(features, target)
new_observation = [[0,0,0,1,0,1,0]]

print(model.predict(new_observation))


features = np.random.randint(2, size=(100,3))
target = np.random.randint(2, size=(100,1)).ravel()

classifer = MultinomialNB(class_prior= [0.25, 0.5])

model = classifer.fit(features, target)



