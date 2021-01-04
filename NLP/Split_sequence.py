from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

string = "The science of today is not the techology of tomorrow"

print(word_tokenize(string))