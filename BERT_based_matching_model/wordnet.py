import nltk
from nltk.corpus import wordnet as wn

def get_definitions(word):
    synsets = wn.synsets(word)
    definitions = []
    for synset in synsets:
        definitions.append(synset.definition())
    return definitions

def main():
    nltk.download('omw-1.4')
    nltk.download('wordnet')
    word = "sports"
    definitions = get_definitions(word)
    for definition in definitions:
        print(f"Definition of '{word}': {definition}")
    word = "fear"
    definitions = get_definitions(word)
    for definition in definitions:
        print(f"Definition of '{word}': {definition}")
if __name__ == "__main__":
    main()