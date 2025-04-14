import nltk_engine
import spacy_engine
import os

def get_file_names(path):
    try:
        return [os.path.join(path, f) for f in os.listdir(path)]
    except FileNotFoundError:
        return []
if __name__ == '__main__':
    names = get_file_names("data")
    files = [open(f, encoding="utf-8", errors="ignore") for f in names]

    # files = [open("/Users/steventruong/School/csc483/jeopardy/data/enwiki-20140602-pages-articles.xml-0099.txt")]

    ir = nltk_engine.IRSystem(files)
    # ir = faster_engine.IRSystem(files)