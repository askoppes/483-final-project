import collections
import math
import argparse
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import json
from tqdm import tqdm
from functools import lru_cache
import os
import string

ps = PorterStemmer()

weights_path = "IRdata/weights2.json"
counts_path = "IRdata/counts2.json"

docID_pattern = re.compile(r"\[\[.+\]\]")
skip_line_pattern = re.compile(r"=+.+=+")

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
              "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", 
              "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", 
              "theirs", "themselves", "what", "which", "who", "whom", "this", "that", 
              "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", 
              "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", 
              "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", 
              "with", "about", "against", "between", "into", "through", "during", "before", "after", 
              "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", 
              "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", 
              "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", 
              "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", 
              "should", "now"]

@lru_cache(maxsize=10000000)
def cached_stem(token):
    return ps.stem(token)

def get_file_names(path):
    try:
        return [os.path.join(path, f) for f in os.listdir(path)]
    except FileNotFoundError:
        return []

class IRSystemSelector:
    def __init__(self):
        pass
        
    def calc_weights(self, titles):
        # Reset weights and counts
        self.weights = {}
        self.counts = {}
        self.titles = {}

        file_count = 0
        
        names = get_file_names("data")
        files = [open(f, encoding="utf-8", errors="ignore") for f in names]
        for f in files:
            docID = None
            seen = set()
            lines = f.readlines()
            for line in lines:
                # Detect the title of the current document
                if docID_pattern.match(line) and line[2:-3] in titles:
                    # Check if prev was none
                    if docID != None and docID in self.weights:
                        # Calculate weights from term frequency for previous document
                        total = 0
                        for term, count in self.weights[docID].items():
                            tf = 1 + math.log10(count)
                            total += tf * tf
                            self.weights[docID][term] = tf

                        # Cosine normalization
                        for term, weight in self.weights[docID].items():
                            cos = weight * math.sqrt(1/total)
                            self.weights[docID][term] = cos

                    docID = line[2:-3]
                    seen = set()
                    continue
                # If we're on a title that is not in our set, set docID to none and skip it
                elif docID_pattern.match(line) and docID != None:
                    docID = None
                    continue

                
                # If title not found yet
                if (docID == None):
                    continue
                
                # Skip new lines
                if line == "\n":
                    continue

                # Skip header lines
                if skip_line_pattern.match(line):
                    continue

                if (docID not in self.weights):
                    self.weights[docID] = {}

                tokens = line.strip().split()     
                for token in tokens:
                    token = cached_stem(token)
                    # Count document frequency
                    if token not in self.counts:
                        self.counts[token] = 0
                    if token not in seen:
                        self.counts[token] += 1
                        seen.add(token)

                    # Count term frequency in document
                    if token not in self.weights[docID]:
                        self.weights[docID][token] = 0
                    self.weights[docID][token] += 1
                
            # Calculate weights from term frequency for very last document
            if docID is not None and docID in self.weights:
                total = 0
                for term, count in self.weights[docID].items():
                    tf = 1 + math.log10(count)
                    total += tf * tf
                    self.weights[docID][term] = tf

                # Cosine normalization
                for term, weight in self.weights[docID].items():
                    cos = weight * math.sqrt(1 / total)
                    self.weights[docID][term] = cos

            file_count += 1
        
        # Get version of all the docIDs (titles) with no stop words to compare with LLM results
        for title in self.weights.keys():
            # remove any stop words (so we don't fail to match a title if the only non-match is stop words)
            title_copy = title
            title_copy.translate(str.maketrans('', '', string.punctuation))
            filtered_title = [term for term in title_copy.lower() if not term in set(stopwords.words('english'))]
            no_stop_words = " ".join(filtered_title)
            self.titles[no_stop_words] = title


    def run_query(self, query, top_ten):
        terms = query.strip().split()
        expanded = []
        # synonyms added
        for term in terms:
            synset = wn.synsets(term)
            if synset:
                expanded.extend(synset[0].lemma_names())
            else:
                expanded.append(term)
        terms = list(set(expanded))
        terms = [cached_stem(term) for term in terms]
        self.calc_weights(top_ten)
        return self._run_query(terms, top_ten)

    def _run_query(self, terms, top_ten):
        # Use ltn to weight terms in the query:
        #   l: logarithmic tf
        #   t: idf
        #   n: no normalization

        # Return the top-10 document for the query 'terms'
        result = []

        options = []
        for possible in top_ten:
            title_copy = possible
            title_copy.translate(str.maketrans('', '', string.punctuation))
            filtered_result = [term for term in title_copy.lower() if not term in set(stopwords.words('english'))]
            possible_title = " ".join(filtered_result)
            if possible_title in self.titles.keys() and self.titles[possible_title] not in options:
                options.append(self.titles[possible_title])

        weights = {}
        # Count terms
        for term in terms:
            if term not in weights:
                weights[term] = 0
            weights[term] += 1
        
        # Calc tf-idf weight from counts
        for term in terms:
            if term in self.counts:
                # Duplicates in query
                if (weights[term] <= 0):
                    continue
                tf = 1 + math.log10(weights[term])
                idf = math.log10(len(self.weights)/self.counts[term])
                weights[term] = tf * idf
            else:
                weights[term] = 0

        # Traverse docs to calc weights for each doc
        sums = {}
        
        for docID in options:
            title_tokens = [cached_stem(w.lower()) for w in docID.split() if w.lower() not in set(stopwords.words('english'))]
            query_tokens = [t for t in terms if t not in set(stopwords.words('english'))]
            
            similarity = 0
            if (docID in set(self.weights.keys())) and (docID in set(options)) and (set(title_tokens).isdisjoint(set(query_tokens))):
                # the document actually exists, also there's no exact match for the title in the query
                for term in terms:
                    if term in self.weights[docID]:
                        similarity += weights[term] * self.weights[docID][term]
            if similarity not in sums:
                sums[similarity] = []
            sums[similarity].append(docID)
        
        if len(sums.keys()) == 0:
            # we probably won't get much this way, but this at least ensures we never return nothing
            for docID, df_weights in self.weights.items():
                # print("title? ", docID)
                title_tokens = [cached_stem(w.lower()) for w in docID.split() if w.lower() not in stop_words]
                query_tokens = [t for t in terms if t not in stop_words]
            
                if set(title_tokens).isdisjoint(set(query_tokens)):
                    similarity = 0
                    for term in terms:
                        if term in df_weights:
                            similarity += weights[term] * df_weights[term]
                    if similarity not in sums:
                        sums[similarity] = []
                    sums[similarity].append(docID)
                else:
                    continue

        highest = sorted(sums.keys(), reverse=True)
        i = 0
        while i < len(highest) and len(result) < 10:
            result += sums[highest[i]]
            i += 1

        if len(result) == 0:
            return top_ten
        return result


def main(corpus):
    ir = IRSystemSelector(open(corpus))

    while True:
        query = input('Query: ').strip()
        if query == 'exit':
            break
        results = ir.run_query(query)
        print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("CORPUS",
                        help="Path to file with the corpus")
    args = parser.parse_args()
    main(args.CORPUS)