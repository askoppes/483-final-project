import collections
import math
import argparse
import re
import nltk
from nltk.stem import PorterStemmer
import json
from tqdm import tqdm
from functools import lru_cache

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

class IRSystem:
    def __init__(self, files=None):
        # Use lnc to weight terms in the documents:
        #   l: logarithmic tf
        #   n: no df
        #   c: cosine normalization

        # Store the vecorized representation for each document
        #   and whatever information you need to vectorize queries in _run_query(...)

        # YOUR CODE GOES HERE
        self.weights = {}
        self.counts = {}
        
        # Load in the weights and counts from JSON files
        if files == None:
            with open(weights_path) as file:
                print("Loading Weights")
                self.weights = json.load(file)
            with open(counts_path) as file:
                print("Loading Counts")
                self.counts = json.load(file)   
            return

        # Parse file and calculate weights and counts
        file_count = 0
        for f in files:
            print(f"Working on file {file_count}")
            docID = None
            seen = set()
            lines = f.readlines()
            for line in tqdm(lines):
                # Detect the title of the current document
                if docID_pattern.match(line):
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
            
       

        with open(weights_path, "w") as file:
            print("Writing to json file")
            json.dump(self.weights, file, indent=4)
        with open(counts_path, "w") as file:
            print("Writing to json file")
            json.dump(self.counts, file, indent=4)

    #def run_query(self, category, query):
    def run_query(self, query):
        terms = query.strip().split()
        terms = [cached_stem(term) for term in terms]
        #topic = category.strip().split()
        #topic = [cached_stem(word) for word in topic]
        #return self._run_query(topic, terms)
        return self._run_query(terms)

    #def _run_query(self, topic, terms):
    def _run_query(self, terms):
        # Use ltn to weight terms in the query:
        #   l: logarithmic tf
        #   t: idf
        #   n: no normalization

        # Return the top-10 document for the query 'terms'
        result = []

        #all_terms = terms + topic

        weights = {}
        # Count terms
        #for term in all_terms:
        for term in terms:
            if term not in weights:
                weights[term] = 0
            weights[term] += 1
        
        # Calc tf-idf weight from counts
        #for term in all_terms:
        for term in terms:
            if term in self.counts:
                # Duplicates in query
                if (weights[term] < 0):
                    continue
                tf = 1 + math.log10(weights[term])
                idf = math.log10(len(self.weights)/self.counts[term])
                weights[term] = tf * idf
            else:
                weights[term] = 0

        # Traverse docs to calc weights for each doc
        sums = {}
        
        for docID, df_weights in self.weights.items():
            print("title? ", docID)
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
        while len(result) < 10:
            result += sums[highest[i]]
            i += 1

        result = result[:10]
    
        return result


def main(corpus):
    ir = IRSystem(open(corpus))

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