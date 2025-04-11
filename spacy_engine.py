import collections
import math
import argparse
import re
import spacy
import json
from tqdm import tqdm
import time

model = spacy.load("en_core_web_sm")

weights_path = "IRdata/weights.json"
counts_path = "IRdata/counts.json"

docID_pattern = re.compile(r"^\[\[.+\]\]\n$", re.MULTILINE)
skip_line_pattern = re.compile(r"=+.+=+")

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
            print("Loading Weights")
            with open(weights_path) as file:
                self.weights = json.load(file)
            print("Loading Counts")
            with open(counts_path) as file:
                self.counts = json.load(file)   
            return

        # Parse file and calculate weights and counts
        file_count = 0
        for f in files:
            text = f.read()
            ids = docID_pattern.findall(text)
            lines = docID_pattern.split(text)[1:]
            print("Attempting Pipe")
            start_time = time.time()
            docs = list(model.pipe(lines, batch_size=256, n_process=8, disable=["parser", "ner"]))
            print(f"Working on file {file_count}")
            for i, tokens in tqdm(enumerate(docs), total=len(docs)):
                docID = ids[i][2:-3]
                seen = set()
                if docID not in self.weights:
                    self.weights[docID] = {}
                for token in tokens:
                    if token.is_punct or token.is_space:
                        continue
                    token = token.lemma_

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

                # Calculate weights from term frequency
                total = 0
                for term, count in self.weights[docID].items():
                    if (count <= 0):
                        print(term + " " + str(count))
                        print(docID)
                    tf = 1 + math.log10(count)
                    total += tf * tf
                    self.weights[docID][term] = tf

                # Cosine normalization
                for term, weight in self.weights[docID].items():
                    cos = weight * math.sqrt(1/total)
                    self.weights[docID][term] = cos
            file_count += 1
            end_time = time.time()
            print(end_time-start_time)
            
        

        with open(weights_path, "w") as file:
            json.dump(self.weights, file, indent=4)
        with open(counts_path, "w") as file:
            json.dump(self.counts, file, indent=4)

    def run_query(self, query):
        terms = model(query)
        terms = [term.lemma_ for term in terms]
        return self._run_query(terms)

    def _run_query(self, terms):
        # Use ltn to weight terms in the query:
        #   l: logarithmic tf
        #   t: idf
        #   n: no normalization

        # Return the top-10 document for the query 'terms'
        result = []

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
            similarity = 0
            for term in terms:
                if term in df_weights:
                    similarity += weights[term] * df_weights[term]
            if similarity not in sums:
                sums[similarity] = []
            sums[similarity].append(docID)

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
