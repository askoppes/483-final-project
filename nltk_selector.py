import collections
import math
import argparse
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
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

class IRSystemSelector:
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
        self.titles = {}
        #self.title_weights = {}
        #self.title_counts = {}
        
        # Load in the weights and counts from JSON files
        if files == None:
            with open(weights_path) as file:
                print("Loading Weights")
                self.weights = json.load(file)
            with open(counts_path) as file:
                print("Loading Counts")
                self.counts = json.load(file)   
            # Get stemmed version of all the docIDs (titles) to compare with LLM results
            for title in self.weights.keys():
                # remove any stop words (so we don't fail to match a title if the only non-match is stop words)
                filtered_title = [term for term in title.lower() if not term in set(stopwords.words('english'))]
                #title_terms = [cached_stem(term) for term in filtered_title]
                fixed_title = " ".join(filtered_title)
                self.titles[fixed_title] = title
                '''
                # Old scoring titles stuff (get rid of if we abandon this path)
                included = set()
                curr_weights = {}
                #stem_title = " ".join(title_terms)
                #self.titles[stem_title] = title
                for term in title_terms:
                    if term not in included:
                        tf = self.log_tf(term, title_terms)
                        curr_weights[term] = tf
                        included.add(term)
                        if term in self.title_counts:
                            self.title_counts[term] += 1
                        else:
                            self.title_counts[term] = 1
                # now perform cosine normalization on all weights
                weight_sum = 0
                for weight in curr_weights:
                    # first need sum of square of all weights
                    weight_sum += (curr_weights[weight] ** 2)
            
                cosine_norm = math.sqrt(weight_sum)
            
                for item in curr_weights:
                    curr_weights[item] /= cosine_norm
            
                self.title_weights[title] = curr_weights   # now add the weights
                '''
            return        # don't delete this!!!

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
        
        # Get stemmed version of all the docIDs (titles) to compare with LLM results
        for title in self.weights.keys():
            # remove any stop words (so we don't fail to match a title if the only non-match is stop words)
            filtered_title = [term for term in title.lower() if not term in set(stopwords.words('english'))]
            #title_terms = [cached_stem(term) for term in filtered_title]
            no_stop_words = " ".join(filtered_title)
            self.titles[no_stop_words] = title
            '''
            included = set()
            curr_weights = {}
            #stem_title = " ".join(title_terms)
            #self.titles[stem_title] = title
            for term in title_terms:
                if term not in included:
                    tf = self.log_tf(term, title_terms)
                    curr_weights[term] = tf
                    included.add(term)
                    if term in self.title_counts:
                        self.title_counts[term] += 1
                    else:
                        self.title_counts[term] = 1
            # now perform cosine normalization on all weights
            weight_sum = 0
            for weight in curr_weights:
                # first need sum of square of all weights
                weight_sum += (curr_weights[weight] ** 2)
            
            cosine_norm = math.sqrt(weight_sum)
            
            for item in curr_weights:
                curr_weights[item] /= cosine_norm
            
            self.title_weights[title] = curr_weights   # now add the weights
            '''
            
       

        with open(weights_path, "w") as file:
            print("Writing to json file")
            json.dump(self.weights, file, indent=4)
        with open(counts_path, "w") as file:
            print("Writing to json file")
            json.dump(self.counts, file, indent=4)

    '''
    def log_tf(self, term, document):
        # Calculates the logarithmic tf of a term given a document
        count = 0
        for token in document:
            if token == term:
                count += 1
        if count == 0:
            return 0
        else:
            return 1 + math.log(count, 10)
    '''

    #def run_query(self, category, query):
    def run_query(self, query, top_ten):
        terms = query.strip().split()
        terms = [cached_stem(term.lower()) for term in terms]
        # stem the top results provided by the LLM to match more possible titles
        #topic = category.strip().split()
        #topic = [cached_stem(word) for word in topic]
        #return self._run_query(topic, terms)
        return self._run_query(terms, top_ten)

    #def _run_query(self, topic, terms):
    def _run_query(self, terms, top_ten):
        # Use ltn to weight terms in the query:
        #   l: logarithmic tf
        #   t: idf
        #   n: no normalization

        # Return the top-10 document for the query 'terms'
        result = []

        options = []
        for possible in top_ten:
            filtered_result = [term for term in possible.lower() if not term in set(stopwords.words('english'))]
            #possible_terms = [cached_stem(term) for term in filtered_result]
            possible_title = " ".join(filtered_result)
            if possible_title in self.titles.keys():
                options.append(self.titles[possible_title])

        # calculate idf for the document *titles* (delete later if we decide not to use it)
        '''
        options = []    # possible options for the title of our document
        possible_weights = {}
        for possible in top_ten:
            filtered_result = [term for term in possible.lower() if not term in set(stopwords.words('english'))]
            possible_terms = [cached_stem(term) for term in filtered_result]
            weights = {}
            included = set()
            for term in possible_terms:
                if term in included:
                    continue
                tf = self.log_tf(term, possible_terms)
                if term not in self.title_counts:
                    # if term from query not in document collection, it has weight of 0
                    weights[term] = 0
                else:
                    # if it is in the document collection, calculate idf
                    idf = math.log(len(self.title_weights) / self.title_counts[term], 10)
                    total_weight = tf * idf
                    weights[term] = total_weight
            possible_weights[possible] = weights
            #possible_title = " ".join(possible_terms)
            #if possible_title in self.titles.keys():
                # only keep the possible docs that actually exist
                #options.append(self.titles[possible_title])
        
        # score the document *titles* to find the best ones to narrow down our search to
        option_sums = {}
        for doc_title, title_score in self.title_weights.items():
            for possibility, possibility_weights in possible_weights.items():
                curr_score = 0
                for possible_term in possibility_weights.keys():
                    if possible_term in title_score:
                        curr_score += title_score[possible_term] * possibility_weights[possible_term]
                if curr_score not in option_sums:
                    option_sums[curr_score] = []
                option_sums[curr_score].append(doc_title)
        
        best_options = sorted(option_sums.keys(), reverse=True)
        i = 0
        while i < len(best_options) and len(best_options) < 10 and best_options[i] > 0:
            # only include document possibilities with score greater than 0 (to really narrow our search)
            options.append(option_sums[best_options[i]])
            i += 1
        '''

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
        
        for docID in options:
            # print("title? ", docID)
            title_tokens = [cached_stem(w.lower()) for w in docID.split() if w.lower() not in set(stopwords.words('english'))]
            #fix_title = " ".join(title_tokens)
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

        #result = result[:10]
    
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