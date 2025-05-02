import nltk_engine
#import spacy_engine
from tqdm import tqdm


# Load in the IR system from the .json files
ir = nltk_engine.IRSystem()
#ir = spacy_engine.IRSystem()

questions = open("questions.txt").readlines()
queries = []
for i in range(0, len(questions), 4):
    cur = (questions[i].strip(), questions[i+1].strip(), questions[i+2].strip())
    queries.append(cur)

correct = 0
file = open("answers.txt", "w")
for query in tqdm(queries):
    valid = query[2].split('|')
    result = ir.run_query(query[1])
    if result[0] in valid:
        print("CORRECT ANSWER FOUND:")
        print(query)
        correct += 1
    else:
        print("INCORRECT ANSWER FOUND:")
        print(query[0])
        print(query[1])
        print(result[0])
        print("Correct answer was:", query[2])
    file.write(f"{query[0]}\n{query[1]}\n{result}\n\n")
    
print(f"Precision: {(correct/100)}")


