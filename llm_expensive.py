import torch
import os
from transformers import pipeline
from huggingface_hub import InferenceClient
from tqdm import tqdm

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
api_key = "TEMP"

client = InferenceClient(
    provider="cerebras",
    api_key=api_key
)

print("Loading LLM")

if (os.path.exists("cur_question.txt")):
    with open("cur_question.txt", "r") as f:
        questions_to_skip = int(f.readline().strip())
        correct = int(f.readline().strip())
else:
    questions_to_skip = 0
    correct = 0

questions_to_skip = 0 if questions_to_skip == 100 else questions_to_skip
correct = 0 if questions_to_skip == 100 else correct

questions = open("questions.txt").readlines()
queries = []
for i in range(questions_to_skip*4, len(questions), 4):
    cur = {"category":questions[i].strip(), "prompt":questions[i+1].strip(), "answer":[x.lower() for x in questions[i+2].strip().split("|")]}
    queries.append(cur)

system_msg = "You are a Jeopardy expert and you are tasked with answering the question by providing only the wikipedia page title.\n" \
"Do not answer in a question format. You are to not speculate and only refer to wikipedia pages for your answers. Your answers must be relevant to the given category"


msg = "Give me a list of the 10 best possible answers in a CSV format. (EG. The Washington Post,The New York Times,...).\n" \
"Only output the answers, do not add any other filler text such as 'Here are the possible answers:'.\n" \
"Ensure that you do not include duplicate answers.\n" \
"If the category has an instruction in parentheses, ensure that you follow it's instructions as it specifies what kind of answer it should be. (Ex. We'll give you the museum, you give us the state. ONLY GIVE THE STATE)\n" \
"Make sure the answers are all on one line and does not include any quotes or delimiters besides the commas.\n" \
"The question is delimited by ''' and the category is delimited for ---.\n" \
"---{}--- '''{}'''"

# file = open("answers.txt", "w")
i = 0
for query in tqdm(queries):
    #result = ir.run_query(query[0], query[1])
    print(query["prompt"])

    messages = [
        {"role":"user", "content": system_msg + msg.format(query["category"], query["prompt"])}
    ]

    output = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=250
    )

    answers = output.choices[0].message["content"]
    answers = answers.split(",")
    answers = [x.strip().strip("\"").lower() for x in answers]
    print(answers)
    
    if len(set(query["answer"]) & set(answers)) != 0:
        print('Correct')
        correct += 1
    # file.write(f"{query[0]}\n{query[1]}\n{result}\n\n")
    i += 1
    with open("cur_question.txt", "w") as f:
        f.write(str(questions_to_skip+i) + "\n" + str(correct))
    
print(f"Accuracy: {(correct/100)*100}")



