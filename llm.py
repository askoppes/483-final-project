import torch
import os
from transformers import pipeline
from huggingface_hub import InferenceClient
from tqdm import tqdm

model_id = "meta-llama/Llama-3.2-3B-Instruct"

print("Loading LLM")
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

if (os.path.exists("cur_question.txt")):
    with open("cur_question.txt", "r") as f:
        questions_to_skip = int(f.readline().strip())
        correct = int(f.readline().strip())
else:
    questions_to_skip = 0
    correct = 0

correct = 0 if questions_to_skip == 100 else correct
questions_to_skip = 0 if questions_to_skip == 100 else questions_to_skip

questions = open("questions.txt").readlines()
queries = []
for i in range(questions_to_skip*4, len(questions), 4):
    cur = {"category":questions[i].strip(), "prompt":questions[i+1].strip(), "answer":[x.lower() for x in questions[i+2].strip().split("|")]}
    queries.append(cur)

system_msg = "You are a Jeopardy expert and you are tasked with answering the question by providing only the Wikipedia page title.\n" \
"Do not answer in a question format. You are to not speculate and you must refer only to Wikipedia pages for your answers. Your answers must be relevant to the given category."


msg = "Provide a comma-separated list of the 10 best possible answers to the given question. (EG. The Washington Post,The New York Times,...). The list must be ordered such that the answer that best answers the question is the first item in the list, the second best answer is the second item in the list, and so on. Your list must contain exactly 10 answers.\n" \
"Only output the answers, do not add any other filler text such as 'Here are the possible answers:'.\n" \
"When creating your list of answers, ensure that you follow these guidelines:\n" \
"Ensure that all answers in your list are unique, that is, each answer in your list should appear exactly once.\n" \
"The questions you are answering provide clues to the answers which you are seeking, and thus use words different from those of an appropriate answer. Ensure that all of the answers you give answer the question rather than repeating a part of it. For example, if the question is 'George Washington was born in what is now this state.', 'George Washington' would be a bad answer. Additionally, as another example, if the question is 'This person is the editor-in-chief of The New York Times.', then 'The New York Times', 'New York Times', 'N.Y. Times', and 'NY Times' would all be bad answers.\n" \
"If the category includes an instruction in parentheses, starting with 'Alex: ', ensure that you follow its instructions as it specifies what kind of answer it should be. For example, if the instructions say 'Alex: We've given you the monument. You give us the city.', then you must give the city in which the monument in the question is located.\n" \
"Make sure the answers are all on one line and does not include any quotes or delimiters besides the commas.\n" \
"The question is delimited by ''' and the category is delimited by ---.\n" \
"---{}--- '''{}'''"

# file = open("answers.txt", "w")
i = 0
for query in tqdm(queries):
    #result = ir.run_query(query[0], query[1])
    print(query["prompt"])
    # messages = [
    # {"role": "system", "content": system_msg},
    # {"role": "user", "content": msg.format(query["category"], query["prompt"])},
    # ]

    messages = [
        {"role":"user", "content": system_msg + msg.format(query["category"], query["prompt"])}
    ]
    print(query["answer"])
    outputs = pipe(
        messages,
        max_new_tokens=256,
        pad_token_id=128001  
    )
    print(outputs[0]["generated_text"][-1]["content"])
    answers = outputs[0]["generated_text"][-1]["content"].split(",")
    answers = [x.strip().strip("\"").lower() for x in answers]

    
    
    #if len(set(query["answer"]) & set(answers)) != 0:
    if answers[0] in query["answer"]:
        # first answer given by the LLM is correct
        print('Correct')
        correct += 1
    # file.write(f"{query[0]}\n{query[1]}\n{result}\n\n")
    i += 1
    with open("cur_question.txt", "w") as f:
        f.write(str(questions_to_skip+i) + "\n" + str(correct))
    
print(f"Precision: {correct/100}")      # Precision = (# of correct answers retrieved) / (# total answers retrieved)



