import torch
import os
from tqdm import tqdm
import nltk_selector
from dotenv import load_dotenv
from openai import OpenAI



load_dotenv()
ir = nltk_selector.IRSystemSelector()

print("Loading LLM")
api_key = os.getenv("OPENAI_KEY")
if not api_key:
    raise Exception("No OpenAI API key found")

client = OpenAI(
    api_key=api_key
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
"The questions you are answering use words different from those of an appropriate answer. Ensure that all of the answers you give answer the question rather than repeating a part of it. For example, if the question is 'George Washington was born in what is now this state.', 'George Washington' would be a bad answer. Additionally, as another example, if the question is 'This person is the editor-in-chief of The New York Times.', then 'The New York Times', 'New York Times', 'N.Y. Times', and 'NY Times' would all be bad answers.\n" \
"Prefer titles that best match what the question is asking for. Many questions contain the word 'this' or 'these' followed by or preceded by a descriptor. Make sure to provide answers that match these phrases, as they provide some specifics about the kind of answer you are looking for. For example, if the question contains details about 'these cuddly animals,' you must answer a kind of animal. If the question contains 'this woman,' you must answer a woman's name. If the question contains 'this European country,' your answer must be a country in Europe.\n"\
"If the question provided is just a phrase, your job is to find the answers that best match the given category, with the question being a more specific clue to narrow down your search. For example, if you are given the question 'Phoenix' and the category 'IT IS THE CAPITAL OF...', then you are to find the US state whose capital is Phoenix.\n" \
"If the question is asking for a person, answer the person's name. For example, if the question is, 'This man was the fourteenth president of the United States.', 'Franklin Pierce' is the best answer, but 'President Franklin Pierce' and 'President Pierce' are bad answers.\n" \
"If the question includes a year or range of years (this may include broad ranges, such as 'the late 1700s'), your answer must be relevant to some event that occurred in that year. For example, if your question is 'In 1869, this man was inaugurated as US President.', then your answer must be the name of the president whose term began in 1869.\n"\
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
    outputs = client.chat.completions.create(
    model="gpt-4o-mini",
    store=True,
    temperature=0,
    messages=messages
    )

    print("LLM generated answers:")
    print(outputs.choices[0].message.content)
    answers = outputs.choices[0].message.content.split(",")
    answers = [x.strip().strip("\"").replace("-", " ") for x in answers]

    filtered_answers = ir.run_query(query["prompt"], answers)
    print("tf-idf filtered answers:")
    print(filtered_answers)

    
    
    #if len(set(query["answer"]) & set(answers)) != 0:
    if filtered_answers[0].lower() in query["answer"]:
        # first answer given by the LLM is correct
        print('Correct')
        correct += 1
    # file.write(f"{query[0]}\n{query[1]}\n{result}\n\n")
    i += 1
    with open("cur_question.txt", "w") as f:
        f.write(str(questions_to_skip+i) + "\n" + str(correct))
    
print(f"Precision: {correct/100}")      # Precision = (# of correct answers retrieved) / (# total answers retrieved)



