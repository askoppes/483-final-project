# How to run:

1. **Create a virtual environment (can do this from VScode):** \
    python -m venv .env
2. **Activate the virtual environment (depends on OS, can do from VScode too)** \
    MacOS: source .env/bin/activate
3. **Install dependencies: python -m pip install -r requirements.txt**
4. **Import the wiki pages data into a folder named `data` and create another folder named `IRdata`**
5. **Run `load_data.py` to create .json files** 
6. **Run `test_questions.py` to test queries (This counts getting top 10 as correct)**

___

# Llama Instructions:

1. **New dependencies were added to the requirements.txt, make sure to update and reinstall it.**
2. **Register an account on Hugging Face and create an access token with the read permission.**
3. **Request access to the Llama 3.2 model and/or the Llama 4 model, more specifics can be found in the .py files under model_id**
4. **Before running `llm.py`, in the terminal use the command `huggingface-cli login`**


___

# OpenAI Instructions:
1. **Make sure to reinstall requirements.txt**
2. **Create a new file `.env` and copy and paste the contents of `.env.sample` to it.**
3. **Add your API key in between the quotes. Be careful to never push the .env file to the GitHub.**
4. **run `llm_expensive.py`**



In `load_data.py` and `test_questions.py` you can pick whether to run Porter Stemmer (nltk) or spaCy. Porter Stemmer is a lot faster and spaCy took me about 2 hours to run.


___

# Files in this Repository:
- **Any file ending with .out (except for compare.out), as well as results.txt:** Output produced from running the code during various experiments, or from running code on a specific day.
- **answers.txt:** Used for our previous experiments with a pure tf-idf search engine. It contains all the answers we got for each question using this approach.
- **compare.out:** Used to compare one run of the program to others using diff.
- **cur_question.txt:** Used to keep track of the number of questions answered, and the number of questions that were correct. You can also use it to skip some questions by changing the first number in the file to any number under 100.
- **llm.py:** RUN THIS to run our final code. It includes all interactions with the Large Language Model, as well as calculating evaluation metrics.
- **llm_expensive.py:** Used for experiments with more powerful LLMs (but they cost money).
- **load_data.py:** Used previously to load the Wikipedia pages and the pure tf-idf search engine.
- **nltk_engine.py:** Our benchmark for our project, as well as the original tf-idf search engine that we attempted to use. Similar to the one from Homework 3. Utilizes a Porter stemmer for queries and documents.
- **nltk_selector.py:** A modified tf-idf search engine to re-rank the possible answers to the query provided by the LLM. Also utilizes a Porter stemmer for queries and documents.
- **questions.txt:** The file of Jeopardy questions and answers used for testing.
- **requirements.txt:** Required libraries to install for our project. Some libraries may require a re-installation of NumPy that is lower than 2.0.0 to work.
- **spacy_engine.py:** A tf-idf search engine that uses spaCy to lemmatize words instead of stemming. Otherwise, it is the same as nltk_engine.py, and also performed about the same.
- **test_questions.py:** Run this code to run the benchmark tests. Tests the tf-idf engine with the questions provided.
