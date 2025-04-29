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