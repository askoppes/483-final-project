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

In `load_data.py` and `test_questions.py` you can pick whether to run Porter Stemmer (nltk) or spaCy. Porter Stemmer is a lot faster and spaCy took me about 2 hours to run.