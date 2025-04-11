How to run:

1. Create a virtual environment (can do this from VScode): python -m venv .env
2. Activate the virtual environment (depends on OS, can do from VScode too)
3. Install dependencies: python -m pip install -r requirements.txt
4. Import the wiki pages data into a folder named `data` and create another folder named `IRdata`
5. Run load_data.py to create .json files 
6. Run test_questions.py to test queries (This counts getting top 10 as correct)
