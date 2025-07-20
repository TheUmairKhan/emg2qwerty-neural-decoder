To apply post proccessing and do Levenshtein spell-checking you do NOT need to train the model in Colab_setup.ipynb. Spell-checking occurs during test time

Include spellcheck.py inside the emg2qwerty folder alongside all the other relevant Python files.

Replace data.py and lightning.py with the files provided in the submission.

words.txt is the corpus we use for spell-checking.

Before testing the model, in **spellcheck.py line 26, change the file path to where ever you store words.txt** 