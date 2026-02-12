import os
import time
from basic import BasicTokenizer

# open some text and train a vocab of 512 tokens

dirname = os.path.dirname(os.path.abspath(__file__))
taylorswift_file = os.path.join(dirname, "taylorswift.txt")
text = open(taylorswift_file, "r", encoding="utf-8").read()

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

t0 = time.time()
for TokenizerClass, name in zip([BasicTokenizer], ["basic"]):

    # construct the Tokenizer object and kick off verbose training
    tokenizer = TokenizerClass()
    tokenizer.train(text, 512, verbose=True)
    # writes two files in the models directory: name.model, and name.vocab
    prefix = os.path.join("models", name)
    tokenizer.save(prefix)
t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")