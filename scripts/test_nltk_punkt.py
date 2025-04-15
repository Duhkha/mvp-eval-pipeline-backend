import nltk

print("Attempting to use nltk.sent_tokenize...")

try:
    text = "This is the first sentence. This is the second sentence! Is this the third? Yes."
    sentences = nltk.sent_tokenize(text)
    print("\nSuccessfully tokenized sentences:")
    for i, s in enumerate(sentences):
        print(f"{i+1}: {s}")
except Exception as e:
    print(f"\nERROR: Failed during nltk.sent_tokenize.")
    print(e)
    print("\nAttempting to locate 'punkt' data again...")
    try:
        finder = nltk.data.find("tokenizers/punkt")
        print(f"NLTK found 'punkt' resource at: {finder}")
    except LookupError:
        print("NLTK could not find the 'punkt' resource via nltk.data.find.")
    except Exception as find_err:
        print(f"An error occurred trying to find 'punkt': {find_err}")

print("\nScript finished.")