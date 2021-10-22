
from string import punctuation

with open("./nepali_stopwords.txt", "r") as f:
    stopwords = f.read()


punct = list(punctuation)



def clean_text_dataset(text):
    cleaned_text = []

    text_ = text.split()

    for text_split in text_:
        if text_split not in stopwords and text_split  not in punct:
            cleaned_text.append(text_split)

    return cleaned_text





