import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer

def get_vocabulary(data):
    vectorizer = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
    fitted = vectorizer.fit(data)
    return fitted.vocabulary_

def get_verb_to_idx():
    path = 'verb_classification/data/of500_images_resized'
    pickle_save = 'verb_classification/data/verb_id_fulldata.map'
    files = os.listdir(path)
    verbs = []
    for f in files:
        verbs.append(f.split("_")[0])

    verb2idx = get_vocabulary(verbs)
    reverse = {verb2idx[i]:i for i in verb2idx.keys()}

    final_dict = {'verb2id':verb2idx, 'id2verb':reverse}

    pickle.dump(final_dict, open(pickle_save, "wb"))
    print("done!")


if __name__ == '__main__':
    get_verb_to_idx()