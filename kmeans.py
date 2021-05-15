from preprocess import preprocess_raw_sent
import nltk
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
import os.path
import re
import time
import os
from shutil import copyfile
import pandas as pd
import math
import multiprocessing
from scipy.spatial import distance_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_a_doc(filename):
    file = open(filename, encoding='utf-8')
    article_text = file.read()
    file.close()
    return article_text   


def load_docs(directory):
	docs = list()  
	for name in os.listdir(directory):
		filename = directory + '/' + name
		doc = load_a_doc(filename)
		docs.append((doc, name))
	return docs

def clean_text(text):
    cleaned = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", ",", "'", "(", ")")).strip()
    check_text = "".join((item for item in cleaned if not item.isdigit())).strip()
    if len(check_text.split(" ")) < 4:
        return 'None'
    return text

def get_sentence_embeddings(sentence, word_embeddings):
    if len(sentence) != 0:
        v = sum([word_embeddings.get(w, np.zeros((50,))) for w in sentence.split()])/(len(sentence.split())+0.001)
    else:
        v = np.zeros((50,))
    return v

def distance_from_centroid(row):
    # if (row['embeddings'] == row['centroid']).all():
    #     res = 0
    # else:
    #     res = 1
    # return 1 - cosine_similarity(row['embeddings'].reshape(1,50), row['centroid'].reshape(1,50))[0,0]
    return distance_matrix([row['embeddings']], [row['centroid']])[0][0]

def start_run(processID, sub_stories, save_path, word_embeddings):
   
    for example in sub_stories:
        start_time = time.time()
        raw_sents = re.split("\n\n", example[0])[1].split(' . ')
        title = re.split("\n\n", example[0])[0] 
        abstract = re.split("\n\n", example[0])[2]

        #remove too short sentences
        df = pd.DataFrame(raw_sents, columns =['raw'])
        df['preprocess_raw'] = df['raw'].apply(lambda x: clean_text(x))
        newdf = df.loc[(df['preprocess_raw'] != 'None')]
        raw_sentences = newdf['preprocess_raw'].values.tolist()
        if len(raw_sentences) == 0:
            continue

        preprocessed_sentences = []
        for raw_sent in raw_sentences:
            preprocessed_sent = preprocess_raw_sent(raw_sent)
            preprocessed_sentences.append(preprocessed_sent)

        preprocessed_abs_sentences_list = []
        raw_abs_sent_list = abstract.split(' . ')
        for abs_sent in raw_abs_sent_list:
            preprocessed_abs_sent = preprocess_raw_sent(abs_sent)
            preprocessed_abs_sentences_list.append(preprocessed_abs_sent)    
        preprocessed_abs_sentences = (" ").join(preprocessed_abs_sentences_list)  

        if len(preprocessed_sentences) < 7 or len(preprocessed_abs_sentences_list) < 3:
            continue

        data = pd.DataFrame(list(zip(raw_sentences, preprocessed_sentences)), columns =['raw_sentence', 'sentence'])

        data['embeddings'] = data['sentence'].apply(lambda x: get_sentence_embeddings(x,word_embeddings))
        
        
        NUM_CLUSTERS = int(0.2*len(raw_sentences))
        X = np.array(data['embeddings'].tolist())

        model = KMeans(n_clusters=NUM_CLUSTERS, init='k-means++', max_iter = 50, n_init=1)
        model.fit(X)

        assigned_clusters = model.labels_
        data['cluster'] = pd.Series(assigned_clusters, index=data.index)
        data['centroid'] = data['cluster'].apply(lambda x: model.cluster_centers_[x])

        data['distance_from_centroid'] = data.apply(distance_from_centroid, axis=1)
        summary=' '.join(data.sort_values('distance_from_centroid',ascending = True).groupby('cluster').head(1).sort_index()['raw_sentence'].tolist())
        
        print("Done!")
        print('time for processing', time.time() - start_time)
        file_name = os.path.join(save_path, example[1] )    
        f = open(file_name,'w', encoding='utf-8')
        f.write(summary)
        f.close()

def multiprocess(num_process, stories, save_path, word_embeddings):
    processes = []
    n = math.floor(len(stories)/5)
    set_of_docs = [stories[i:i + n] for i in range(0, len(stories), n)] 
    for index, sub_stories in enumerate(set_of_docs):
        p = multiprocessing.Process(target=start_run, args=(
            index,sub_stories, save_path[index],word_embeddings))
        processes.append(p)
        p.start()      
    for p in processes:
        p.join()



def main():
    # Setting Variables
    directory = 'full_text_data'
    save_path=['hyp1', 'hyp2', 'hyp3', 'hyp4', 'hyp5']

    if not os.path.exists('hyp1'):
        os.makedirs('hyp1')
    if not os.path.exists('hyp2'):
        os.makedirs('hyp2')
    if not os.path.exists('hyp3'):
        os.makedirs('hyp3')
    if not os.path.exists('hyp4'):
        os.makedirs('hyp4')
    if not os.path.exists('hyp5'):
        os.makedirs('hyp5')

    word_embeddings = {}
    f = open('glove.6B.50d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()

    # list of documents
    stories = load_docs(directory)
    start_time = time.time()
    
    multiprocess(5, stories, save_path, word_embeddings)

    # start_run(1, stories, save_path[0], word_embeddings)

    print("--- %s mins ---" % ((time.time() - start_time)/(60.0*len(stories))))

if __name__ == '__main__':
    main() 