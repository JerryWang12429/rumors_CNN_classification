import pandas as pd
from gensim.models import word2vec

pd.set_option('max_colwidth', 100)


def remove_punctuation(s):
    res = s.replace('\'，\', ', '')
    return res


def creat_corpus(File):
    news = File
    corpus_num = 0
    with open("corpus.txt", 'w', encoding='utf-8') as output:
        for text in news['seg_text']:
            split = eval(text)
            for seg in split:
                output.write(''.join(seg) + '\n')
                corpus_num += 1
                if corpus_num % 1000 == 0:
                    print("已處理 %d " % corpus_num)
    print('total %d word in corpus' % corpus_num)
    output.close()


def word_2vec(filepath):
    sentences = word2vec.LineSentence(filepath)
    model = word2vec.Word2Vec(sentences,
                              size=100,
                              min_count=1,
                              negative=10,
                              window=5)
    model.save('word2vec.model')


if __name__ == "__main__":
    file = pd.read_csv('data/data_seged_monpa.csv')
    file['seg_text'] = file['seg_text'].apply(remove_punctuation)
    creat_corpus(file)
    word_2vec('corpus.txt')
"""     corpus = ""
    split = file['seg_text'].iloc[20]
    split = eval(split)
    for seg in split:
        corpus = corpus + seg + '\n'
    print(corpus) """
