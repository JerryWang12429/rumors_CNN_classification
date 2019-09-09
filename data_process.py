import pandas as pd
import jieba
from tqdm import tqdm
import time
import re
import string
import zhon.hanzi

pd.set_option('max_colwidth', 100)
jieba.set_dictionary('dict.txt.big')


def data_prepocess():
    tqdm.write('importing data...')

    # import cofact data
    replies = pd.read_csv('data/replies.csv', lineterminator='\n')
    article_replies = pd.read_csv('data/article_replies.csv', lineterminator='\n')
    articles = pd.read_csv('data/articles.csv', lineterminator='\n')

    print('articles', articles.shape)
    print('article_replies', article_replies.shape)
    print('replies', replies.shape)

    tqdm.write('association..')

    article_has_replied = pd.merge(articles,
                                   article_replies,
                                   how='inner',
                                   left_on=['id'],
                                   right_on=['articleId'])

    # replace opinionated to not_rumor
    article_has_replied['replyType'] = article_has_replied[
        'replyType'].replace(['OPINIONATED'], 'NOT_RUMOR')

    # remove link in text
    article_has_replied['text'] = article_has_replied['text'].str.replace(
        r'http\S+|www.\S+|\n', '', case=False)
    # remove space and special character
    article_has_replied['text'] = article_has_replied['text'].apply(
        lambda x: demoji(x))
    article_has_replied['text'] = article_has_replied['text'].apply(
        remove_punctuation)
    # drop unused column
    article_has_replied = article_has_replied.drop(columns=[
        'userIdsha256_x', 'tags', 'appId_x', 'hyperlinks', 'createdAt_x',
        'updatedAt_x', 'lastRequestedAt', 'userIdsha256_y', 'appId_y',
        'createdAt_y', 'updatedAt_y'
    ])

    print('merge article and replies', article_has_replied.shape)
    print(article_has_replied.columns)

    return article_has_replied


def demoji(text):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        # u"\U000024C2-\U0001F251"
        u"\U00010000-\U0010ffff"
        u"\U00002FF0-\U000033FF"
        u"\U0000FF1A-\U0000FF20"
        "]+",
        flags=re.UNICODE)
    return (emoji_pattern.sub(r'', text))


def remove_punctuation(s):
    for i in s:
        if i in string.punctuation:
            s = s.replace(i, "")
        elif i in string.whitespace:
            s = s.replace(i, "")
        elif i in zhon.hanzi.punctuation:
            s = s.replace(i, "")
    return s


def segmentation(article_csv):
    article = article_csv
    lists = []
    for ids in tqdm(article.index):
        text = article['text'].iloc[ids]
        seg_text = list(jieba.cut(text))
        lists.append(seg_text)
        time.sleep(0.01)

    article.loc[:, 'seg_text'] = lists
    # print(article['seg_text'].head(10))
    article.to_csv('data/data_seged.csv')
    tqdm.write('data csv saved')


if __name__ == "__main__":
    article_label = data_prepocess()
    segmentation(article_label)
    tqdm.write('finished')
