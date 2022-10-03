"""
Preprocess tweets.
"""

import re
import string
import nltk
from nltk import word_tokenize
# nltk.download('punkt')
from emoji import demojize

EMOTICONS = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3', ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L',
    ':<', ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\',
    ':-c', ':c', ':{', '>:\\', ';('
])

PUNCT = set(string.punctuation + '@')


def split_hashtag(tweet):
    '''
    take a string, find all hashtags, remove "#" and split hashtag into words
    '''
    tweet_toks = tweet.split(" ")
    final_tweet_toks = []
    for i in range(len(tweet_toks)):
        if tweet_toks[i].startswith("#"):
            hashtag = tweet_toks[i][1:]
            split_hashtag = re.findall('[0-9]+|[A-Z][a-z]+|[A-Z][A-Z]+|[a-z]+',
                                       hashtag)
            final_tweet_toks.extend(split_hashtag)
        else:
            final_tweet_toks.append(tweet_toks[i])
    tweet = " ".join(final_tweet_toks)
    return tweet


def preprocess_tweet(tweet):
    """
    preprocess one string (tweet):
    1. remove URLs
    2. replace all mentions with "@user"
    3. remove or split hashtags
    4. emojis to description
    5. to lower case
    6. remove punctuations
    7. remove non-ascii
    8. remove emoticons

    """
    # remove URLs
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',
                   'http', tweet)
    tweet = re.sub(r'http\S+', 'http', tweet)
    # remove usernames
    tweet = re.sub('@[^\s]+', '@user', tweet)
    # remove the # in hashtag and split hashtags
    # tweet = split_hashtag(tweet)
    # remove hashtags
    tweet = re.sub('#[^\s]+', '', tweet)
    # emojis to description
    tweet = demojize(tweet)
    # convert text to lower-case
    tweet = tweet.lower()
    #Remove any other punctuation
    tweet = ''.join([char for char in tweet if char not in PUNCT])
    #Remove non-ascii characters
    tweet = re.sub(r'[^\x00-\x7F]+', '', tweet)
    #Remove stopwords and emoticons from final word list
    # stop_words = set(stopwords.words('english'))
    tweet = ' '.join([w for w in word_tokenize(tweet) if w not in EMOTICONS])

    return tweet


def preprocess(corpus):
    """
    preprocess a list of strings (tweets)
    """
    outcorpus = []
    for text in corpus:
        outcorpus.append(preprocess_tweet(text))
    return outcorpus
