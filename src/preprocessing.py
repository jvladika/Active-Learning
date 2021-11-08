import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from bs4 import BeautifulSoup


def strip_html(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()


def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text


def remove_special_characters(text, remove_digits=False):
    if remove_digits:
        pattern = r'[^a-zA-z\s]'
    else:
        pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', text)
    return text


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text


def simple_stemmer(text):
    ps = PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text


def remove_stopwords(text, is_lower_case=True):
    stop_words = set(stopwords.words('english')) 
    tokens = word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [
            token for token in tokens if token not in stop_words]
    else:
        filtered_tokens = [
            token for token in tokens if token.lower() not in stop_words]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def imdb_preprocess(text):
    text = denoise_text(text)
    return text


def imdb_preprocess_cutoff(text, size=200):
    text = text.lower()
    text = denoise_text(text)
    text = remove_special_characters(text)
    text = remove_stopwords(text)
    tokens = text.split(' ')
    end = min(len(tokens), size)
    text = ' '.join(tokens[0:end])
    return text