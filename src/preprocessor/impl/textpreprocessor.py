import re
import string
from tqdm import tqdm 

tqdm.pandas()

from src.utility.constant import (
    LOWERCASE_COMPONENT,
    MASK_EMOJI_COMPONENT, 
    MASK_URL_COMPONENT, 
    NORMALIZATION_COMPONENT, 
    REMOVE_HTML_TAG_COMPONENT, 
    REMOVE_PUNCT_COMPONENT,
    EMOJI_MASK,
)
from src.preprocessor.interface import IPreprocessor

class TextPreprocessor(IPreprocessor):
    def __init__(self, component):
        self.component = component

    def preprocess(self, batch):
        if MASK_URL_COMPONENT in self.component:
            batch = batch.progress_apply(lambda row: self.__mask_url(row))

        if REMOVE_HTML_TAG_COMPONENT in self.component:
            batch = batch.progress_apply(lambda row: self.__remove_html_tag(row))

        if MASK_EMOJI_COMPONENT in self.component:
            batch = batch.progress_apply(lambda row: self.__mask_emoji(row))

        if NORMALIZATION_COMPONENT in self.component:
            batch = batch.progress_apply(lambda row: self.__normalization(row))

        if REMOVE_PUNCT_COMPONENT in self.component:
            batch = batch.progress_apply(lambda row: self.__remove_punctuation(row))

        if LOWERCASE_COMPONENT in self.component:
            batch = batch.progress_apply(lambda row: self.__lower(row))
        
        batch = batch.progress_apply(lambda row: self.__remove_excess_whitespace(row))
        return batch

    def available_component(self):
        return [
            MASK_URL_COMPONENT,
            REMOVE_HTML_TAG_COMPONENT,
            MASK_EMOJI_COMPONENT,
            REMOVE_PUNCT_COMPONENT,
            NORMALIZATION_COMPONENT
        ]

    def __mask_url(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    def __remove_html_tag(self, text):
        html=re.compile(r'<.*?>')
        return html.sub(r'',text) 

    def __mask_emoji(self, text):
        emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
        return emoji_pattern.sub(EMOJI_MASK, text)

    def __remove_punctuation(self, text):
        table = str.maketrans('', '', string.punctuation)
        return text.translate(table)

    def __normalization(self, text):
        text = re.sub(r"won\'t", " will not", text)
        text = re.sub(r"won\'t've", " will not have", text)
        text = re.sub(r"can\'t", " can not", text)
        text = re.sub(r"don\'t", " do not", text)
        
        text = re.sub(r"can\'t've", " can not have", text)
        text = re.sub(r"ma\'am", " madam", text)
        text = re.sub(r"let\'s", " let us", text)
        text = re.sub(r"ain\'t", " am not", text)
        text = re.sub(r"shan\'t", " shall not", text)
        text = re.sub(r"sha\n't", " shall not", text)
        text = re.sub(r"o\'clock", " of the clock", text)
        text = re.sub(r"y\'all", " you all", text)

        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"n\'t've", " not have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'d've", " would have", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ll've", " will have", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
        text = re.sub(r"\'re", " are", text)
        return text 

    def __lower(self, text):
        return text.lower()

    def __remove_excess_whitespace(self, text):
        text = re.sub('\s+', ' ', text)
        text = re.sub('[ \t]+$', '', text)
        return text