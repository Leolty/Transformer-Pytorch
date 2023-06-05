import spacy

class Tokenizer:

    def __init__(self, lang='en'):
        if lang == 'en':
            self.nlp = spacy.load('en_core_web_sm')
        
    def tokenize(self, text):

        return [tok.text for tok in self.nlp.tokenizer(text)]
    
    def tokenize_batch(self, batch):

        return [self.tokenize(text) for text in batch]