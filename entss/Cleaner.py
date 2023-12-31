from entss import utils

class Cleaner:
    """
    A class for text cleaning and preprocessing.

    Args:
        text_scrubber (function, optional): A function for scrubbing text. If None the default scrubber will be used.
        sent_splitter (function, optional): A function for splitting text into sentences. If none PySBD will be used.
        synonym_dict (dict, optional): A dictionary containing synonym replacements. Default is None.
        keywords (list, optional): A list of keywords to tag documents containing those words.

    Attributes:
        sent_splitter (function): The function for sentence splitting.
        text_scrubber (function): The function for text scrubbing.
        synonym_replacer (function): The function for synonym replacement.
        synonym_dict (dict): The dictionary containing synonym replacements.
        keyword_tagger (function): The function for tagging documents that contain keywords.
        keywords (list): The list of keywords passed to the keyword_tagger

    Methods:
        clean(docs, textcol='text'): Cleans and preprocesses input text data.

    """
    def __init__(self, 
                 text_scrubber = None,
                 sent_splitter = None,
                 synonym_dict = None,
                 keyword_list = None
                 ):
        self.sent_splitter = sent_splitter or utils.sent_splitter
        self.text_scrubber = text_scrubber or utils.scrub_text
        self.synonym_replacer = utils.synonym_replacer
        self.synonym_dict = synonym_dict
        self.keyword_tagger = utils.tag_keywords
        self.keyword_list = keyword_list

    def clean(self,
              docs, 
              textcol='text',
              scrub = True,
              split = True,
              synonyms = False,
              keywords = True):
        """
        Cleans and preprocesses input text data.

        Args:
            docs (str, list, or pandas DataFrame): A string, list of strings, or DataFrame containing text to be cleaned.
            textcol (str, optional): The column containing text to be processed if passing a DataFrame. 
                This column will be replaced by the processed text.

        Returns:
            pandas DataFrame or list: Processed and cleaned text data.
        """
        # Replace synonyms if specified
        if synonyms:
            docs = self.synonym_replacer(docs, synonym_dict = self.synonym_dict, textcol = textcol)        
        # Scrub text
        if scrub:
            docs = self.text_scrubber(docs, textcol = textcol)
        # Split sentences
        if split:
            docs = self.sent_splitter(docs, textcol = textcol)
        # Tag keywords
        if keywords:
            docs = self.keyword_tagger(docs, textcol = textcol, keywords = self.keyword_list)

        return docs