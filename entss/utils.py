import regex as re
import pysbd
import pandas as pd
import numpy as np
import pkg_resources
from cmdstanpy import write_stan_json, CmdStanModel

def scrub_text(docs, textcol = 'text'):
    """
    Basic text cleaning that removes URLs, non-latin characters except Emoji, and strips white space.

    Args:
        text (str, list of str, or DataFrame): A string or list of strings to be scrubbed.

    Returns:
        str or list: The scrubbed input text(s)
    """
    if isinstance(docs, str):
        text = [docs]
    elif isinstance(docs, pd.DataFrame):
        if textcol not in docs.columns:
            raise ValueError(f"'{textcol}' column not found. If you are passing a DataFrame make sure to specify the name of the text column if it is not 'text'")
        tempdocs = docs.copy() # create a copy so I don't alter the original dataframe
        text = tempdocs[textcol]
    else:
        text = docs
        
    cleaned_texts = []
    for t in text:
        # Remove URLs
        t = re.sub(r'https?://\S+|www\.\S+', '', t)
        # remove non-latin characters except emoji
        t = re.sub(r'[^\p{Latin}\s\p{Emoji}\p{P}]+', '', t)
        # Strip white space
        t = t.strip()
        cleaned_texts.append(t)
    
    if len(cleaned_texts) == 1:
        return cleaned_texts[0]
    
    if isinstance(docs, pd.DataFrame):
        # If a DataFrame was passed, add the cleaned text back to it
        tempdocs[textcol] = cleaned_texts
        return tempdocs     
    return cleaned_texts

def sent_splitter(docs, textcol = 'text'):
    """
    Split documents into sentences. Uses PySBD unless another sentence splitter function is passed.

    Args:
        text (str, list of str, or DataFrame): A string or a list of strings to be split into sentences.

    Returns:
        list: A list of sentences.
    """
    # convert to list if a single string was passed
    if isinstance(docs, str):
        text = [docs]
    elif isinstance(docs, pd.DataFrame):
        if textcol not in docs.columns:
            raise ValueError(f"'{textcol}' column not found. If you are passing a DataFrame make sure to specify the name of the text column if it is not 'text'")
        tempdocs = docs.copy()
        text = tempdocs[textcol]
    else:
        text = docs
        
    # initialize the sentence splitter
    splitter = pysbd.Segmenter(language = 'en', clean = False)
    
    # instantiate list for results
    split_sentences = []
    # loop through string, splitting each and adding to the list
    for t in text:
        sentences = splitter.segment(t)
        split_sentences.append(sentences)

    # If a list of text was passed, and that list was split into sublists, convert to a df with doc numbers
    if isinstance(docs, (list, pd.Series, np.ndarray)):
        if any(isinstance(item, list) for item in split_sentences): 
            # Initialize an empty list to store the data
            data = []
            # Iterate through the elements of the split documents
            for i, item in enumerate(split_sentences):
                # If the element is a list, iterate through its items and append to the data list with doc numbers
                if isinstance(item, list):
                    for sub_item in item:
                        data.append([i, sub_item])
                # If the element is not a list, append it as is
                else:
                    data.append([i, item])
            # Create a DataFrame from the data list with the specified column names
            split_sentences = pd.DataFrame(data, columns=["doc_num", "text"])
            return split_sentences

    # if a dataframe was passed, explode the dataframe and return it
    if isinstance(docs, pd.DataFrame):
        tempdocs[textcol] = split_sentences
        tempdocs = tempdocs.explode(textcol).reset_index(names = 'doc_num')
        return tempdocs        

    # return list rather than nested list if only a string was passed
    if len(split_sentences) == 1:
        return split_sentences[0]
        
    return split_sentences

def synonym_replacer(docs, synonym_dict, case_sensitive=False, textcol = 'text'):
    """
    Replace synonyms in a given text or list of texts based on a dictionary.

    Args:
        docs (str, list of str, or DataFrame): The input text, list of texts, or dataframe where synonyms should be replaced. If passing a dataframe textcol sould be specified.

        synonyms_dict (dict): A dictionary where keys are words and values are synonyms that will replace them.
        Values can be either strings or lists of strings. Regular expressions can also be passed.

        case_sensitive (bool, optional): Determines whether the replacement is case sensitive (default is True). 
        If using case insensitive matching your synonym dictionary should contain all case variations you want to replace.

    Returns:
        str or list: The input text(s) with synonyms replaced according to the provided dictionary.
    """
    # check if the input text is a string or a list of strings
    if not isinstance(synonym_dict, dict):
        raise ValueError("synonym_dict must be a dictionary to replace synonyms")
    if isinstance(docs, str):
        text = [docs]
    elif isinstance(docs, pd.DataFrame):
        if textcol not in docs.columns:
            raise ValueError(f"'{textcol}' column not found. If you are passing a DataFrame make sure to specify the name of the text column if it is not 'text'")
        tempdocs = docs.copy()
        text = docs[textcol]
    else:
        text = docs
    
    replaced_texts = []
    flags = re.IGNORECASE if not case_sensitive else 0
    # Create a regex pattern for identifying words in the text. '(?:\b|(?<=\W))' identifies word boundaries as nonword characters.
    pattern = re.compile(r'(?:\b|(?<=\W))' + r'(?:\b|(?=\W))|(?:\b|(?<=\W))'.join(re.escape(key) for key in synonyms.keys()) + r'(?:\b|(?=\W))', flags = flags)
    # function to replace matches with synonyms
    def replace(match):
        return synonym_dict[match.group(0)]
    
    for t in text:
        replaced_text = t
        # Replace words in the text using the regex pattern
        replaced_text = pattern.sub(replace, t)
        replaced_texts.append(replaced_text)

    if len(replaced_texts) == 1:
        # return the edited string if only a string was passed
        return replaced_texts[0]

    if isinstance(docs, pd.DataFrame):
        tempdocs[textcol] = replaced_texts
        return tempdocs
    
    else:
        # return the list of edited strings if a list was passed
        return replaced_texts

def tag_keywords(docs, keywords, textcol = 'text'):
    """
    Tag documents based on whether or not they contain a keyword.

    Args:
        texts (list or pandas DataFrame): List of texts to search for keywords or a DataFrame containing the text.
        keywords (list): List of keywords to search for. Accepts regular expressions.
        textcol (str, optional): Name of the column containing text if a DataFrame is provided. Default is 'text'.

    Returns:
        pandas DataFrame: DataFrame with presence indicators for each keyword in the texts.
                          'text' column contains the original texts.
                          Each keyword will have a corresponding column indicating presence (1) or absence (0).
    """
    # check if keywords were passed
    if not isinstance(keywords,  (list, pd.Series, np.ndarray)):
        raise ValueError("You must pass a list of keywords as a list, Pandas Series, or Numpy Array.")

    # if a DataFrame is passed, create copy and extract the texts from the specified textcol
    if isinstance(docs, pd.DataFrame):
        if textcol not in docs.columns:
            raise ValueError(f"'{textcol}' column not found. If you are passing a DataFrame make sure to specify the name of the text column if it is not 'text'")
        tempdocs = docs.copy()
        text = tempdocs[textcol]
    else:
        if isinstance(docs,  (list, pd.Series, np.ndarray)):
            # check if the list contains sublists of sentences. If so, we will convert it to a df with doc numbers
            if any(isinstance(item, list) for item in docs): 
                # Initialize an empty list to store the data
                data = []
                # Iterate through the elements of the original list
                for i, item in enumerate(docs):
                    if isinstance(item, list):
                        # If the element is a list, iterate through its items and append to the data list with doc numbers
                        for sub_item in item:
                            data.append([i, sub_item])
                    else:
                        # If the element is not a list, append it as is
                        data.append([i, item])
                # Create a DataFrame from the data list with the specified column names
                tempdocs = pd.DataFrame(data, columns=["doc_num", "text"])
                text = tempdocs['text']
            else:
                text = docs
    
    # compile regular expression patterns for keywords
    keyword_patterns = [re.compile(keyword, re.IGNORECASE) for keyword in keywords]
    
    # initialize a dictionary to store the data for the DataFrame
    data = {'text': text}
    
    # iterate over keywords and their patterns to check for presence in texts
    for keyword, pattern in zip(keywords, keyword_patterns):
        data[keyword] = [1 if pattern.search(string) else 0 for string in text]
    
    # if a DataFrame or list of lists was passed, add keyword columns to it and return
    if isinstance(tempdocs, pd.DataFrame):
        tempdocs = tempdocs.join(pd.DataFrame(data).drop('text', axis=1))
        return tempdocs
    
    # if a list of texts is passed, create a DataFrame from the data dictionary and return
    return pd.DataFrame(data)

def generate_hypoth(targets, dimensions):
    """
    Generate a dictionary of lists with targets as keywords and lists of hypothesis dimensions as values. The returned dictionary can be passed to a Classifier object along with a template to label documents.

    Args:
        targets (list of str): A list of target strings to be inserted into the hypothesis template.
        dimensions (list of str): A list of dimension strings to be inserted into the hypothesis template.
        assign (bool): Whether or not to assign the generated hypotheses to the object's hypoth attribute

    Returns:
        dict: A dictionary with targets as keys and lists of entailment hypotheses as values.
    """
    result_dict = {}
    for target in targets:
        # filled_templates = []
        # for dimension in dimensions:
        #     filled_templates.append(dimension)
        result_dict[target] = [dimension for dimension in dimensions]
    return result_dict


def stanify(data, targets, dimensions, groupids = None, grainsize = None, output_dir = None):
    """
    Convert a DataFrame into a format suitable for Stan heirarchical modeling.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The input DataFrame containing the raw data.

    targets : list of strings
        A list of column names in the DataFrame representing the count of total documents towards a target.

    dimensions : list of strings
        A list of the dimensions along which you wish to scale and are represented in your dataframe. 
        You may pass entire column names or a list of suffixes and all columns with those suffixes will be used. 
        For example, a list of ["support", "oppose"] will use all columns ending in "support" or "oppose".

    groupsids: string
        The name of the column that identifies each row with a group, if one is present.

    groups: int
        The number of groups in the data

    output_dir : If defined will export the data as a JSON to the specified directory

    Returns:
    --------
    dict
        A dictionary containing the processed data suitable for use in Stan modeling.
        The dictionary includes keys for 'y' (observed data), 'J' (number of rows),
        'K' (number of items), 'N' (number of observations), 'jj' (row indices),
        'kk' (item indices), 'X' (total counts for each observation), and 'grainsize' (granularity level).

    Example:
    --------
    df = pd.read_csv('data.csv')
    targets = ['target1', 'target2']
    dimensions = ['dimension1', 'dimension2']
    stan_data = stanify(df, targets, dimensions)

    """
    # Get counts of documents classified for each dimension
    y_cols = [col for col in data.columns if any(col.endswith(word) for word in dimensions)]
    y = data[y_cols]
    # Number of rows, items and observations in the data
    J = y.shape[0] #j rows
    K = y.shape[1] # K items
    N = J*K # N obs
    # flatten y for stan
    y = y.values.flatten(order = 'F')
    y = [int(val) for val in y] # convert to int for stan
    # Compile an array of index numbers for variables so that Stan is matching the right coefficients and counts
    jj = np.tile(np.arange(1,J+1),K) #row for the item
    kk = np.repeat(np.arange(1, K + 1), J) # item for the row
    
    # get counts of total relevant documents for each observation
    x_cols = [col for col in data.columns if any(col.endswith(word) for word in targets)]
    X = data[x_cols].sum(axis = 1)
    # flatten for Stan
    X = np.tile(X, K)
    if groupids is None:
        groupids = np.repeat(1, J)
    else:
        groupids = data[groupids]

    groups = len(set(groupids))

    # Convert to a dictionary
    stan_data = dict(y=y, J = J, K = K, N = N, jj = jj, kk = kk, X = X, gg = groupids, G = groups)
    
    if grainsize:
        stan_data['grainsize'] = grainsize
        
    if output_dir:
        write_stan_json(output_dir, stan_data)
        
    return stan_data


def generate_inits(data, left_cols, right_cols, alphas = None):
    """
    Generate a dictionary of initial ideal point values for Stan. 
    This will evaluate if an observation made more statements associated with the right side of the scale or the left side of the scale. 
    This prevents reflective invariance when scaling so that the values of the scale are aligned with the expected direction.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame.
        left_cols (str or list): The name(s) of the column(s) associated with the left side of the scale for determining initial values.
        right_cols (str or list): The name(s) of the column(s) associated with the right side of the scale for determining initial values.
        alphas (list): An optional list of priors for item coefficients in the model. Items associated with the right side of the scale should have
        a positive value and items associated with the left side of the scale should have a negative value. Values should be listed in the same
        order that they appear in the dataframe.

    
    Returns:
        Dict: A dictionary containing 1's for observations that made more statements associated with the left side of the scale and -1 for observations
        that made more statements associated with the right side of the scale.
    """
    if isinstance(right_cols, str):
        right_cols = [right_cols]
    if isinstance(left_cols, str):
        left_cols = [left_cols]

    right_sum = data[right_cols].sum(axis=1)
    left_sum = data[left_cols].sum(axis=1)

    theta_inits = [1 if rs > ls else -1 if ls > rs else 0 for rs, ls in zip(right_sum, left_sum)]
    inits = dict(theta = theta_inits)
    
    if alphas:
        inits['alpha'] = alphas
        
    return inits

def load_newsletters():
    stream = pkg_resources.resource_stream(__name__, 'data/letters_testset.csv')
    return pd.read_csv(stream)

def load_stan_model(model_type):
    model_file_path = pkg_resources.resource_filename(__name__, f'models/model_{model_type}.stan')
    return CmdStanModel(stan_file=model_file_path)