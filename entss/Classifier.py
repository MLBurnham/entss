from transformers import pipeline
import pandas as pd
import torch
import warnings
from entss import utils
from tqdm import tqdm
import numpy as np

class Classifier:
    """
    A class for text classification using a HuggingFace zero-shot pipeline.

    This class provides methods for generating hypotheses, classifying text data,
    and cleaning results.

    Attributes:
        model_name (str): The name of the model to use for classification.
        hypoth (dict): A dictionary of target hypotheses.
        targets (list of str): A list of target strings.
        dimensions (list of str): A list of dimension strings.
        template (str): A string template with placeholders for the target and dimension.
        classifier: The HuggingFace pipeline object for classification.

    Methods:
        generate_hypoth: Generate target hypotheses.
        classify_text: Classify text data based on targets and labels.
    """
    def __init__(self,
                 model_name = "sileod/deberta-v3-base-tasksource-nli",
                 device = None, 
                 batch_size = 1,
                 classifier = None,
                 hypoth = None,
                 targets = None,
                 dimensions = ["supports", "opposes", "is neutral towards"],
                 template = "The author of this text {{dimension}} {{target}}"):
        self.model_name = model_name
        self.batch_size = batch_size
        self.classifier = classifier
        self.hypoth = hypoth
        self.targets = targets
        self.dimensions = dimensions        
        self.template = template

        # if no device was specified, check if GPU is available. Use CPU otherwise
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            print("GPU found, using GPU to classify text.")
            self.device = torch.device("cuda:0")
        else:
            print("No GPU found, using CPU to classify text.")
            self.device = torch.device("cpu")

        # if targets were passed but a hypothesis dictionary was not passed, generate a hypothesis dictionary
        if targets is not None and hypoth is None:
            self.generate_hypoth()
        
        # if a model name was passed but a classifier was not, initialize a classifier            
        if classifier is None:
            self.classifier = pipeline("zero-shot-classification", 
                                        model = self.model_name, 
                                        device = self.device
                                       )
    def generate_hypoth(self, targets = None, dimensions = None, assign = True):
        """
        Generate a dictionary of lists with targets as keywords and lists of filled templates as values. The returned dictionary can be passed to a Classifier object along with a template to label documents.
    
        Args:
            targets (list of str): A list of target strings to be inserted into the template. Will inherit object targets if none are passed.
            dimensions (list of str): A list of dimension strings to be inserted into the template. Will inherit object dimensions if none are passed
            assign (bool): Whether or not to assign the generated hypotheses to the object's hypoth attribute
    
        Returns:
            dict: A dictionary with targets as keys and lists of entailment hypotheses as values.
        """
        # inherit object arguments if none are passed
        targets = targets or self.targets
        dimensions = dimensions or self.dimensions
        # call on the generate hypoth function to generate the dictionary
        hypoth = utils.generate_hypoth(targets, dimensions)
        # assign to the object attribute if specified
        if assign:
            self.hypoth = hypoth
        return hypoth

    def label(self, data, textcol = 'text', hypoth = None, template = None, aggregate_on = None):
        """
        Classify text data based on given targets and labels using a HuggingFace pipeline.

        Args:
            data (pandas.DataFrame or list): Input data to classify.
            textcol (str): Name of the column containing text if passing a DataFrame.
            hypoth (dict): A dictionary of targets(keys) and entailment hypotheses(values). If None the object's hypothesis dictionary is used.
            template(str): The hypothesis to be used for entailment classification. If None is passed then it inherits the template from the classifier object.
            aggregate_on(str): If a column name is passed, group by that column and sum document counts. Typically this would be the author column, and thus a dataframe with the counts of how many times an author expressed a stance is returned. If None, a dataframe of labeled documents will be returned. Individual document labels are not preserved if aggregating. Because compute times can be long, it is thus recommended to leave this argument as None unless you are positive of your results.

        Returns:
            pandas.DataFrame: A DataFrame containing classification results.
        """
        hypoth = hypoth or self.hypoth 
        template = template or self.template
        # prep the template for passing to the classifier
        template = template.replace('{{dimension}}', '{}')
        # extract list of targets from the hypotheses
        targets = list(hypoth.keys())
        # create flag for multilabel classification to differentiate label handeling for binary and multilabel cases
        if len(self.dimensions) > 1: 
            multilabel = True

        # Processing if a list like was passed, convert to a df
        if isinstance(data, (list, pd.Series, np.ndarray)):
            warnings.warn("Passing a list like as input data. Documents will be classified for all targets. If this is not intended pass a dataframe with columns indicating relevant targets.", UserWarning)
            tempdf = pd.DataFrame({textcol: data})    
        # if a dataframe was passed, create a copy
        elif isinstance(data, pd.DataFrame):
            missing_targets = [target for target in targets if target not in data.columns]
            if missing_targets:
                raise ValueError(f"DataFrame is missing columns for the following targets: {', '.join(missing_targets)}. If you pass a dataframe it must also have columns indicating which documentes relate to which targets. If your dataframe does not have this, you can use the cleaning class to prepare your data or pass your data as a list.")
            tempdf = data.copy()
        else:
            raise ValueError("Unsupported data type. Use either a list or a pandas DataFrame as input.")

        # Initialize a list 
        label_dfs = []
        # loop through each target, populating the template and classifying relevant documents
        for target in tqdm(targets, desc="Classifying text"):
            filled_template = template.replace("{{target}}", target)
            candidate_labels = hypoth[target]

            # Filter rows where the target is true if a dataframe with labels was passed
            if isinstance(data, pd.DataFrame):
                target_rows = tempdf[target] == 1
                text = tempdf.loc[target_rows, textcol]
            # take all rows of text if a list like was passed
            else:
                text = tempdf[textcol]

            # Use classifier to get predictions
            res = self.classifier(list(text), candidate_labels, hypothesis_template=filled_template, multi_label=False, batch_size = self.batch_size)
            self.classifier.call_count = 0 # prevents warnings

            # extract labels from the results
            # if using multi-label classification extract the most likely candidate label
            if multilabel: 
                labels = [label['labels'][0] for label in res]
                 # Add results to dataframe
                if isinstance(data, pd.DataFrame):
                    # Need to cast the column as an object first to avoid a warning
                    tempdf[f"{target}_lab"] = np.NaN
                    tempdf[f"{target}_lab"] = tempdf[f"{target}_lab"].astype('object')         
                    
                    tempdf.loc[target_rows, f"{target}_lab"] = labels
                else:
                    tempdf[f"{target}_lab"] = labels
                # Convert to one-hot encoding
                dums = pd.get_dummies(tempdf[f"{target}_lab"], prefix=target, dtype=float)
                # get_dummies() won't create a column if none of a label was found.
                # Check to see if all columns are present and if not, 
                # add a column of zeros to indicate that the label does not appear in the data.
                for label in labels:
                    if f"{target}_{label}" not in dums:
                        dums[f"{target}_{label}"] = 0
                label_dfs.append(dums)
            # if using binary entialment classification extract label
            else:                
                # if probability of entailment > 0.5, return 1, else 0
                labels = [1 if label['scores'][0] > 0.5 else 0 for label in res]
                # convert labels to a series and append to list of results
                labels = pd.Series(labels, name = f"{target}_{self.dimensions[0]}")
                label_dfs.append(labels)

        # Concatenate labels into a single df
        label_df_concat = pd.concat(label_dfs, axis=1)

        # Add labels to df and drop original label columns
        tempdf = pd.concat([tempdf, label_df_concat], axis=1)
        # if multiple labels, drop the original label columns
        if multilabel:
            tempdf.drop([f"{target}_lab" for target in targets], axis=1)

        if aggregate_on is not None:
            count = tempdf.groupby(aggregate_on).size().reset_index(name='Count')['Count']
            tempdf = tempdf.drop(textcol, axis = 1).groupby(aggregate_on).sum().reset_index()
            tempdf['doc_num'] = count
        return tempdf