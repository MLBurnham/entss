[![Version](https://img.shields.io/badge/version-0.0.1-blue.svg)](https://github.com/your-username/entss)
[![Language](https://img.shields.io/badge/language-Python-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-GNU%20GPLv3-orange.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)
[![Twitter](https://img.shields.io/badge/twitter-%40ML_Burn-blue.svg)](https://twitter.com/ML_Burn)

# Entss: Entailment Classification and Semantic Scaling
 
Entss is a library for inferring political beliefs from text using transformers and Bayesian IRT. It provides tools to simplify the process of cleaning, labeling, and modeling your data so that you don't have to be an expert in NLP or Bayesian statistics, and can save you a lot of programming even if you are!

Under the hood, Entss uses zero-shot stance detection via entailment classification to label documents based on the beliefs they express, and estimates the ideology of document authors with semantic scaling. 

Entss is a modular package designed around three classes, the Cleaner(), Classifier(), and Modeler(). Sensible defaults are provided for each class to enable fast inference, but users can also pass their own cleaning functions, models, etc. to the classes.

## Installation
Entss relies on PyTorch to label documents and CmdStan to estimate ideal points. It's highly recommended that you set these up in a conda environment to use Entss. If you want to use the Scaler() you will need an install of CmdStan. The [installation instructions for CmdStanPy](https://mc-stan.org/cmdstanpy/installation.html) recommend installing CmdStanPy with Conda, which will automatically install CmdStan. For example, the following command will create a new environment called 'entss' with CmdStanPy and CmdStan installed. 

```bash
conda create -n entss -c conda-forge cmdstanpy
```
If you want to use GPU acceleration for labeling documents, make sure to follow the [PyTorch installation instructions](https://pytorch.org/get-started/locally/) for your version of CUDA and install it to your newly created environment.

Entss can be installed with:
```bash
pip install git+https://github.com/MLBurnham/entss.git#egg=entss
```

## A Minimal Example
```python
import entss as en
df = en.load_newsletters()
df.head()
targets = ['biden', 'trump']
dimensions = ['supports', 'opposes', 'neutral']

# Clean
mrclean = en.Cleaner(keyword_list = targets)
df = mrclean.clean(df, synonyms = False, scrub = True, split = True, keywords = True)

# Label
mturk = en.Classifier(targets = keywords, dimensions = dimensions)
df = mturk.label(df, aggregate_on = 'Last Name')

# Model
banks = en.Scaler()
fit, summary = banks.stan_fit(df, targets = targets, dimensions = ['supports', 'opposes'], left_init_cols = 'trump_opposes', right_init_cols = 'trump_supports', summary = True)
```

## Getting Started

Entss uses zero-shot entailment classification to detect the expressed beliefs in a document. Entailment is a classification task that determines the logical relationship between two sentences. For example, the sentence:
```
Cats like all sausages.
```
Entails the sentence:
```
Cats like salami as a treat.
```
Contradicts the sentence:
```
Cats hate pepperoni.
```
and is neutral to the sentence:
```
Dogs like salami.
```
By pairing documents about political topics (e.g. "I'm voting for Biden in 2024") with statements about the authors belief ("The author of this text supports Biden") we can use entailment classification to infer how many times someone expressed support for a political position in our dataset.

Ents comes with a sample dataset of newsletters sent by members of congress:

```python
import entss as en
df = en.load_newsletters()
df.head()
```
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child jp-OutputArea-executeResult">
<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html" tabindex="0">
<div>
<style scoped="">
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th>text</th>
<th>Last Name</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>News from Congressman John Moolenaar My team ...</td>
<td>Moolenaar</td>
</tr>
<tr>
<th>1</th>
<td>News from Congressman Brian Mast HONORING THE...</td>
<td>Mast</td>
</tr>
<tr>
<th>2</th>
<td>A message from Congresswoman Ann Wagner About...</td>
<td>Wagner</td>
</tr>
<tr>
<th>3</th>
<td>Dear  , Happy New Year! I hope your holiday s...</td>
<td>O’Halleran</td>
</tr>
<tr>
<th>4</th>
<td>Commitment to service is part of the tapestry...</td>
<td>Palazzo</td>
</tr>
</tbody>
</table>
</div>
</div>
</div>
</div>
</div>
</div>

The first step is to determine which issues or "targets" you are interested in (e.g. biden, trump, abortion etc.) and along which dimensions of belief you want to label documents (e.g. support, oppose, neutral).
```python
targets = ['biden', 'trump']
dimensions = ['supports', 'opposes', 'neutral']
```

### Data Cleaning
Entailment classification can require a lot of data preparation. Entss streamlines this process with the Cleaner() class. The class can scrub text of URLs and text artifacts, split sentences, tag documents for targets or keywords they contain, and locate synonyms for your keywords.

In this example, we split newsletters into sentences, scrub the text, and label each sentence that contains a mention of Biden or Trump.

```python
mrclean = en.Cleaner(keyword_list = targets)

df = mrclean.clean(df, synonyms = False, scrub = True, split = True, keywords = True)

df.head()
```
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child jp-OutputArea-executeResult">
<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html" tabindex="0">
<div>
<style scoped="">
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th>doc_num</th>
<th>text</th>
<th>Last Name</th>
<th>biden</th>
<th>trump</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>0</td>
<td>News from Congressman John Moolenaar My team a...</td>
<td>Moolenaar</td>
<td>0</td>
<td>0</td>
</tr>
<tr>
<th>1</th>
<td>0</td>
<td>Starting on December 27 , the new phone number...</td>
<td>Moolenaar</td>
<td>0</td>
<td>0</td>
</tr>
<tr>
<th>2</th>
<td>0</td>
<td>Then, on January 3 , the new office address wi...</td>
<td>Moolenaar</td>
<td>0</td>
<td>0</td>
</tr>
<tr>
<th>3</th>
<td>0</td>
<td>We look forward to continuing that work in the...</td>
<td>Moolenaar</td>
<td>0</td>
<td>0</td>
</tr>
<tr>
<th>4</th>
<td>0</td>
<td>You can also submit your information using thi...</td>
<td>Moolenaar</td>
<td>0</td>
<td>0</td>
</tr>
</tbody>
</table>
</div>
</div>
</div>
</div>
</div>
</div>
The resulting dataframe contains an index indicating which document a sentence belongs to and a binary column indicatig if the sentence mentioned our keywords.

### Document Labeling
The classifier uses a template to generate belief statements about our targets and dimensions that we use for entailment classification. The default template is "The author of this text {{dimension}} {{target}}", but you can also supply your own. In our example, the Classifier() will generate the following statements about Biden:

```
The author of this text supports Biden
The author of this text opposes Biden
The author of this text is neutral towards Biden
```

Each sentence that contains the word Biden will be paired with these statements and a zero-shot entailmetn classifier will choose the best label.

```python
# belief statements are automatically generated when the class is instantiated. 
mturk = en.Classifier(targets = keywords, dimensions = dimensions)
# if you pass a column name to the aggregate_on argument, the classifier will group the data on that column and produce aggregate counts. Otherwise a dataframe with document labels is returned.
df = mturk.label(df, aggregate_on = 'Last Name')
```

You can pass any model from the HuggingFace Hub to the classifier, but it is recommended you use a model trained for zero-shot entailment classification.

We now have a dataframe with how many times each person in our dataset expressed certain opinions about Biden or Trump:

```python
df.head()
```

<div class="jp-OutputArea-child jp-OutputArea-executeResult">
<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html" tabindex="0">
<div>
<style scoped="">
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th>Last Name</th>
<th>doc_num</th>
<th>biden</th>
<th>trump</th>
<th>biden_neutral</th>
<th>biden_opposes</th>
<th>biden_supports</th>
<th>trump_neutral</th>
<th>trump_opposes</th>
<th>trump_supports</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>Adams</td>
<td>5264</td>
<td>0</td>
<td>0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
</tr>
<tr>
<th>1</th>
<td>Allen</td>
<td>11997</td>
<td>1</td>
<td>0</td>
<td>0.0</td>
<td>0.0</td>
<td>1.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
</tr>
<tr>
<th>2</th>
<td>Amodei</td>
<td>22262</td>
<td>1</td>
<td>0</td>
<td>0.0</td>
<td>1.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
</tr>
<tr>
<th>3</th>
<td>Armstrong</td>
<td>12587</td>
<td>0</td>
<td>0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
</tr>
<tr>
<th>4</th>
<td>Auchincloss</td>
<td>6720</td>
<td>1</td>
<td>1</td>
<td>0.0</td>
<td>0.0</td>
<td>1.0</td>
<td>0.0</td>
<td>1.0</td>
<td>0.0</td>
</tr>
</tbody>
</table>
</div>
</div>
</div>
</div>
</div>
</div>

### Scaling
Entss uses a Bayesian IRT model to estimate ideology based on how many of the total documents generated expressed a particular belief. When we instantiate the model we can specify the number of chains, how many parallel chains to run, and whether we want to run a multi-threaded model. Here we will just use the defaults.

You can pass a dataframe to the modeler, or if you have data already formatted for Stan you can pass that as a dictionary. If passing a dataframe you need to specify columns to calculate the initial values. These are columns that you expect people on the left end of the scale to have higher values for (left_init_cols) and columns you expect people on the right end of the scale to have higher values for (right_init_cols). 
```python
banks = en.Scaler()

fit, summary = banks.stan_fit(df, targets = targets, dimensions = ['supports', 'opposes'], left_init_cols = 'trump_opposes', right_init_cols = 'trump_supports', summary = True)
```
stan_fit() will output a stan fit model object you can use to evaluate the model and extract ideal point estimates. If summary = True it will also return a dataframe of parameter estimates, standard deviations, and R-hats.






