[![Version](https://img.shields.io/badge/version-0.0.1-blue.svg)](https://github.com/your-username/entss)
[![Language](https://img.shields.io/badge/language-Python-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-GNU%20GPLv3-orange.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)
[![Twitter](https://img.shields.io/badge/twitter-%40ML_Burn-blue.svg)](https://twitter.com/ML_Burn)

# Entss: Entailment Classification and Semantic Scaling
 
Entss is a library for inferring political beliefs from text using transformers and Bayesian IRT. It provides tools to simplify the process of cleaning, labeling, and modeling your data so that you don't have to be an expert in NLP or Bayesian statistics, and can save yourself a lot of programming even if you are!

Under the hood, Entss uses zero-shot stance detection via entailment classification to label documents based on the beliefs they express, and estimates the ideology of document authors with semantic scaling. 

Entss is a modular package designed around three classes, the Cleaner(), Classifier(), and Modeler(). Sensible defaults are provided for each class to enable fast inference, but users can also pass their own cleaning functions, models, etc. to the classes.



