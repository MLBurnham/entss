import pandas as pd
from cmdstanpy import CmdStanModel
from entss import utils
import warnings

class Scaler:
    """
    A class for scaling from document counts using Stan.

    Args:
        n_chains (int, optional): The number of chains to use in the Stan model. Defaults to 4.

        parallel_chains (int, optional): The number of parallel chains. Defaults to 4.
        
        model_type (str, optional): The type of Stan model to use ('MAP', 'serial', or 'multi'). 
        MAP uses maximum a posteriori to estimate the mode of the distribution and is significantly faster. 
        The serial and multi options use MCMC to estimate the mean.
        
        n_threads (int, optional): The number of threads per chain. Defaults to 1.

    Attributes:
        n_chains (int): The number of chains to use in the Stan model.
        
        parallel_chains (int): The number of parallel chains.
        
        n_threads (int): The number of threads per chain.
        
        model_type (str): The type of Stan model being used ('serial' or 'multi').
        
        model (CmdStanModel): The CmdStanModel instance for the specified model type.
    """
    def __init__(self,
                 n_chains = 4,
                 parallel_chains = 4,
                 model_type = 'MAP',
                 n_threads = 1
                ):
        if model_type not in ['MAP', 'serial', 'multi']:
            raise ValueError("Invalid 'model_type' value. Must be 'MAP', 'serial', or 'multi.'")
        self.n_chains = n_chains
        self.parallel_chains = parallel_chains
        self.n_threads = n_threads
        self.model_type = model_type

        if self.model_type == 'multi' and n_threads == 1:
             warnings.warn("Using the multithreaded model with n_threads set to 1. Either increase the number of threads per chain or use the serial model.", UserWarning)
        # Initialize the model        
        self.model = utils.load_stan_model(model_type)


    def stan_fit(self,
                 data,
                 targets = None,
                 dimensions = None,
                 groupids = None,
                 inits = None, 
                 grainsize = 1, 
                 left_init_cols = None,
                 right_init_cols = None,
                 n_warmup = 1000,
                 n_sample = 2000,
                 output_dir = 'stan_data.json',
                 summary = True,
                 **kwargs):
        """
        Fit the Stan model to the data.

        Args:
            data (dict or pd.DataFrame): The data to be used for fitting the model. If passing a dictionary it should be an output from the stanify() function.
            
            targets (list): The list of target variables. Required if passing a DataFrame.
            
            dimensions (list): The dimensions of the data. Required if passing a DataFrame.

            groupids (list or list like, optional): The group ID associated with each rown in the data if one is being used.
            
            inits (dict or None, optional): The initial values for the model parameters. Defaults to None. Required if passing a dictionary.
            
            grainsize (int, optional): The grain size for multi-threaded MCMC execution. Defaults to 1.
            
            left_init_cols (list or None, optional): List of column names for left initialization in case of a DataFrame input. Defaults to None.
            
            right_init_cols (list or None, optional): List of column names for right initialization in case of a DataFrame input. Defaults to None.
            
            n_warmup (int, optional): The number of warmup iterations for MCMC. Defaults to 1000.
            
            n_sample (int, optional): The number of sampling iterations for MCMC. Defaults to 5000.
            
            output_dir (str, optional): The directory for storing Stan data. Defaults to 'stan_data.json'.
            
            summary (bool, optional): Whether to generate a summary of the fit. Defaults to True.
            **kwargs: Additional keyword arguments for CmdStanModel.sample().

        Returns:
            fit (CmdStanMCMC): The fitted model object.
            summary (CmdStanSummary, optional): The summary of the fit, if summary=True.
        """

        
        if grainsize == 1 and self.model_type == 'multi' :
            warnings.warn("Using the multithreaded model with a grain size of 1 may lead to slow performance.", UserWarning)

        # prepare data if a dictionary is passed
        if isinstance(data, dict):
            if inits is None:  # Check if inits is not provided
                raise ValueError("If 'data' is a dictionary, 'inits' argument must be provided.")
    
            # if a values other than default was passed for grainsize, set the value in the dictionary
            if self.model_type == 'multi':
                if grainsize != 1:
                    data['grainsize'] = grainsize
                # if no value was passed and the dictionary does not contain a grainsize, add it to the dictionary
                if data.get('grainsize', None) is None:
                    data['grainsize'] = grainsize
            # export the dictionary as a json file for stan to read
            write_stan_json(output_dir, data)
            data = output_dir
    
        # prepare data if a dataframe is passed`
        if isinstance(data, pd.DataFrame):
            # if inits were not passed, check if init column names were passed.
            if inits is None:
                if left_init_cols is None or right_init_cols is None:  # Check if inits is not provided
                    raise ValueError("If 'data' is a DataFrame, both left_init_cols and right_init_cols arguments must be provided.")
                inits = utils.generate_inits(data, left_cols = left_init_cols, right_cols = right_init_cols)
            # if inits were passed as a column name, convert to a dictionary
            if isinstance(inits, str):
                inits = dict(theta=list(data[inits]))
                
            # convert df to a dictionary and export as a json
            if self.model_type == 'multi':
                utils.stanify(data = data, targets = targets, dimensions = dimensions, groupids = groupids, grainsize = grainsize, output_dir = output_dir)            
            else:
                utils.stanify(data = data, targets = targets, dimensions = dimensions, groupids = groupids, output_dir = output_dir)            

            data = output_dir
        
        if self.model_type == 'MAP':
            fit = self.model.optimize(data = data,
                                      inits = inits,
                                      jacobian = True,
                                      **kwargs
                                      )
        else:
            fit = self.model.sample(data = data, 
                                    inits = inits,
                                    iter_warmup = n_warmup,
                                    iter_sampling = n_sample,
                                    chains = self.n_chains, 
                                    parallel_chains = self.n_chains, 
                                    threads_per_chain = self.n_threads,
                                    **kwargs
                                )
        if summary:
            summary = fit.summary()
            return fit, summary
        return fit