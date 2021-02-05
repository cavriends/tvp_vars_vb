## TVP-BVARs

This repository contains all the code for the thesis that I have written to complete the MSc program in Econometrics & Management Science at the Erasmus University Rotterdam. The title of the thesis is: "_Prior sensitivity in time-varying parameter vector autoregressions_". 

A rough outline of the file structure is as follows:

- **data** - contains all the empricial data for the the thesis
- **excel** - contains all the cross-table analysis of the results
- **notebooks** - contains all the notebooks that were used in the thesis
  - *Runtimes.ipynb* - computes the runtimes for each of the VI-based TVP-BVARs
  - *Sensitivity analysis.ipynb* - runs in parallel the sensitivity analysis of the hyperparameters for several priors
  - *Simulation analysis.ipynb* -  analyses the results for the simulation study and compares the results of the VI-based TVP-BVAR to the MCMC-based TVP-BVAR
  - *Simulation datasets.ipynb* - generates all the datasets necessary for the simulation study according to several DGPs
  - *Simulation study.ipynb* - runs in parallel the simulation study for each of the priors
  - *Visualisations.ipynb* - to visualise the plots for the sensitivity analysis
- **rcode** - contains all the rcode that was used
  - *GelmanRubin.R* - calculates the Gelman-Rubin statistic for the MCMC-based TVP-BVAR and BVAR
  - *functions.R* - all the functions that are necessary to conduct the sensitivity analysis
  - *results.R* - analyses the results of the simulation study and compares MCMC-based TVP-BVAR to the BVAR and VAR
  - *runtimes.R* - calculates the average runtimes for the models that were programmed in R
  - *simple_lr.stan* - a simple OLS in Stan syntax to experiment with ADVI
  - *simulation_study.R* - runs the simulation study in parallel
  - *stan_lr.R* - used to experiment with ADVI and compare viability to MCM
- **sensitivity** - contains all .pkl files that were created in the sensitivity analysis
- **simulations** - contains the simulated datasets and results of the simulation study
  - *datasets* - contains the simulated datasets, mind you that this folder is aroud 1GB.
  - *results* - contains all the .pkl files with the results of the simulation study
- **utils** - the additional functions that are used in Python
  - *data_utils.py* - contains the standardisation, transformation and DGP functions
  - *lstm_models.py* - contains the code for an experiment with an LSTM
  - *lstm_utils.py* - contains the extra utilities necessary for the LSTM
  - *tvp_models.py* - contains the implementation of the VI-based TVP-BVAR for the three different priors
- **visualisations** - contains all the visualisations that are in the thesis

If you're not familiar with Git a good place to start is [here](https://docs.gitlab.com/ee/topics/git/index.html).