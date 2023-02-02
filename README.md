# Modular Clinical Decision Support Networks (MoDN) 
This is the repository accompanying the *Modular Clinical Decision Support Networks (MoDN) Updatable, Interpretable, and Portable Predictions for Evolving Clinical Environments* paper.
## Abstract

Modular Clinical Decision Support Networks (MoDN) is a novel decision tree composed of feature-specific neural network modules. It creates dynamic personalised representations of patients, and can make multiple predictions of diagnoses and features, updatable at each step of a consultation.
The model is validated on a real-world Clinical Decision Support System  (CDSS) derived dataset, comprising 3,192 pediatric outpatients in Tanzania. 

MoDN significantly outperforms 'monolithic' baseline models (which take all features at once at the end of a consultation) with a mean macro $F_1$ score across all diagnoses of $0.749$ vs  $0.651$ for logistic regression  and  $0.620$ for multilayer perceptron  ($p<0.001$).

To test collaborative learning between  imperfectly interoperable (IIO) datasets, we create subsets with various percentages of feature overlap and port a MoDN model trained on one subset to another. Even with only 60\% common features, fine-tuning a MoDN model on the new dataset or just making a composite model with MoDN modules matched the ideal scenario of sharing data in a perfectly interoperable setting. 

MoDN integrates into consultation logic by providing interpretable continuous feedback on the predictive potential of each question in a CDSS questionnaire. The modular design allows it to compartmentalise training updates to specific features and collaboratively learn between IIO datasets without sharing any data.

## Plots
The experiments were run on python version 3.8.10. The **data** folder must contain the data (link to anonymized data https://zenodo.org/record/400380#.Yug5kuzP00Q). The **models** folder contains the scripts used to run the different experiments and produce the plots. 

The script **main.py** calls the preprocessing pipeline on the data and trains the model either performing 5 times 2-fold CV (saving the different metrics) or just training a single model. 

The script **iio_training.py** performs the IIO experiments (i.e. compartmentalization and fine tuning). The different models and performance scores are saved to the **updated_centralized** folder.

After having run both these files, one can run **statistical_tests.py** to produce the plots (uses saved metrics by the two previous scripts). 

Run **unsupervised_learning.py** to produce the plot with the clustering of the state. Metrics and performance scores are saved to the *saved_objects* folder and plots to the *saved_plots* folder.


## Other files
**baselines.py** contains the functions to compute the KNN and logistic regression baselines. 

**dataset_generation.py** puts the data in shape to be used by the models. 

**distributed_training_parameters.py** contains the parameters used by the **distributed_training.py** file. 

**graph_functions.py** contains many functions to produce some plots. 

**modules.py** contains the module and state definitions. 

**training_procedures_epoct.py** contains the training and testing processes for the model. 

**utils_distributed.py** contains some utlitary functions for the compartmentalisation and fine-tuning experiments. 

**utils_epoct.py** contins utilitary functions specific to the epoct data and **utils.py** general utilitary functions. 

## Reproducing the results
To reproduce the results reported in *Modular Clinical Decision Support Networks (MoDN) Updatable, Interpretable, and Portable Predictions for Evolving Clinical Environments*, install the necessary dependencies using:

`sudo apt install texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super`

`pip install -r requirements.txt` (from the root directory)

Create a **data** folder and download the anonymized data from https://zenodo.org/record/400380#.Yug5kuzP00Q. Then run the different scripts as described in the **Plots** paragraph.





