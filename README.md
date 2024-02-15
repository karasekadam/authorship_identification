This is a source code for my bachelor thesis.
The thesis is focused on authorship identification, specifically on comparing three different authorship identification algorithms.

The authorship identification algorithms are implemented in BertAA.py, EmailDetective.py and EnsembleModel.py files.

To run prepared experiment run the main.py script.

To run experiment with different setup you can generate experiment sets by changing parameters in the data_loader.py or by changing parameters of the classification models in main.py.

If you wanna use any of the implemeted models on your own datasets, you can just pass your dataset in pandas dataframe format with "author" and "text" column to the experiment function in main.py.

For more information please refer to my thesis https://is.muni.cz/th/uukk2/.
