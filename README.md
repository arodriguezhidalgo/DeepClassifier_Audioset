# DeepClassifier_Audioset

Code originally proposed to implement approachs for the classification of auditory files from Audioset. This initial code allows to extract and store auditory features, as well as classifying them using the proposed model. 
* An example is depicted in [Main_classifier.ipynb](https://github.com/arodriguezhidalgo/DeepClassifier_Audioset/blob/master/Main_classifier.ipynb).
* New Deep models can be written into the file [Audioset_models.py](https://github.com/arodriguezhidalgo/DeepClassifier_Audioset/blob/master/Audioset_models.py), which can be then called from the main notebook.


## Versioning
v1.1:
* Included an updated version of the classifiers considering binary and multiclass classification.
* New models: LSTM/GRU, Attention, etc. 
* Dependency: [URL](https://github.com/datalogue/keras-attention).


v1.0:
* Original implementation proposed for the classification of two labels, namely `speech`and `music`. 
* Easily modifiable to work with several labels from Audioset.

