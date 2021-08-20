# Datascience teaching notebooks for [biodatascience101.github.io](https://biodatascience101.github.io)

Collection of notebooks used for teaching data science, developed by Magnus Haraldson Høie, Andreas Fønss Møller and Tobias Hegelund Olsen for Biodatascience101.

## Installation

```bash
# Download
git clone https://github.com/Magnushhoie/Datascience_notebooks
cd Datascience_notebooks

# Install environment
conda env create --name biodatascience101 --file environment.yml

# Activate environment and run Jupyter notebook server
conda activate biodatascience101
jupyter notebook

# Once the server is running, copy paste the given URL into your browser of choice
```

#### [Module 1: Data Exploration - Biomarkers for cerebral Malaria in protein expression data](https://github.com/Magnushhoie/Datascience_notebooks/blob/master/Module_1_Malaria_PandasIO.ipynb)
<img src="https://raw.githubusercontent.com/Magnushhoie/Datascience_notebooks/master/img/module1_logo.png">

This module covers dataset I/O handling, data modelling and normalisation in Pandas, principal component analysis and differential gene expression analysis. The case data is on quantification of relative protein levels in malaria-infected mice from work by [Tiberti, N et al Scientific Reports 2016](https://www.nature.com/articles/srep37871).

#### [Module 2: Data Visualisation - Sequence features for thermostability in proteins from extremophiles](https://github.com/Magnushhoie/Datascience_notebooks/blob/master/Module_2_Sequence_DataVisualization.ipynb)
<img src="https://github.com/Magnushhoie/Datascience_notebooks/blob/master/img/module2_logo.png?raw=true" width="504" height="322">

This module covers efficient data processing in Numpy, analysis and visualisation of biological sequences and graphing in Seaborn and Matplotlib. The case data comes from a dataset of 7.7 million bacterial sequences with associated temperature data compiled by the [iGEM Potsdam team for Kaggle](https://www.kaggle.com/igempotsdam/protein-heat-resistance-dataset), collected from the Bacterial Diversity Metadatabase and UniProt.
