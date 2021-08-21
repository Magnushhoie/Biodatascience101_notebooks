<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://biodatascience101.github.io">
    <img src="img/biodatascience101.png" alt="Logo">
  </a>

  <h3 align="center">Biodatascience101 notebooks</h3>

  <p align="center">
    Collection of notebooks used in teaching for <a href="https://biodatascience101.github.io">biodatascience101.github.io</a>
  <br>
    Developed by Magnus H. Høie, Andreas Fønss Møller and Tobias Hegelund Olsen.
    <br />
  </p>
</p>



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

## Notebooks

#### [Module 1: Data Exploration - Biomarkers for cerebral Malaria in protein expression data](https://github.com/Magnushhoie/Datascience_notebooks/blob/master/Module_1_Malaria_PandasIO.ipynb)
<img src="https://raw.githubusercontent.com/Magnushhoie/Datascience_notebooks/master/img/module1_logo.png">

This module covers dataset I/O handling, data modelling and normalisation in Pandas, principal component analysis and differential gene expression analysis. The case data is on quantification of relative protein levels in malaria-infected mice from work by [Tiberti, N et al Scientific Reports 2016](https://www.nature.com/articles/srep37871).

---

#### [Module 2: Data Visualisation - Sequence features for thermostability in proteins from extremophiles](https://github.com/Magnushhoie/Datascience_notebooks/blob/master/Module_2_Sequence_DataVisualization.ipynb)
<img src="https://github.com/Magnushhoie/Datascience_notebooks/blob/master/img/module2_logo.png?raw=true" width="504" height="322">

This module covers efficient data processing in Numpy, analysis and visualisation of biological sequences and graphing in Seaborn and Matplotlib. The case data comes from a dataset of 7.7 million bacterial sequences with associated temperature data compiled by the [iGEM Potsdam team for Kaggle](https://www.kaggle.com/igempotsdam/protein-heat-resistance-dataset), collected from the Bacterial Diversity Metadatabase and UniProt.

---

#### [Module 4: Machine-learning classification of benign and malignant tumors](https://github.com/Magnushhoie/Datascience_notebooks/blob/master/Module_4_Machine_learning.ipynb)
<img src="https://github.com/Magnushhoie/Datascience_notebooks/blob/master/img/module_4.png?raw=true" width="500">

Module covering machine-learning dataset pre-processing, exploratory data-analysis, comparison of Scikit-learn prediction model, visualization of model decision-boundaries and evaluation of model performance.

---

#### [Module 5: Protein Sequences Analysis - Analysing antibodies binding SARS-CoV-2](https://github.com/Magnushhoie/Datascience_notebooks/blob/master/Module_5_ProteinClustering.ipynb)
<img src="https://github.com/Magnushhoie/Datascience_notebooks/blob/master/img/module_5.png?raw=true" width="500">

This module covers different analytical methods of protein sequences such as sequence alignments, clustering and logoplots. We will be looking at coronavirus binding antibodies, exploring how they can be clustered using PCA, t-SNE and UMAP, how these clusters can be visualized using logoplots and how to interpret these findings.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Magnushhoie/Datascience_notebooks.svg?style=for-the-badge
[contributors-url]: https://github.com/Magnushhoie/Datascience_notebooks/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Magnushhoie/Datascience_notebooks.svg?style=for-the-badge
[forks-url]: https://github.com/Magnushhoie/Datascience_notebooks/network/members
[stars-shield]: https://img.shields.io/github/stars/Magnushhoie/Datascience_notebooks.svg?style=for-the-badge
[stars-url]: https://github.com/Magnushhoie/Datascience_notebooks/stargazers
[issues-shield]: https://img.shields.io/github/issues/Magnushhoie/Datascience_notebooks.svg?style=for-the-badge
[issues-url]: https://github.com/Magnushhoie/Datascience_notebooks/issues
