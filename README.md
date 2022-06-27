# Supervised Learning
## Project: Rank Nursery-School Applications

---

## Project Description:

In this project, we will employ several supervised algorithms of our choice to rank applications for nursery schools accurately. we will then choose the best candidate algorithm from preliminary results and further optimize this algorithm to best model the data. our goal with this implementation is to construct a model that accurately predicts the ranking of nursery-school applications. Nursery Database was derived from a hierarchical decision model originally developed to rank applications for nursery schools. It was used for several years in the 1980s when there was excessive enrollment in these schools in Ljubljana, Slovenia, and the rejected applications frequently needed an objective explanation. The dataset for this project originates from the UCI Machine Learning Repository.

---

### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project.

### Code

Template code is provided in the `Nursery Model.ipynb` notebook file. You will also be required to use the included `visuals.py` Python file and the `nursery.csv` dataset file to complete your work. While some code has already been implemented to get you started, you will need to implement additional functionality when requested to successfully complete the project. Note that the code included in `visuals.py` If you are interested in how the visualizations are created in the notebook, please feel free to explore this Python file.

### Run

In a terminal or command window, navigate to the top-level project directory `Nursery Model/` (that contains this README) and run one of the following commands:

```bash
ipython notebook Nursery Model.ipynb
```  
or
```bash
jupyter notebook Nursery Model.ipynb
```

This will open the iPython Notebook software and project file in your browser.

### Data

The modified nursery dataset consists of approximately 13,000 data points, with each datapoint having 8 features. [Nursery.names](https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.names).Within machine-learning, this dataset was used for the evaluation of HINT (Hierarchy INduction Tool), which was proved to be able to completely reconstruct the original hierarchical model. This, together with a comparison with C4.5, is presented in, with the original dataset hosted on [UCI](https://archive.ics.uci.edu/ml/datasets/Nursery).

**Features**
- `parents`: Parents' occupation (usual, pretentious, great_pret)
- `has_nursery`: Child's nursery (proper, less_proper, improper, critical, very_crit)
- `form`: Form of the family (complete, completed, incomplete, foster)
- `children`: Number of children (1, 2, 3, more)
- `housing`: Housing conditions (convenient, less_conv, critical)
- `finance`: Financial standing of the family (convenient, inconv)
- `social`: Social conditions (non-prob, slightly_prob, problematic)
- `health`: Health conditions (recommended, priority, not_recom)

**Target Variable**
- `target`: Nursery-School Applications (very_recom, recommend, not_recom, priority, spec_prior)
