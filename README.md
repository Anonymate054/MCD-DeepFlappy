<h1 align="center">
  <br>
  <b>DeepFlappy</b>
  <b>By: Luis Enrique</b>
  <br>
</h1>

<p align="center">
  Deep Learning Project</a>.
  <br>
</p>


## Description

Flappy Bird with AI

## Technologies Used

- Python 3 ![Python](https://img.shields.io/badge/Python-3.11-blue)

## Project Objective

Deep Learning Project

## Analysis Section

Your project analysis section

## Prerequisites

- [Anaconda](https://www.anaconda.com/download/) >=4.x
- Optional [Mamba](https://mamba.readthedocs.io/en/latest/)

## Create environment

```bash
conda env create -f environment.yml
activate deepflappy
```

or 

```bash
conda install -c conda-forge mamba
mamba env create -f environment.yml
activate deepflappy
```

## Modules and default modules

To install the default modules located in the `scripts` directory, use the following command:

```bash
pip install --editable .
```

For more information about the user's modules, refer to `install.md`.

## The resulting directory structure

The directory structure of your new project will look something like this (depending on the settings that you choose):

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
├── module         <- User module directory
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-anony-initial-data-exploration`.
│
├── scripts            <- Default modules and scripts for the project
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
```