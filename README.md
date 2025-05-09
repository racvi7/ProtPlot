Welcome to ProteinPlot's documentation!
============================================

Welcome to the documentation for **ProteinPlot**, a University project package for analysing PDB entries in python.

ProteinPlot is a very simple tool for managing **.pdb** extensions in a way, that it can be easily manipulated. There are multiple functions regarding to the demonstration of the protein structure, structural similarities. Due to the utilization of pandas dataframes, multiple different calculations become simply managable even for Python novices.

Introduction
------------

ProteinPlot helps in the analysis of PDB entries. The given files from the databank of rcsb.org is loaded into pandas dataframes and in this way manipulations become easier. In this package a few examples can be found for protein structure visualization.

Description
--------
- PDB entry reading into pandas df:
    ** Easy to use function, directly downloading the protein file from the rcsb.org databank
  
- Structure alignment and comparison
    ** A very basic alignment tool for finding similar protein chains between two pdb entries
    ** Torsion angle calculations between arbitrary defined atoms

- Ramachadran plots
    ** Eye catching Ramachadran plots, showing the distribtions of the secondary structures
  
- 2D and 3D figures
   ** Slice plots for 2D visualization of protein structures
   ** Static and interactive 3D visualizations

Installation
------------
Available on pip, with the command: pip install ProteinPlot

pip project: [ProteinPlot on PyPI](https://pypi.org/project/ProteinPlot/)

For more details see the [Example Google Colab notebook](https://colab.research.google.com/drive/1C3GE2vf-RWxhAlUEDwfVW5a6ehMTbhd_?usp=sharing)

Requirements
-----------

The installation via peep automatically pulls the newest version of the following packages:

** numpy
** pandas
** seaborn
** matplotlib
** plotly

For convinience, if you install this package into a fresh directory, **jupyter** is installed and you have a basic local IDE for managing your work.


Documentation
---------------

https://protplot.readthedocs.io/en/latest/
