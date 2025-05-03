"""
Created on Sat Mar 29 11:17:15 2025


----------------------------------------------------------------------------------------------------
THIS IS THE MAIN FILE FOR PROTPLOT PROJECT

The aim of this project is to create a universal way to deal with the '.pdb' extension
and to manage the protein files with pandas.
There are multiple functions regarding to the demonstration of the protein structure.


This file contains the functions given by the package. 
Be aware that this package was made for a University project in Hungary, PPKE ITK.

----------------------------------------------------------------------------------------------------


@authors: domedenes, viktoriaracz, maximilianpoku
"""

#-------------------------------------DEPENDENCIES--------------------------------------------------

#Importing the dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

import urllib.request as urllib2
import io




#-------------------------------------FUNCTIONS-----------------------------------------------------

#---------------------------------------------------------------------------------------------------
#Function - read_pdb()

def read_pdb(pdb_code, chain = None):
    """
    

    Parameters
    ----------
    pdb_code : TYPE str
        The pdb entry ID under which a protein is annoted in the databank rcsb.org
        
    chain : TYPE str, optional
        The given chain identifier which should be returned. Important in alignements. The default is None, when all of the chains are returned.

    Returns
    -------
    out : Pandas df
        An easily managable form of the given protein structure file.
        
    NOTE THAT INTERNET CONNECTION IS NEEDED!

    """
    

    response = urllib2.urlopen(f'https://files.rcsb.org/download/{pdb_code}.pdb') #defining the url where the pdb file is located
    read_response = response.read()
    workable = io.BytesIO(read_response).readlines()
    what = [workable[i].decode('utf-8') for  i in range(len(workable))]
    
    

    temporary_outcome = list()
    for i in range(len(what)):
        temp = what[i]
        temp = temp[0:4]
        
        if temp == 'ATOM':
            
            temporary_outcome.append(what[i]) # if it is an atom then we load it
            
    
    #defining a dictionary based on the character places in the string
    data = {'atom_index':[temporary_outcome[i][6:11].replace(' ', '') for i in range(len(temporary_outcome))],
            'atom_type':[temporary_outcome[i][12:16].replace(' ', '')  for i in range(len(temporary_outcome))],
            'amino_type':[temporary_outcome[i][17:20].replace(' ', '') for i in range(len(temporary_outcome))],
            'chain_index':[temporary_outcome[i][21].replace(' ', '') for i in range(len(temporary_outcome))],
            'amino_index':[temporary_outcome[i][22:26].replace(' ', '') for i in range(len(temporary_outcome))],
            'x':[temporary_outcome[i][31:38].replace(' ', '') for i in range(len(temporary_outcome))],
            'y':[temporary_outcome[i][39:46].replace(' ', '') for i in range(len(temporary_outcome))],
            'z':[temporary_outcome[i][47:54].replace(' ', '') for i in range(len(temporary_outcome))],
            'b_factor':[temporary_outcome[i][61:67].replace(' ', '') for i in range(len(temporary_outcome))]
            }
    
    
    #defining the dataframe
    out = pd.DataFrame(data)
    
    #adjusting the desired format
    numeric_columns = ['atom_index', 'amino_index', 'x', 'y', 'z', 'b_factor']
    out[numeric_columns] = out[numeric_columns].astype('float64')
        
    if chain:
        out = out[out.chain_index == chain]
    
    else:
        print(f'No chain was defined, note that all posibble chains in the entry {pdb_code} is returned')
        
    return out

#example = read_pdb('6vxx', chain = 'A')
#example2 = read_pdb('6vxx')

#---------------------------------------------------------------------------------------------------
#Function - plot_projection()

def plot_projection(protein_df, colorcode = 'default', atoms = 'all',
                    aminos = 'all', alpha = 1, marker = 'x', Title = None, figsave = None):
    
    plot_df = protein_df.copy() # no modification to be made
    
    if atoms != 'all':
        plot_df = plot_df[plot_df.atom_type.isin(atoms)] #filter for the given atoms
        
    if aminos != 'all':
        plot_df = plot_df[plot_df.amino_type.isin(aminos)] #filter for the given amino acids
        
    if colorcode == 'default':
        ccode = {'C' : 'black', 'N' : 'blue', 'O' : 'red', 'S' : 'yellow'} #dict for coloring
    
    #Defining the main type of the atoms for coloring
    defined_atoms = plot_df.atom_type.values
    defined_atoms = [element[0] for element in defined_atoms]
    plot_df['defined_atoms'] = defined_atoms
    datoms = np.unique(plot_df.defined_atoms)
    
    
    dpg = plot_df.groupby('defined_atoms')
    
    fig, axes = plt.subplots(nrows = 1, ncols = 3,figsize = (18,6)) #opening the figure
    for dat, plotter in dpg:
        
        axes[0].scatter(plotter.x.values[0], plotter.y.values[0], alpha = 1,
                        marker = marker, color = ccode[dat], label = dat) #just because the meaningful legend

        axes[0].scatter(plotter.x.values, plotter.y.values, alpha = alpha,
                        marker = marker, color = ccode[dat])
        axes[0].set_xlabel('X', fontsize = 14)
        axes[0].set_ylabel('Y', fontsize = 14)
        
        
        axes[1].scatter(plotter.x.values, plotter.z.values, alpha = alpha,
                        marker = marker, color = ccode[dat])
        axes[1].set_xlabel('X', fontsize = 14)
        axes[1].set_ylabel('Z', fontsize = 14)
        
        
        axes[2].scatter(plotter.z.values, plotter.y.values, alpha = alpha,
                        marker = marker, color = ccode[dat])
        axes[2].set_xlabel('Z', fontsize = 14)
        axes[2].set_ylabel('Y', fontsize = 14)
    
    
    
    fig.legend(fontsize = 16)
    
    if Title:
        fig.suptitle(Title, fontsize = 22)
    
    if figsave:
        plt.savefig(figsave, dpi = 200)
        
    plt.show()
        
        
        
        
    
#plot_projection(example2, alpha = 0.55,aminos = ['CYS'], atoms = ['CA', 'SG'], 
                #Title = 'Test projection plot', figsave = 'test.jpg')

#---------------------------------------------------------------------------------------------------
# Function - plot_structure_3d()

def plot_structure_3d(
    protein_df1,
    protein_df2=None,
    atoms1='all', atoms2='all',
    aminos1='all', aminos2='all',
    colorcode1='default', colorcode2='default',
    alpha1=1.0, alpha2=1.0,
    marker1='o', marker2='x',
    marker_size1=20, marker_size2=20,
    fig_size=(10, 10),
    Title=None, figsave=None
):
    """
    Static 3D scatter plot of one or two protein structures using matplotlib.

    Parameters
    ----------
    protein_df1, protein_df2 : pd.DataFrame
        DataFrames from `read_pdb()` for the proteins to be plotted.
    atoms1, atoms2 : list or 'all'
        List of atom types to include (e.g., ['CA']) or 'all'.
    aminos1, aminos2 : list or 'all'
        List of amino acids to include (e.g., ['CYS']) or 'all'.
    colorcode1, colorcode2 : dict, 'default', or 'b_factor'
        'default' = color by atom type,
        dict = manual color mapping,
        'b_factor' = gradient color by B-factor.
    alpha1, alpha2 : float
        Transparency of markers (0.0–1.0).
    marker1, marker2 : str
        Matplotlib marker symbols (e.g., 'o', '^').
    marker_size1, marker_size2 : float
        Size of the markers.
    fig_size : tuple
        Size of the figure in inches (width, height).
    Title : str
        Title of the plot.
    figsave : str or None
        If provided, path to save the figure (e.g., 'plot.png').

    Returns
    -------
    None
        Displays a 3D scatter plot of the protein structure(s).
    """

    def plot_single_structure(ax, df, atoms, aminos, colorcode, alpha, marker, marker_size, label_prefix):
        plot_df = df.copy()

        if atoms != 'all':
            plot_df = plot_df[plot_df.atom_type.isin(atoms)]

        if aminos != 'all':
            plot_df = plot_df[plot_df.amino_type.isin(aminos)]

        use_bfactor_gradient = (colorcode == 'b_factor')

        if use_bfactor_gradient:
            scatter = ax.scatter(
                plot_df.x.values,
                plot_df.y.values,
                plot_df.z.values,
                c=plot_df.b_factor.values,
                cmap='coolwarm',
                alpha=alpha,
                marker=marker,
                s=marker_size,
                label=f"{label_prefix}: B-factor"
            )
            plt.colorbar(scatter, ax=ax, label='B-factor')
        else:
            if colorcode == 'default':
                ccode = {'C': 'black', 'N': 'blue', 'O': 'red', 'S': 'yellow'}
            else:
                ccode = colorcode

            plot_df['defined_atoms'] = plot_df.atom_type.str[0]
            dpg = plot_df.groupby('defined_atoms')

            for dat, plotter in dpg:
                if dat in ccode:
                    ax.scatter(
                        plotter.x.values,
                        plotter.y.values,
                        plotter.z.values,
                        color=ccode[dat],
                        label=f"{label_prefix}: {dat}",
                        alpha=alpha,
                        marker=marker,
                        s=marker_size
                    )

    # Create plot
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')

    plot_single_structure(ax, protein_df1, atoms1, aminos1, colorcode1, alpha1, marker1, marker_size1, 'P1')

    if protein_df2 is not None:
        plot_single_structure(ax, protein_df2, atoms2, aminos2, colorcode2, alpha2, marker2, marker_size2, 'P2')

    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_zlabel('Z', fontsize=14)

    if Title:
        ax.set_title(Title, fontsize=18)

    ax.legend(title='Atom Type')
    
    if figsave:
        plt.savefig(figsave, dpi=300)

    plt.tight_layout()
    plt.show()

#example usage:

#example2 = protplot.read_pdb('6vxx')
#example = protplot.read_pdb('7v7n')

#plot_structure_3d(example, example2, atoms1=['CA'], atoms2=['CA'], colorcode1={'C': 'green'}, colorcode2={'C': 'red'}, alpha1=0.9, alpha2=0.5, marker1='o', marker2='^',marker_size1=40, marker_size2=60)

#---------------------------------------------------------------------------------------------------
#Function - plot_structure_3d()
# This function is an interactive version of the plot_structure_3d function using Plotly.


def plot_structure_3d_interactive(
    protein_df1,
    protein_df2=None,
    atoms1='all', atoms2='all',
    aminos1='all', aminos2='all',
    colorcode1='default', colorcode2='default',
    alpha1=1.0, alpha2=1.0,
    marker1='circle', marker2='circle',
    marker_size1=5, marker_size2=5,
    fig_width=1000, fig_height=1000,
    Title=None, figsave=None
):
    """
    Interactive 3D scatter plot of one or two protein structures using Plotly.

    Parameters
    ----------
    protein_df1, protein_df2 : pd.DataFrame
        DataFrames from `read_pdb()` for the proteins to be plotted.
    atoms1, atoms2 : list or 'all'
        List of atom types to include (e.g., ['CA']) or 'all'.
    aminos1, aminos2 : list or 'all'
        List of amino acids to include (e.g., ['GLY']) or 'all'.
    colorcode1, colorcode2 : dict, 'default', or 'b_factor'
        'default' = atom type coloring,
        dict = manual color mapping,
        'b_factor' = color by B-factor gradient.
    alpha1, alpha2 : float
        Marker opacity (0.0 to 1.0).
    marker1, marker2 : str
        Plotly marker symbol (e.g., 'circle', 'square').
    marker_size1, marker_size2 : float
        Size of the marker spheres.
    fig_width, fig_height : int
        Size of the interactive plot in pixels.
    Title : str
        Optional title for the plot.
    figsave : str or None
        If provided, saves the plot as an interactive HTML file.

    Returns
    -------
    None
        Opens an interactive 3D protein visualization.
    """

    fig = go.Figure()

    def add_structure_to_plot(df, atoms, aminos, colorcode, alpha, marker, marker_size, label_prefix):
        plot_df = df.copy()

        if atoms != 'all':
            plot_df = plot_df[plot_df.atom_type.isin(atoms)]

        if aminos != 'all':
            plot_df = plot_df[plot_df.amino_type.isin(aminos)]

        use_bfactor_gradient = (colorcode == 'b_factor')
        if not use_bfactor_gradient:
            if colorcode == 'default':
                ccode = {'C': 'black', 'N': 'blue', 'O': 'red', 'S': 'yellow'}
            else:
                ccode = colorcode
            plot_df['defined_atoms'] = plot_df.atom_type.str[0]

        if use_bfactor_gradient:
            fig.add_trace(go.Scatter3d(
                x=plot_df.x,
                y=plot_df.y,
                z=plot_df.z,
                mode='markers',
                marker=dict(
                    size=marker_size,
                    color=plot_df.b_factor,
                    colorscale='RdBu_r',
                    colorbar=dict(title='B-factor'),
                    opacity=alpha,
                    symbol=marker
                ),
                name=f"{label_prefix}: B-factor"
            ))
        else:
            for atom_type in plot_df['defined_atoms'].unique():
                subset = plot_df[plot_df['defined_atoms'] == atom_type]
                fig.add_trace(go.Scatter3d(
                    x=subset.x,
                    y=subset.y,
                    z=subset.z,
                    mode='markers',
                    marker=dict(
                        size=marker_size,
                        color=ccode.get(atom_type, 'gray'),
                        opacity=alpha,
                        symbol=marker
                    ),
                    name=f"{label_prefix}: {atom_type}"
                ))

    # Add first structure
    add_structure_to_plot(protein_df1, atoms1, aminos1, colorcode1, alpha1, marker1, marker_size1, 'P1')

    # Optional second structure
    if protein_df2 is not None:
        add_structure_to_plot(protein_df2, atoms2, aminos2, colorcode2, alpha2, marker2, marker_size2, 'P2')

    fig.update_layout(
        title=Title or '3D Protein Structure(s)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        legend_title='Atom Type',
        width=fig_width,
        height=fig_height
    )

    if figsave:
        fig.write_html(figsave)

    fig.show()
    

#example usage:
#example2 = protplot.read_pdb('6vxx')
#example = protplot.read_pdb('7v7n')

#plot_structure_3d_interactive(example, example2, atoms1=[CA], atoms2=['CA'],colorcode1='b_factor', colorcode2={'C': 'red'},marker_size1=4, marker_size2=6,alpha1=0.9, alpha2=0.4)


    
    
    
    
        
    
