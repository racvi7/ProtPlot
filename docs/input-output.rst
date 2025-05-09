Basic Input and Output
=======================

This section documents the essential input and output options for using the protplot library to load and visualize PDB structures.

read_pdb
--------
.. function:: read_pdb(pdb_code, chain=None)

   Downloads and parses a PDB file from the RCSB Protein Data Bank.

   **Parameters:**
   - **pdb_code** (`str`): The 4-character PDB ID (e.g., `'7v7n'`).
   - **chain** (`str`, optional): Specific chain identifier. If not set, returns all chains.

   **Returns:**
   - `pandas.DataFrame`: Structured protein data including atomic coordinates and types.

Saving Figures
--------------

All plotting functions in the protplot package include an optional `figsave` parameter to export the generated figures.

- **plot_projection(..., figsave='filename.png')**
  
  Saves a 2D projection plot to the specified file path.

- **plot_structure_3d(..., figsave='filename.png')**

  Saves the 3D matplotlib figure as a static image.

- **plot_structure_3d_interactive(..., figsave='filename.html')**

  Exports the interactive Plotly figure to an HTML file that can be opened in a browser.

- **plot_ramachandran(...)**

  This function does not currently support a `figsave` parameter directly, but you can save the plot using:

  .. code-block:: python

     plt.savefig("ramachandran_plot.png")
