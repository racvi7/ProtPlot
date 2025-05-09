Structure Comparison
====================

Protein_alignment
-----------------

Calculate RMSD between two protein structures by comparing Cα atoms. Compare specific chains or create a chain-to-chain heatmap.

**Note:** The aligned chains can also be visualized directly using the `plot_structure_3d` and `plot_structure_3d_interactive` functions by passing two structures.  
See :doc:`visualization` for full details on these visualization functions.

Parameters:

- **df1, df2**: DataFrames from :func:`read_pdb()`.
- **chains**: List of two chain IDs or `'Combo_matrix'` for full heatmap.

Examples:

.. code-block:: python

    # Compare specific chains
    df1 = protplot.read_pdb('7lyn')
    df2 = protplot.read_pdb('7lym')
    protplot.Protein_alignment(df1, df2, chains=['A', 'A'])

    # Compare all chains (matrix)
    protplot.Protein_alignment(df1, df2, chains='Combo_matrix')


extract_phi_psi
---------------

Extract φ and ψ backbone torsion angles from atomic coordinates stored in a DataFrame.

Parameters:

- **df**: DataFrame from :func:`read_pdb()` with N, CA, and C atoms.

Returns:

- Array of (phi, psi) angle pairs.

Examples:

.. code-block:: python

    df = protplot.read_pdb('7v7n')
    angles = protplot.extract_phi_psi(df)
