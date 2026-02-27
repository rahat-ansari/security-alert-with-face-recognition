Jupyter Notebook extension: "Run object tracking cells"

What this does
- Adds a toolbar button to classic Jupyter Notebook labeled `Run object tracking cells`.
- When clicked, the extension searches for code cells tagged with the metadata tag `object_tracking` and executes them in place.
- If no tagged cells are found, it will execute all code cells and show a modal notice.

Files
- `static/main.js` - the nbextension frontend code.
- `install_nbextension.py` - helper script to copy and enable the extension (tries to call `jupyter nbextension` automatically).

Install
1. From the repository root run:
   - `python jupyter_nbextension/install_nbextension.py`
   or manually:
   - `jupyter nbextension install jupyter_nbextension/static --user`
   - `jupyter nbextension enable object_tracking/main --user`

Usage
- Open a classic Jupyter Notebook (not JupyterLab).
- Tag any code cell you want the button to run with the tag `object_tracking` (View -> Cell Toolbar -> Tags).
- Click the toolbar button (play icon) to run the tagged cells.

Notes
- This is for the classic notebook frontend, not JupyterLab.
- If you want a JupyterLab extension instead, tell me and I will scaffold a lab extension.
