#!/usr/bin/env python3
"""Execute the final artifact notebook with this interpreter; keep outputs private."""
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    notebook = nbformat.read(ROOT / 'notebooks/01_final_project.ipynb', as_version=4)
    nbformat.validate(notebook)
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client = NotebookClient(notebook, km=manager, timeout=180,
                            resources={'metadata': {'path': str(ROOT / 'notebooks')}})
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    destination = ROOT / 'data/private/final_project_executed.ipynb'
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, destination)
    count = sum(cell.cell_type == 'code' for cell in notebook.cells)
    print(f'Final notebook: {count} code cells executed successfully.')
    print('Execution copy: data/private/final_project_executed.ipynb (ignored by Git).')


if __name__ == '__main__':
    main()
