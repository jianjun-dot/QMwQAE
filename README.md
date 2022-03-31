# QMwQAE
This module combines the Combining quantum models with Quantum amplitude estimation to estimate probabilities of stochastic events

# Installation

You can create a new environment for this project using `venv`:

    $ python3 -m venv <env-name>
    $ source ./<env-name>/bin/activate

Alternatively, you can use `conda`:

    (base) $ conda create --name <env-name> 
    (base) $ conda activate <env-name>

We then install the packages using `pip`:
    
    $ pip install -r requirements.txt

To generate the full documentation, this project uses the `sphinx` package. The full documentation is prepared using `auto-doc` function in `sphinx`. Navigate to the `docs` folder. If there are no `rst` files, use the following command to generate the relevant files:

    $ sphinx-apidoc -f -o source ../QMwQAE

This should generate the `rst` files for the documentation. The documentation can then be generated using the following commmand:

    $ make html

The documentation `html` files should be found inside `docs/build/` directory, and you can use the browser of choice to open the `index.html` to view the full documentation.
