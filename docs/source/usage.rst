Usage
=====
.. role:: python(code)
  :language: python
  :class: highlight

.. _installation:

Example of Stochastic Models
----------------------------
This module contains three sample stochastic model: Perturbed coin process, dual Poisson process and Nemo process.
The classical models are all defined within in the script :python:`examples.py` within the :python:`QMwQAE/` directory. 
Their corresponding parameters classes are also defined within the same file, and these classes can be imported into
the experiment file to be input into the :python:`Experiment` class to run the algorithm. Within the :python:`experiment.py` file, there are also examples of how to 
initialize the relevant classes. Just run the following command within the terminal to see the algorithm in action

.. code-block:: bash

    $ python experiment.py

Creating new stochastic models
------------------------------
As seen in the example classes, a new :python:`parameters` class must be created. This parameters must be built on top 
of the :python:`Simulation_parameters` class as defined in the :python:`circuit_builder.py` file. 
This class requires the definition of the following few quantities:

* :python:`sequence`: the sequence for the algorithm to estimate its probability

* :python:`memory_size`: the number of qubits that is required for the memory register

* :python:`starting_state`: the initial causal state of the system

There are a few optional parameters that the user can define

* :python:`sample_size`: number of trials to conduct. Increasing this will help to give better estimate due to the Central Limit Theorem

* :python:`shots`: the number of shots per experiment. Increasing this also increases the precision of the estimate.

* :python:`method`: the schedule to be used for the experiment.

When the user define the new parameters class, they also have to define some variables for the 
algorithm to work. The user can provide the variables in three different manners:

1. The user can provide the starting vectors to populate the odd-numbered columns in the unitary matrix. This variable should be
a list of :python:`numpy.ndarray` vectors with the variable name :python:`self.starting_vectors`. The starting vectors 
can be accompanied by a list of causal states also in :python:`numpy.ndarray` under the variable name :python:`self.causal_states`.


2. The user can provide the starting vectors, accompanied by a set of Kraus operators that define the action on the 
memory qubits. The Kraus operators should be provided in a list of :python:`numpy.ndarray` under the variable name :python:`self.kraus`.


3. The user can provide the Kraus operators with the Markov order of the process, under the variable :python:`self.markov_order`, 
and the past, under the variable :python:`self.past`.


The user should also provide two different variables for saving of data

* :python:`self.graph_dir`: the directory to save the graphs generated

* :python:`self.data_dir`: the directory to save the data generated


Exporting to Qiskit
-------------------

This package allows the user to embed the Grover iterator and the algorithm operator 
into the :python:`EstimationProblem` class provided by Qiskit. This allows the user to export the circuit
created to be used with other algorithms within the :python:`Qiskit` package. The export function is within the :python:`experiment.py` file, 
called :python:`Experiment.export_estimation_problem`. The user can initialize the :python:`Experiment` class with the defined :python:`parameters` object.
Calling the function :python:`export_estimation_problem` with the variable :python:`export` set as :python:`True` will export the :python:`EstimationProblem` object with the :python:`pickle` module, saving the object in the
:python:`estimationProblemPickled.pkl` file. 

.. code-block:: python3

    my_expt = experiment.Experiment(my_params) # my_params is the defined parameters object
    my_expt.export_estimation_problem(export = True) # file saved if export set to True

To import the pickled file, use

.. code-block:: python3

    problem = pickle.load(<path-to-file>)

alternatively, the :python:`Experiment.export_estimation_problem` can also output the :python:`EstimationProblem` object directly

.. code-block:: python3

    problem = my_expt.export_estimation_problem()

An example of how to use the :python:`EstimationProblem` object is given in the file :python:`qiskit_qe.py` under the :python:`QmwQAE/` directory.



