
from qiskit import BasicAer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import MaximumLikelihoodAmplitudeEstimation
import experiment
import examples
import circuit_builder

def main():
    """function to demonstrate the exporting of EstimationProblem
    """
    p = 0.2 # old notation
    print("p : {}".format(1-p))

    starting_state = 0
    print("starting causal state: {}".format(starting_state))

    sequence = '000'
    print("sequence: {}".format(sequence))

    shots = 100
    sample_size = 100

    sim_params = examples.Perturbed_coin_simulation_params(p, sequence, sample_size, shots, starting_state)

    my_expt = experiment.Experiment(sim_params)

    problem = my_expt.export_estimation_problem()

    backend = BasicAer.get_backend("statevector_simulator")
    quantum_instance = QuantumInstance(backend)

    mlae = MaximumLikelihoodAmplitudeEstimation(
        evaluation_schedule=3, quantum_instance=quantum_instance  # log2 of the maximal Grover power
    )
    # defaults to an exponential schedule
    # if a list present, the powers of Grover operator to use
    # defaults to a brute force search for MLE

    mlae_result = mlae.estimate(problem)

    print("Estimate:", mlae_result.estimation)

if __name__ == "__main__":
    main()