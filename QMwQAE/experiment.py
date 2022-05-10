import numpy as np
from qiskit import QuantumCircuit
import circuit_builder
import examples
from tqdm import tqdm
import matplotlib.pyplot as plt
from qiskit.algorithms import EstimationProblem
import pickle

class Experiment():
    def __init__(self, sim_params: circuit_builder.Simulation_parameters):
        self.sim_params = sim_params

    def _classical_equivalent_runs_calculator(self, shots: int, max_depth: int)-> int:
        """calculates the equivalent number of classical runs

        Args:
            shots (int): number of shots
            max_depth (int): maximum number of Grover iterator

        Returns:
            int: classical shots
        """
        max_depth_index = self.sim_params.max_depth_range.index(max_depth)
        depth_range = self.sim_params.max_depth_range[:max_depth_index+1]
        
        assert depth_range[-1] == max_depth

        return shots * np.sum([2 * m +1 for m in depth_range])

    def _classical_sample_single(self, shots: int, sequence: str, classical_model) -> list:
        """samples the classical model

        Args:
            shots (int): number
            sequence (str): _description_
            classical_model (object): the classical model object

        Returns:
            list: list of probabilities
        """
       
        initial_state = self.sim_params.starting_state
        sequence_length = self.sim_params.sequence_length
        prob_list = []
        my_post_processor = circuit_builder.Classical_post_processor()
        for _ in range(1):
            results = classical_model.sample(shots, sequence_length, initial_state)
            curr_counts = results.get(sequence, 0)
            curr_prob = my_post_processor.ml_estimation(shots, curr_counts)
            prob_list.append(curr_prob)
        return prob_list

    def _amplitude_estimate_bulk_sample_single(self, max_depth: int)-> list:
        """samples the quantum model

        Args:
            max_depth (int): maximum number of Grover iterators

        Returns:
            list: list of probabilities
        """
    
        sample_size = self.sim_params.sample_size
        shots = self.sim_params.shots

        all_estimates = []
        my_qc = circuit_builder.Quantum_circuit_builder(self.sim_params)
        my_qc.build_circuit_components(self.sim_params)
        max_depth_index = self.sim_params.max_depth_range.index(max_depth)
        depth_range = self.sim_params.max_depth_range[:max_depth_index]

        for _ in range(sample_size):
            results = my_qc.run_amplitude_estimation_circuit(depth_range, shots)
            my_processor = circuit_builder.Post_processor(results)
            theta = my_processor.ml_estimation(shots)
            estimated_prob = (np.sin(theta))**2
            all_estimates.append(estimated_prob)

        return all_estimates

    def compare_quantum_advantage(self, classical_model):
        """run multiple trials and get the results for both the quantum and classical model

        Args:
            classical_model (object): object for classical model
        """
        
        # unpack parameters
        sequence = self.sim_params.sequence_to_amplify
        shots = self.sim_params.shots
        max_depth_range = self.sim_params.max_depth_range
        
        all_quantum_estimates = []
        all_classical_estimates = []
        true_probability = classical_model.calculate_true_prob(sequence, self.sim_params.starting_state)
        for curr_depth in tqdm(max_depth_range, desc = 'progress', leave = False):
            curr_quantum_estimates = self._amplitude_estimate_bulk_sample_single(curr_depth)
            curr_quantum_mean = np.mean(curr_quantum_estimates)
            curr_quantum_std = np.std(curr_quantum_estimates)
            # unbiased estimate
            curr_quantum_std = np.sqrt(len(curr_quantum_estimates)/(len(curr_quantum_estimates)-1)) * curr_quantum_std
            all_quantum_estimates.append([curr_quantum_mean, curr_quantum_std])
            classical_runs = self._classical_equivalent_runs_calculator(shots, curr_depth)
            curr_classical_estimates = self._classical_sample_single(classical_runs, sequence, classical_model)
            curr_classical_mean = np.mean(curr_classical_estimates)
            curr_classical_std = np.std(curr_classical_estimates)
            curr_classical_std = np.sqrt(len(curr_classical_estimates)/(len(curr_classical_estimates)-1)) * curr_classical_std
            all_classical_estimates.append([curr_classical_mean, curr_classical_std])
            print("\n mean: {}, median: {}, std: {}, min: {}, max: {}".format(curr_quantum_mean, np.median(curr_quantum_estimates), curr_quantum_std, min(curr_quantum_estimates), max(curr_quantum_estimates)))
        
        all_quantum_estimates = np.array(all_quantum_estimates, dtype = float)
        print(all_quantum_estimates)
        all_classical_estimates = np.array(all_classical_estimates, dtype = float)

        fname = 'quantum_v_classical_sampling_{}'.format(self.sim_params.get_fname())
        
        true_prob_line = [true_probability for _ in range(len(max_depth_range))]
        all_quantum_error = np.abs(all_quantum_estimates[:,0] - true_probability)
        all_classical_error = np.abs(all_classical_estimates[:,0] - true_probability)
        
        data_out = np.array([list(max_depth_range), 
                            all_classical_estimates[:,0], 
                            all_classical_estimates[:,1],
                            all_classical_error,
                            all_quantum_estimates[:,0],
                            all_quantum_estimates[:,1],
                            all_quantum_error,
                            true_prob_line]).T

        np.savetxt(self.sim_params.data_dir+fname+".csv", data_out, delimiter = ',', 
                    header = 'depth, classical estimates, classical std, classical error, quantum estimates, quantum std, quantum error, true probability')
        
        plt.figure(figsize=(6.5,6))
        plt.clf()
        plt.errorbar(max_depth_range, all_quantum_estimates[:,0], yerr = all_quantum_estimates[:,1], fmt = 'o-', capsize = 5, label = 'quantum' )
        plt.errorbar(max_depth_range, all_classical_estimates[:,0], yerr = all_classical_estimates[:,1], fmt = 'o-', capsize = 5, label = 'classical' )
        plt.plot(max_depth_range, true_prob_line, 'o-', label = 'true')
        plt.legend()
        plt.xlabel('depth')
        plt.ylabel('probability')
        plt.title("Probability estimate against number of Grover iterators")
        plt.tight_layout()
        # plt.show()
        plt.savefig(self.sim_params.graph_dir+fname+'.png')

        plt.clf()
        plt.plot(max_depth_range, all_quantum_error, 'o-', label = 'quantum')
        plt.plot(max_depth_range, all_classical_error, 'o-', label = 'classical')
        plt.legend()
        plt.xlabel('depth')
        plt.ylabel('absolute error')
        plt.title("Absolute error against number of Grover iterators")
        plt.tight_layout()
        # plt.show()
        plt.savefig(self.sim_params.graph_dir+'error_'+fname+'.png')

        plt.clf()
        plt.semilogy(max_depth_range, all_quantum_estimates[:,1], 'o-', label = 'quantum' )
        plt.semilogy(max_depth_range, all_classical_estimates[:,1],  'o-', label = 'classical' )
        plt.legend()
        plt.xlabel('depth')
        plt.ylabel('standard deviation')
        plt.title("Error bar against number of Grover iterators")
        plt.tight_layout()
        # plt.show()
        plt.savefig(self.sim_params.graph_dir+'std_'+fname+'.png')
    
    def estimate_probability(self, *args):
        """estimate the probability of the given sequence, averaged over the sample size

        Raises:
            Exception: invalid input
        """
        if len(args) == 2:
            max_depth = self._calc_required_depth(args[0], args[1])
            max_depth = max(max_depth, 5)
            print("Using {} Grover iterators... \n".format(max_depth))
        elif len(args) == 1:
            max_depth = args[0]
        else:
            raise Exception("Input format: (error, estimated_prob) or (max_depth)")
        self.sim_params.set_sampling_scheme(max_depth, "LIS")
        quantum_estimates = self._amplitude_estimate_bulk_sample_single(max_depth)
        quantum_mean = np.mean(quantum_estimates)
        quantum_std = np.std(quantum_estimates)
        print("estimated probability: {} ({})".format(quantum_mean, quantum_std))
    
    def estimate_probability_quick(self, *args):
        """quick estimate of the probability with only 1 trial

        Raises:
            Exception: invalid input
        """
        if len(args) == 2:
            max_depth = self._calc_required_depth(args[0], args[1])
            max_depth = max(max_depth, 5)
            print("Using {} Grover iterators... \n".format(max_depth))
        elif len(args) == 1:
            max_depth = args[0]
        else:
            raise Exception("Input format: (error, estimated_prob) or (max_depth)")
        
        shots = self.sim_params.shots
        self.sim_params.set_sampling_scheme(max_depth, "LIS")
        sample_size = 1
        all_estimates = []
        my_qc = circuit_builder.Quantum_circuit_builder(self.sim_params)
        my_qc.build_circuit_components(self.sim_params)

        for _ in range(sample_size):
            results = my_qc.run_amplitude_estimation_circuit(self.sim_params.max_depth_range, shots)
            my_processor = circuit_builder.Post_processor(results)
            theta = my_processor.ml_estimation(shots)
            estimated_prob = (np.sin(theta))**2
            all_estimates.append(estimated_prob)
        quantum_mean = np.mean(all_estimates)
        print("estimated probability: {} ({})".format(quantum_mean, args[0]))

    def _calc_required_depth(self, error: float, estimated_prob: float) -> int:
        """calculate the number of Grover iterators for the given error and estimated probability

        Args:
            error (float): required error
            estimated_prob (float): rough estimate of probability

        Returns:
            int: number of Grover iterators
        """
        theta = np.arcsin(np.sqrt(estimated_prob))
        min_depth = (np.sqrt((3 * theta * (1-theta))/(4*self.sim_params.shots)) /error)**(2/3)-1
        return int(np.ceil(min_depth))
    
    def export_estimation_problem(self, export = False) -> EstimationProblem:
        """export the problem as an EstimationProblem to use in other Qiskit algorithms

        Returns:
            EstimationProblem: the amplitude estimation problem
        """
        total_qubits = self.sim_params.sequence_length + self.sim_params.memory_size
        sequence = self.sim_params.sequence_to_amplify
        my_qc = circuit_builder.Quantum_circuit_builder(self.sim_params)
        my_qc.build_circuit_components(self.sim_params)
        A = my_qc.unitary_evolution

        class Q_operator(QuantumCircuit):
            def __init__(self, size, circuit):
                super().__init__(size)
                self.Q = circuit
                self.total_qubits = size
                self.append(circuit, [x for x in range(size)])
            def power(self, k):
                circuit = QuantumCircuit(self.total_qubits)
                for _ in range(k):
                    circuit.append(self.Q, [x for x in range(self.total_qubits)])
                return circuit
        
        Q = Q_operator(total_qubits, my_qc.amplifier)
        qubits_to_measure = [x for x in range(self.sim_params.sequence_length)]
        
        good_state_fn = lambda string : all(string[i] == sequence[i] for i in range(len(string)))

        problem = EstimationProblem(
            state_preparation=A,
            grover_operator = Q,
            objective_qubits= qubits_to_measure,
            is_good_state = good_state_fn
        )
        if export:
            pickle.dump(problem, "estimationProblemPickled.pkl")

        return problem

if __name__ == "__main__":

    ###############################
    # for perturbed coin
    p = 0.1
    method = ["EIS", 2]
    sequence = '0010'

    sequence_length = len(sequence)
    shots = 100
    sample_size = 1000
    starting_state = 0
    max_depth = 8

    my_coin = examples.Perturbed_coin(p, p, starting_state)
    my_params = examples.Perturbed_coin_simulation_params(p, sequence, sample_size, shots, starting_state, method, max_depth)
    
    my_expt = Experiment(my_params)
    error = 0.0001
    estimated_prob = 0.0001
    # my_expt.compare_quantum_advantage(my_coin)

    # my_expt.estimate_probability(10)
    # my_expt.estimate_probability(error, estimated_prob)
    my_expt.estimate_probability_quick(error, estimated_prob)
    #################################

    ################################
    # # for dual poisson process
    # p = 0.1
    # q1 = 0.1
    # q2 = 0.2

    # method = ["LIS", 1]
    # sequence = '000'
    
    # sequence_length = len(sequence)
    # shots = 100
    # sample_size = 100
    # starting_state = 0
    # max_depth = 5

    # my_classical_poisson = examples.dual_poisson_process(p, q1, q2)
    # my_params = examples.Dual_poisson_sim_params(p, q1, q2, sample_size, sequence, shots, method, max_depth)
    
    # my_expt = Experiment(my_params)
    # my_expt.compare_quantum_advantage(my_classical_poisson)

    ###################################
    # for Nemo process
    # p = 0.1
    
    # method = ["LIS", 1]
    # sequence = '000'
    
    # sequence_length = len(sequence)
    # shots = 100
    # sample_size = 100
    # starting_state = 0
    # max_depth = 5

    # my_nemo = examples.Nemo_process(p)
    # my_params = examples.Nemo_sim_params(p, sequence, 2, sample_size, shots, starting_state, method, max_depth)
    # my_expt = Experiment(my_params)
    # my_expt.compare_quantum_advantage(my_nemo)
