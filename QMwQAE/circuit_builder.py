import numpy as np
from scipy.linalg import polar
import helper.gram_schmidt as gs
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info.operators import Operator
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

class Simulation_parameters():
    """class to contain all the essential parameters for the quantum circuit. A new stochastic process will be 
    build on top of this class
    """
    def __init__(self, sequence: str, memory_size: int, starting_state, sample_size = 1, shots = 100, method = "EIS"):
        """initialize the object with the relevant parameters

        Args:
            sequence (str): sequence that we want to estimate the probability
            memory_size (int): size of the quantum memory
            starting_state (int): initial causal state.
            sample_size (int, optional): number of trials to conduct. Defaults to 1.
            shots (int, optional): number of shots. Defaults to 100.
            method (str, optional): schedule to use. Defaults to "EIS".
        """

        self.sequence_to_amplify = sequence
        self.sequence_length = len(sequence)
        self.sample_size = sample_size
        self.shots = shots
        self.starting_state = starting_state
        self.kraus = []
        self.starting_vectors = []
        self.method = method
        self.memory_size = memory_size

        

    def set_starting_vectors(self, starting_vectors):
        self.starting_vectors = starting_vectors
        
    def set_kraus_ops(self, kraus):
        self.kraus = kraus

    def set_schedule(self, max_depth: int, power = 1):
        
        self.create_depth_list(max_depth, power)
        print(self.max_depth_range)
    
    def create_depth_list(self, max_depth: int, power = 1) -> list:
        """creates the schedule for the experiment

        Args:
            max_depth (int): maximum number of Grover iterators to use
            power (int, optional): degree of the polynomial scheme. Defaults to 1.

        Raises:
            Exception: invalid method given

        Returns:
            list: schedule for the experiment
        """
        if self.method == "EIS":
            max_power = int(np.log2(max_depth))
            max_depth_range = [0] + [2**x for x in range(0,max_power+1)]
        elif self.method == "LIS":
            max_depth_range = [x for x in range(0, max_depth+1)]
        elif self.method == "PIS":
            max_base = int(max_depth**(1./power))
            max_depth_range = [x**power for x in range(0, max_base+1)]
        else:
            raise Exception("Invalid Method: {}".format(self.method))
        self.max_depth_range = max_depth_range
        return max_depth_range

    def import_kraus(self, relative_path: str):
        """imports the kraus operator from file

        Args:
            relative_path (str): relative path to the file
        """
        with open(relative_path, 'rb') as kraus_file:
            kraus_ops = pickle.load(kraus_file)
        self.kraus = []
        for i in range(kraus_ops.shape[0]):
            self.kraus.append(kraus_ops[i])

class Quantum_circuit_builder():
    """class that builds the quantum circuit

    """
    def __init__(self, sim_params: Simulation_parameters):
        """simulation parameters object that describes the stochastic process to simulate

        Args:
            sim_params : subclass of Simulation parameters
        """
        self.sequence_to_amplify = sim_params.sequence_to_amplify[::-1]
        self.sequence_length = len(self.sequence_to_amplify)
        self.memory_size = sim_params.memory_size
    
    def build_unitary_from_vectors(self, start_vectors = [], method = 'polar') -> np.ndarray:
        """creates the unitary matrix from a set of start vectors created from (Binder et. al., 2017) paper 

        Args:
            start_vectors (list, optional): set of start vectors. Defaults to [].
            method (str, optional): method to fill up the unitary, either by Gram Schmidt or polar decomposition. Defaults to 'polar'.

        Returns:
            np.ndarray : unitary matrix 
        """
        if method == 'polar':
            unitary_matrix = np.zeros((start_vectors[0].shape[0], start_vectors[0].shape[0]), dtype = float)
            for i in range(len(start_vectors)):
                unitary_matrix[:, i*2] = start_vectors[i]
            polar_decomposition = polar(unitary_matrix)
            unitary_matrix = polar_decomposition[0]

        elif method == 'gram_schmidt':
            my_gs = gs.Gram_schmidt(start_vectors[0].shape[0], start_vectors = start_vectors)
            my_gs.generate_vectors()
            my_gs.build_unitary()
            unitary_matrix = my_gs.unitary
        
        self.unitary_matrix = unitary_matrix
        return unitary_matrix

    def build_unitary_from_kraus(self, kraus_operators: list) -> np.ndarray:
        """Create unitary matrix from set of kraus operators. Currently, this is only
        set to alphabet size of 2

        Args:
            kraus_operators (list): list of kraus operators (np.ndarray)

        Returns:
            np.ndarray: unitary matrix
        """
        unitary_matrix = np.zeros((kraus_operators[0].shape[0]*2, 
                                   kraus_operators[0].shape[1]*2), dtype = complex) # not generalized to arbitrary alphabet size
        shape = kraus_operators.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    unitary_matrix[i + j*2, k * 2] = kraus_operators[i,j,k]
        polar_decomposition = polar(unitary_matrix)
        unitary_matrix = polar_decomposition[0]
        self.unitary_matrix = unitary_matrix
        return unitary_matrix

    def build_initializer_from_causal_states(self, sim_params: Simulation_parameters):
        """build the unitary matrices and gates to rotate the memory qubit to the 
        correct causal state

        Args:
            sim_params : object containing the simulation parameters
        """
        # unpack causal states
        causal_state_dim = sim_params.causal_states[0].shape[0]
        zero_length_to_append = 2**(int(np.ceil(np.log2(causal_state_dim)))) - causal_state_dim
        new_dim = 2**(int(np.ceil(np.log2(causal_state_dim))))
        # if dimensions not 2^n, append zeros and reshape
        self.initializing_matrix_list = []
        self.initializer_list = [] # store initializing gates
        for causal_state in sim_params.causal_states:
            curr_causal_state = np.append(causal_state, [0 for _ in range(zero_length_to_append)]).reshape(new_dim, 1)
            curr_matrix = np.concatenate((curr_causal_state, np.zeros((new_dim, new_dim-1))), axis = 1, dtype = complex)
            curr_initializing_matrix = polar(curr_matrix)[0]
            self.initializing_matrix_list.append(curr_initializing_matrix)
            curr_initializer = Operator(curr_initializing_matrix)
            curr_initializer_dagger = Operator(np.conj(curr_initializing_matrix.T))
            self.initializer_list.append([curr_initializer, curr_initializer_dagger])

    def build_initializer_from_kraus(self, kraus_ops: np.ndarray, markov_order: int, past: str):
        """creating initializer gates just from the kraus operators, by using the Markov order and
        past sequence

        Args:
            kraus_ops (np.ndarray): list of kraus operators
            markov_order (int): Markov order of the process
            past (str): a certain past sequence
        """
        initializer_list = []
        rand_vector = np.random.rand(kraus_ops.shape[0][0])
        for alphabet in past[-markov_order:]:
            rand_vector = kraus_ops[int(alphabet)].dot(rand_vector)
        start_vector = rand_vector/np.linalg.norm(rand_vector)
        curr_matrix = np.concatenate((start_vector, np.zeros((kraus_ops.shape[0][0], kraus_ops.shape[0][0]-1))), axis = 1)
        initializing_matrix = polar(curr_matrix)[0]
        initializer = Operator(initializing_matrix)
        initializer_dagger = Operator(np.conj(initializing_matrix.T))
        initializer_list.append([initializer, initializer_dagger])
        
        self.initializer_list = initializer_list
    
    def set_initializer(self, sim_params:Simulation_parameters):
        """set the correct initializing gate for the experiment

        Args:
            sim_params (Simulation_parameters): object containing simulation parameters
        """
        initial_state = sim_params.starting_state
        # set initializers
        self.initializer = self.initializer_list[initial_state][0]
        self.initializer_dagger = self.initializer_list[initial_state][1]
            
    def build_unitary_operator(self):
        """create Operator class for unitary to use in quantum circuit 
        from defined unitary matrix
        """
        self.unitary = Operator(self.unitary_matrix)
        unitary_matrix_dagger = np.conj(self.unitary_matrix.T)
        self.unitary_dagger = Operator(unitary_matrix_dagger)

    def build_oracle(self):
        """create the oracle for the quantum circuit
        """
        oracle_matrix = np.eye(2**self.sequence_length)
        idx = int(self.sequence_to_amplify, 2)
        oracle_matrix[idx, idx] = -1 
        oracle = Operator(oracle_matrix)
        oracle_qc = QuantumCircuit(self.sequence_length)
        oracle_qc.append(oracle, range(self.sequence_length))
        self.oracle_qc = oracle_qc.to_instruction()
        self.oracle_qc.name = "oracle"
        
    
    def build_phase_shift_operator(self):
        """create the phase shift gate for the quantum circuit
        """
        state_zero = np.zeros((2**(self.sequence_length+self.memory_size), 2**(self.sequence_length+self.memory_size)))
        state_zero[0, 0] = 1
        phase_shift_matrix = -2 * state_zero + np.eye(2**(self.sequence_length+self.memory_size))
        phase_shift_operator = Operator(phase_shift_matrix)
        phase_shift_qc = QuantumCircuit(self.sequence_length+self.memory_size)
        phase_shift_qc.append(phase_shift_operator, range(self.sequence_length+self.memory_size))
        self.phase_shift_qc = phase_shift_qc.to_instruction()
        self.phase_shift_qc.name = 'phase shift'
        
    
    def build_A_operator(self,visualize = False):
        """create the algorithm operator for the Grover iterator

        Args:
            visualize (bool, optional): option to visualize the quantum circuit. Defaults to False.
        """
        unitary_evolution = QuantumCircuit(self.sequence_length+self.memory_size)
        unitary_evolution.append(self.initializer, [x for x in range(self.sequence_length, self.sequence_length+self.memory_size)])
        for i in range(self.sequence_length):
            unitary_evolution.append(self.unitary, [i]+[x for x in range(self.sequence_length, self.sequence_length+self.memory_size)])
        reverse_evolution = QuantumCircuit(self.sequence_length+self.memory_size)
        for i in reversed(range(self.sequence_length)):
            reverse_evolution.append(self.unitary_dagger, [i]+[x for x in range(self.sequence_length, self.sequence_length+self.memory_size)])
        reverse_evolution.append(self.initializer_dagger, [x for x in range(self.sequence_length, self.sequence_length+self.memory_size)])
        
        if visualize:
            plt.clf()
            unitary_evolution.draw(output='mpl')
            plt.savefig('unitary_evolution.png')
            plt.clf()
            reverse_evolution.draw(output='mpl')
            plt.savefig('reverse_unitary_evolution.png')
            plt.close()

        self.unitary_evolution = unitary_evolution.to_instruction()
        self.unitary_evolution.name = 'A'
        self.reverse_evolution = reverse_evolution.to_instruction()
        self.reverse_evolution.name = 'A dagger'
    
    def build_amplifier(self, visualize = False):
        """assemble the Grover iterator for the quantum circuit

        Args:
            visualize (bool, optional): option to visualize the quantum circuit. Defaults to False.
        """
        amplifier = QuantumCircuit(self.sequence_length+self.memory_size)
        amplifier.append(self.oracle_qc, range(self.sequence_length))
        amplifier.append(self.reverse_evolution, range(self.sequence_length+self.memory_size))
        amplifier.append(self.phase_shift_qc, range(self.sequence_length+self.memory_size))
        amplifier.append(self.unitary_evolution, range(self.sequence_length+self.memory_size))
        if visualize:
            plt.clf()
            amplifier.draw(output='mpl')
            plt.savefig('amplifier.png')
            plt.close()
        
        self.amplifier = amplifier.to_instruction()
        self.amplifier.name = 'amplifier'
    
    def build_circuit_components(self, sim_params: Simulation_parameters):
        """create all the components in the quantum circuit

        Args:
            sim_params (Simulation_parameters): object containing the simulation parameters

        Raises:
            Exception: not enough information provided
        """
        if len(sim_params.kraus) != 0:
            self.build_unitary_from_kraus(sim_params.kraus)
        else:
            self.build_unitary_from_vectors(sim_params.start_vectors)
        
        if len(sim_params.causal_states) != 0:
            self.build_initializer_from_causal_states(sim_params)
        elif len(sim_params.kraus) != 0:
            print("No causal states found... attempting to build causal states from kraus...\n")
            try:
                self.build_initializer_from_kraus(sim_params.kraus, sim_params.markov_order, sim_params.past)
            except:
                raise Exception("not enough information provided for initialization gate!")

        self.set_initializer(sim_params)
        self.build_unitary_operator()
        self.build_A_operator()
        self.build_oracle()
        self.build_phase_shift_operator()
        self.build_amplifier()


    def single_run(self, num_amplifier:int, shots: int, visualize = False)-> dict:
        """run one trial of the quantum circuit at a certain number of Grover iterators

        Args:
            num_amplifier (int): number of Grover iterators to use
            shots (int): number of shots
            visualize (bool, optional): option to visualize the quantum circuit. Defaults to False.

        Returns:
            dict: dictionary of the results
        """
        # build circuit
        qc = QuantumCircuit(self.sequence_length+self.memory_size, self.sequence_length)
        qc.append(self.unitary_evolution, range(self.sequence_length+self.memory_size))
        qc.barrier()
        for _ in range(num_amplifier):
            qc.append(self.amplifier, range(self.sequence_length+self.memory_size))
        qc.measure(range(self.sequence_length), range(self.sequence_length))
        if visualize:
            plt.clf()
            qc.draw(output='mpl')
            plt.savefig('images/single_run.png')
            plt.close()

        #run simulation and get results
        backend = Aer.get_backend('aer_simulator')
        job = execute(qc, backend, shots = shots)
        results = job.result()
        counts = results.get_counts()

        return counts

    def run_amplitude_estimation_circuit(self, depth_range: list, shots: int)-> list:
        """run one trial of the quanutm circuit for MLAE

        Args:
            depth_range (int): the schedule
            shots (int): number of shots per circuit

        Returns:
            list: list of results, with each elements being [number of grover iterators, number of good results] 
        """
        backend = Aer.get_backend('qasm_simulator')
        results = []
        for num_iter in tqdm(depth_range, desc = "running circuits", leave = False):
            # build the circuit
            curr_qc = QuantumCircuit(self.sequence_length+self.memory_size, self.sequence_length)
            curr_qc.append(self.unitary_evolution, range(self.sequence_length+self.memory_size))
            for _ in range(num_iter):
                curr_qc.append(self.amplifier, range(self.sequence_length+self.memory_size))
            curr_qc.measure(range(self.sequence_length), range(self.sequence_length))
            
            # run the circuit and get result
            job = execute(curr_qc, backend, shots = shots)
            result = job.result()
            counts = result.get_counts()
            # print("counts: {} \n".format(counts))
            if self.sequence_to_amplify in counts.keys():
                results.append([num_iter, counts[self.sequence_to_amplify]])
            else:
                results.append([num_iter, 0])
        self.execution_results = results
        return results

class Post_processor():
    """post processor for MLAE
    """
    def __init__(self, results: list):
        """initialization

        Args:
            results (list): list of results from the quantum circuit 
        """
        self.results = results
    
    def log_likelihood(self, num_successes: int, shots: int, num_iter: int, theta: float) -> float:
        """computes the log likelihood function

        Args:
            num_successes (int): number of good states measures
            shots (int): number of shots
            num_iter (int): number of Grover iterators used
            theta (float): current estimated theta

        Returns:
            float: log likelihood of producing result given theta
        """
        prob_success = np.sin((2* num_iter+1) * theta)**2
        prob_failure = np.cos((2* num_iter+1) * theta)**2
        return num_successes * np.log(prob_success) + (shots - num_successes) * np.log(prob_failure)

    def _scan_max_theta(self, thetas: list, shots: int, process: str) -> float:
        """helper function to go through the list of theta and compute
        the log likelihood function and returns the theta with maximum likelihood

        Args:
            thetas (list): list of thetas
            shots (int): number of shots per trial
            process (str): schedule used 

        Returns:
            float: theta with the maximum likelihood
        """
        curr_max_likelihood = -1e99
        theta_with_max_likelihood = 0
        for curr_theta in tqdm(thetas, desc = process, leave = False): 
            curr_likelihood = 0
            for i in range(len(self.results)):
                num_iter = self.results[i][0]
                num_successes = self.results[i][1]
                curr_step_likelihood =  self.log_likelihood(num_successes, shots, num_iter, curr_theta)
                curr_likelihood += curr_step_likelihood
            if curr_likelihood > curr_max_likelihood:
                theta_with_max_likelihood = curr_theta
                curr_max_likelihood = curr_likelihood
        return theta_with_max_likelihood

    def _brute_force_search(self, shots: int) -> float:
        """brute force search through all possible theta values

        Args:
            shots (int): number of shots

        Returns:
            float: theta with maximum likelihood
        """
        thetas = [x /1000 * np.pi/2 for x in range(1, 1000)]
        theta_with_max_likelihood = self._scan_max_theta(thetas, shots, 'estimating likelihood')
        thetas = np.linspace(max(1e-10, theta_with_max_likelihood-0.015), min(np.pi/2, theta_with_max_likelihood+0.015), 1000)
        theta_with_max_likelihood = self._scan_max_theta(thetas, shots, 'refining likelihood')
        return theta_with_max_likelihood
    
    def ml_estimation(self, shots: int, estimation_algo = None)-> float:
        """computes the maximum likelihood algorithm

        Args:
            shots (int): number of shots per trial
            estimation_algo (_type_, optional): algorithm for maximum likelihood function. Defaults to None and uses a brute 
            force search.

        Returns:
            float: theta with maximum likelihood
        """
        if estimation_algo == None:
            return self._brute_force_search(shots)
        else:
            return self.estimation_algo(self.results, shots)
        
class Classical_post_processor():
    """maximum likelihood estimator for the classical algorithm
    """
    def __init__(self):
        return

    def log_likelihood_fn(self, p: float, shots: int, positives: int) -> float:
        """computes the log likelihood function of p given the number of good trials

        Args:
            p (float): current probability
            shots (int): number of shots
            positives (int): number of good trials

        Returns:
            float: log likelihood of p
        """
        return positives *np.log(p) + (shots-positives)*np.log(1-p)

    def _scan_max_likelihood(self, range_of_prob_params: list, process: str, shots: int, positives: int) -> float:
        """go through a given list of probabilities and outputs the probability with maximum likelihood

        Args:
            range_of_prob_params (list): list of probabilities
            process (str): string to describe the algorithm
            shots (int): number of shots
            positives (int): number of good trials

        Returns:
            float: probability with maximum likelihood
        """
        max_likelihood = -1e99
        best_prob = 0
        for curr_prob in tqdm(range_of_prob_params, desc = process, leave = False):
            curr_likelihood = self.log_likelihood_fn(curr_prob, shots, positives)
            if curr_likelihood > max_likelihood:
                max_likelihood = curr_likelihood
                best_prob = curr_prob
                 
        return best_prob
    
    def ml_estimation(self, shots: int, positives: int) -> float:
        """function for maximum likelihood estimation

        Args:
            shots (int): number of shots
            positives (int): number of good trials

        Returns:
            float: probability with maximum likelihood
        """
        range_of_prob = [x/1000 for x in range(1, 1000)]
        p_with_max_likelihood = self._scan_max_likelihood(range_of_prob, process = "estimating likelihood", shots = shots, positives = positives)
        range_of_prob = np.linspace(max(1e-10, p_with_max_likelihood - 0.015), min(1 - 1e-10, p_with_max_likelihood + 0.015), 1000)
        p_with_max_likelihood = self._scan_max_likelihood(range_of_prob, process = "refining likelihood", shots = shots, positives = positives)
        return p_with_max_likelihood