from typing import Iterable
import numpy as np
from circuit_builder import Simulation_parameters

##################################################
## Perturbed coin Process
##################################################
class Perturbed_coin():
    def __init__(self, p0:float,p1:float, state = 2):
        """A more general perturbed coin class within allows for 
        different transition probabilities between causal states 0 and 1

        Args:
            p0 (float): probability of staying in causal state 0 
            p1 (float): probability of staying in causal state 1
            state (int, optional): initial causal state. Defaults to 2.
        """
        self.p0 = p0
        self.p1 = p1
        if state == 2:
            self.state = int(np.round(np.random.rand()))
        else:
            self.state = state

    def reset_state(self, state:int):
        """reset the causal state

        Args:
            state (int): causal state
        """
        self.state = state

    def simulate(self, sequence_length:int)->np.ndarray:
        """simulate a sequence of outputs

        Args:
            sequence_length (int): length of sequence to generate

        Returns:
            np.ndarray: output sequence
        """
        results = []
        for _ in range(sequence_length):
            if self.state == 0: # in the zero state
                temp = np.random.rand()
                if temp < self.p0:
                    results.append(0)
                else:
                    results.append(1)
                    self.state = 1
            else:
                temp = np.random.rand()
                if temp < self.p1:
                    results.append(1)
                else:
                    results.append(0)
                    self.state = 0
        self.result = np.array(results, dtype = int)
        return np.array(results, dtype = int)
    
    def sample(self, sample_size: int, sequence_length: int, initial_state: int) -> dict:
        """conduct a given number of trials, with each trial generating 
        a sequence defined by sequence_length

        Args:
            sample_size (int): number of trials
            sequence_length (int): length of each sequence
            initial_state (int): initial causal state

        Returns:
            dict: dictionary of results, with each sequence as the key and the number of samples as the value
        """
        all_results = {}
        for i in range(2**sequence_length):
            binary_string =  (sequence_length - len(bin(i)[2:])) * '0' +bin(i)[2:]
            all_results[binary_string] = 0
        for _ in range(sample_size):
            self.state = initial_state
            results = self.simulate(sequence_length)
            result_string = ''
            for i in results:
                result_string = result_string + str(i)
            all_results[result_string] += 1
        self.bulk_sample = all_results
        return all_results
    
    def calculate_true_prob(self, sequence: str, initial_state: int)->float:
        """calculate the true probability of a sequence

        Args:
            sequence (str): the sequence to compute
            initial_state (int): initial causal state

        Returns:
            float: probability of generating the sequence
        """
        total_prob = 1
        curr_state = initial_state
        for value in sequence:
            if curr_state == 1:
                if int(value) == 1:
                    total_prob *= self.p1
                elif int(value) == 0:
                    total_prob *= (1-self.p1)
                    curr_state = 0
            elif curr_state == 0:
                if int(value) == 0:
                    total_prob *= self.p0
                elif int(value) == 1:
                    total_prob *= (1-self.p0)
                    curr_state = 1
        return total_prob
        
    def probabilties(self, results: Iterable):
        """calculate the empirical probability 

        Args:
            results (Iterable): list of number of appearance of the sequence in each trial

        Returns:
            float, float: probabilities p0 and p1
        """
        prob1 = np.mean(results)
        prob0 = 1 - prob1
        return prob0, prob1

    def stat_complexity_analytical(self) -> float:
        """calculate the analytical statistical complexity

        Returns:
            float: analytical statistical complexity
        """
        prob0 = (1-self.p1) /( 2- self.p0 - self.p1)
        prob1 = (1-self.p0) / (2- self.p0 - self.p1)
        self.complexity = - (prob0 * np.log2(prob0) + prob1 * np.log2(prob1))
        return self.complexity

    def stat_complexity(self) -> float:
        """calculates the empirical statistical complexity

        Returns:
            float: empirical statistical complexity
        """
        prob0, prob1 = self.probabilties(self.result)
        self.complexity = 0
        comp0 = -prob0 * np.log2(prob0)
        comp1 = -prob1 * np.log2(prob1)
        self.complexity = comp0 + comp1
        return self.complexity 

class Perturbed_coin_simulation_params(Simulation_parameters):
    def __init__(self, p, sequence, sample_size, shots, starting_state):
        """simulation parameters for perturbed coin process

        Args:
            p (float): transition probability
            sequence (str): sequence of interest
            sample_size (int): number of trials to conduct
            shots (int): shots per trial
            starting_state (int): starting causal state
        """
        super(Perturbed_coin_simulation_params, self).__init__(sequence, 1, starting_state, sample_size, shots)
        
        self.p = p
        self.graph_dir = '../graphs/perturbed_coin/' # directories for saving
        self.data_dir = '../data/perturbed_coin/error_analysis/' # directories for saving
        self._define_causal_states()
        self._define_starting_vectors()
    
    def _define_causal_states(self):
        """defines causal states
        """
        causal_state_0 = np.array([np.sqrt(self.p), np.sqrt(1-self.p)])
        causal_state_1 = np.array([np.sqrt(1-self.p), np.sqrt(self.p)])
        self.causal_states = [causal_state_0, causal_state_1]
    
    def _define_starting_vectors(self):
        """defines starting vectors
        """
        first_vector = np.array([np.sqrt(self.p), 0, np.sqrt(1-self.p), 0])
        second_vector = np.array([0, np.sqrt(1-self.p), 0, np.sqrt(self.p)])
        self.start_vectors = [first_vector, second_vector]

    def get_fname(self) -> str:
        """returns the string for the file name

        Returns:
            str: file name
        """
        return 'perturbed_coin_sequence_{}_p_{}_shots_{}_max_depth_{}_sample_size_{}_method_{}'.format(
            self.sequence_to_amplify, self.p, self.shots, self.max_depth_range[-1], self.sample_size, self.method
        )

# ==================================================
# Nemo Process
# ==================================================

# classical simulator
class Nemo_process(object):
    def __init__(self, p: float):
        """The classical model for the Nemo process

        Args:
            p (float): probability of transition to causal state 2
        """
        self.p = p
        self.state = 0
        self.topological_complexity = np.log2(3)

    def simulate(self, sequence_length: int, starting_state = 0) -> str:
        """simulates the Nemo process

        Args:
            sequence_length (int): length of sequence to produce
            starting_state (int, optional): starting causal state. Defaults to 0.

        Returns:
            str: output sequence
        """
        result_container = []
        state_container = []
        if starting_state == "random":
            curr_state = np.random.randint(0, 3)
        else:
            curr_state = starting_state
        state_container.append(curr_state)
        for _ in range(sequence_length):
            if curr_state == 0:
                temp = np.random.rand()
                if temp < self.p:
                    result_container.append(0)
                    curr_state = 0
                else:
                    result_container.append(1)
                    curr_state = 1
            elif curr_state == 1:
                result_container.append(1)
                curr_state = 2
            else:
                temp = np.random.rand()
                if temp < 0.5:
                    result_container.append(0)
                else:
                    result_container.append(1)
                curr_state = 0
            state_container.append(curr_state)
        self.result = np.array(result_container, dtype = int)
        self.state_container = np.array(state_container)
        return ''.join([str(x) for x in result_container])
    
    def sample(self, sample_size: int, sequence_length: int, starting_state: int) -> dict:
        """conduct a given number of trials, with each trial generating 
        a sequence defined by sequence_length

        Args:
            sample_size (int): number of trials
            sequence_length (int): length of each sequence
            initial_state (int): initial causal state

        Returns:
            dict: dictionary of results, with each sequence as the key and the number of samples as the value
        """
        result_dict = {}
        for _ in range(sample_size):
            sequence = self.simulate(sequence_length, starting_state = starting_state)
            result_dict[sequence] = result_dict.get(sequence, 0) + 1
        return result_dict
    
    def calc_stat_complexity(self)-> float:
        """calculate the empirical statistical complexity of the process

        Returns:
            float: statistical complexity
        """
        prob_state_0 = np.sum(self.state_container == 0)/len(self.state_container)
        prob_state_1 = np.sum(self.state_container == 1)/len(self.state_container)
        prob_state_2 = np.sum(self.state_container == 2)/len(self.state_container)
        vec_of_probs = np.array([prob_state_0, prob_state_1, prob_state_2], dtype = float)
        self.stat_complexity = -np.sum(vec_of_probs * np.log2(vec_of_probs))
        return self.stat_complexity
    
    def calc_stat_complexity_analytical(self)-> float:
        """calculate the analytical statistical complexity

        Returns:
            float: statistical complexity
        """
        prob_state_0 = 1/(3 - 2* self.p)
        prob_state_1 = (1-self.p)/(3 - 2* self.p)
        prob_state_2 = (1-self.p)/(3 - 2* self.p)
        vec_of_probs = np.array([prob_state_0, prob_state_1, prob_state_2], dtype = float)
        self.stat_complexity_analytical = -np.sum(vec_of_probs * np.log2(vec_of_probs))
        return self.stat_complexity_analytical
    
    def calculate_true_prob(self, sequence: str, starting_state: int)-> float:
        """calculates the probability of a given sequence

        Args:
            sequence (str): sequence of interest
            starting_state (int): starting causal state

        Returns:
            float: true probability
        """
        curr_state = starting_state
        true_prob = 1
        for element in sequence:
            if element == '0':
                if curr_state == 0:
                    true_prob *= self.p
                    curr_state = 0
                elif curr_state == 1:
                    true_prob *= 0
                elif curr_state == 2:
                    true_prob *= 0.5
                    curr_state = 0
            else:
                if curr_state == 0:
                    true_prob *= (1-self.p)
                    curr_state = 1
                elif curr_state == 1:
                    true_prob *= 1
                    curr_state = 2
                elif curr_state == 2:
                    true_prob *= 0.5
                    curr_state = 0
        return true_prob

# quantum circuit simulation parameters
class Nemo_sim_params(Simulation_parameters):
    def __init__(self, p: float, sequence: str, memory_size: int, sample_size: int, shots: int, starting_state: int):
        """the simulation parameter class for Nemo process

        Args:
            p (float): characteristic probability
            sequence (str): sequence of interest
            memory_size (int): number of memory qubit
            sample_size (int): number of trials to conduct
            shots (int): number of shots per trial
            starting_state (int): starting causal state
        """

        super(Nemo_sim_params, self).__init__(sequence, memory_size, starting_state, sample_size, shots)
        self.p = p
        self.graph_dir = 'graphs/nemo/'
        self.data_dir = 'data/nemo/error_analysis/'
        self.overlap_matrix = np.array([
            [1, np.sqrt(p * (1-p))/(1+p), np.sqrt(2*p)/(1+p)],
            [np.sqrt(p * (1-p))/(1+p), 1, np.sqrt(p)/(1+p)],
            [np.sqrt(2*p)/(1+p), np.sqrt(p)/(1+p), 1]
        ], dtype = float)
        self._build_causal_states()
        self._build_starting_vectors()

    
    def _build_causal_states(self):
        """creates the causal states
        """
        b = (self.overlap_matrix[2,1] - self.overlap_matrix[2,0] * self.overlap_matrix[1,0])/(np.sqrt(1-self.overlap_matrix[1,0]**2))
        c = np.sqrt(1- self.overlap_matrix[2,0]**2 - b**2)

        state_1 = np.array([1,0,0])
        state_2 = np.array([self.overlap_matrix[0,1], np.sqrt(1-self.overlap_matrix[0,1]**2), 0])
        state_3 = np.array([self.overlap_matrix[0,2], b, c])
        self.causal_states = [state_1, state_2, state_3]
    
    def _build_starting_vectors(self)-> list:
        """creates the starting vectors

        Returns:
            list: list of starting vectors
        """
        identity = np.eye(8)
        v1 = np.array([
            np.sqrt(self.p), 
            self.overlap_matrix[0,1] * np.sqrt(1-self.p),
            0,
            np.sqrt(1-self.p) * np.sqrt(1-(self.overlap_matrix[0,1])**2),
            0,
            0,
            0,
            0
        ]) 
        b = (self.overlap_matrix[2,1] - self.overlap_matrix[2,0] * self.overlap_matrix[1,0])/(np.sqrt(1-self.overlap_matrix[1,0]**2))
        c = np.sqrt(1- self.overlap_matrix[2,0]**2 - b**2)
        v2 = 1/(np.sqrt(1-self.overlap_matrix[0,1]**2)) * (self.overlap_matrix[0,2] * identity[1,:] + b * identity[3,:]+ c * identity[5,:]- self.overlap_matrix[0,1] * v1)
        v3 = 1/c * (1/np.sqrt(2) * identity[0,:] + 1/np.sqrt(2) * identity[1,:] - self.overlap_matrix[0,2] * v1 - b * v2)
        v4 = np.array([
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0
        ], dtype= float)
        self.start_vectors = [v1, v2, v3, v4]

        return self.start_vectors
    
    def get_fname(self):
        """returns the string for the file name

        Returns:
            str: file name
        """
        return 'nemo_sequence_{}_p_{}_shots_{}_max_depth_{}_sample_size_{}_method_{}'.format(
            self.sequence_to_amplify, self.p, self.shots, self.max_depth_range[-1], self.sample_size, self.method
        )

# ==================================================
# Dual Poisson Process
# ==================================================

class Dual_poisson_process(object):
    def __init__(self, p: float, q1: float, q2: float):
        """the classical dual poisson process

        Args:
            p (float): probability p
            q1 (float): probability of staying in state 1
            q2 (float): probability of staying in state 2
        """

        self.p = p
        self.q1 = q1
        self.q2 = q2

    def survival_probability(self, k: int) -> float:
        """calculates the survival probability

        Args:
            k (int): number of steps 

        Returns:
            float: survival probability
        """

        return self.p*self.q1**k + (1-self.p)*self.q2**k

    def sample(self, shots: int, sequence_length: int, *args):
        """

        Args:
            shots (int): number of shots of the experiment
            sequence_length (int): maximum length of the sequences
            
        Returns:
            dict: dictionary that contains the sequences with their count
        """
        results = {}
        
        for _ in range(shots):
            curr_string = self.simulate()
            results[curr_string] = results.get(curr_string, 0) + 1

        results = self.max_length_filter(results, sequence_length)
        self.results = results

        return results

    def simulate(self) -> str:
        """simulate the Nemo process

        Returns:
            str: output sequence
        """
        string = []
        curr_k = 0
        emitted_0 = True
        while emitted_0:
            transition_prob = self.survival_probability(curr_k+1)/self.survival_probability(curr_k)
            curr_random_num = np.random.random()
            if curr_random_num < transition_prob:
                string.append('0')
            else:
                string.append('1')
                emitted_0 = False
                break
            curr_k += 1

        return ''.join(string)

    def max_length_filter(self, results: dict, max_length: int)-> dict:
        """filter through the output results for further analysis

        Args:
            results (dict): unfiltered results
            max_length (int): maximum length of the results

        Returns:
            dict: filtered results
        """
        filtered_results = {}
        for i in range(max_length):
            filtered_results['0'*i+'1'] = 0
        filtered_results['0'*(max_length)] = 0
        for key in results.keys():
            if len(key) > max_length:
                filtered_results['0'*(max_length)] += results[key]
            else:
                filtered_results[key] = results[key]
        return filtered_results

    def calculate_true_prob(self, sequence: str, *args)-> float:
        """calculate the exact probability of the sequence

        Args:
            sequence (str): sequence of interest

        Returns:
            float: true probability
        """
        prob = 1
        for idx in range(len(sequence)):
            if sequence[idx] == '0':
                prob *= (self.survival_probability(idx +1)/self.survival_probability(idx))
            if sequence[idx] == '1':
                prob *= (1- self.survival_probability(idx +1)/self.survival_probability(idx))
        return prob

class Dual_poisson_sim_params(Simulation_parameters):
    def __init__(self, p: float, q1: float, q2: float, sample_size: int, sequence: str, shots: int, starting_state = 0):
        """simulation parameters for the dual poisson process

        Args:
            p (float): characteristic probability
            q1 (float): characteristic q1
            q2 (float): characteristic q2
            sample_size (int): number of trials to conduct
            sequence (str): sequence of interest
            shots (int): number of shots per trial
            starting_state (int, optional): starting causal state. Defaults to 0.
        """
        super(Dual_poisson_sim_params, self).__init__(sequence,1, starting_state, sample_size, shots)
        self.p = p
        self.q1 = q1
        self.q2 = q2
        self.p_bar = 1-p
        self.graph_dir = 'graphs/dual_poisson/'
        self.data_dir = 'data/dual_poisson/error_analysis/'
        self.g = (np.sqrt((1-self.q1)*(1-self.q2))/(1-np.sqrt(self.q1 * self.q2)))
        self._build_kraus()
        self._build_first_causal_state()

    def _build_kraus(self) -> np.ndarray:
        """creates the Kraus operator

        Returns:
            np.ndarray: Kraus operators
        """
        g = self.g
        A0 = np.array([
            [np.sqrt(self.q1), g * (np.sqrt(self.q2) - np.sqrt(self.q1))/np.sqrt(1-g**2)],
            [0, np.sqrt(self.q2)]
        ])
        a = np.sqrt(1-self.q1) * ( np.sqrt(self.p) + 1j * np.sqrt(self.p_bar) * g)
        c = 1j * np.sqrt(1-self.q1) * np.sqrt(self.p_bar) * np.sqrt(1-g**2)
        k = np.sqrt(1-self.q2)/np.sqrt(1-self.q1) - g
        A1 = np.array([
            [a, a/np.sqrt(1-g**2)*k],
            [c, c/np.sqrt(1-g**2)*k]
        ])
        kraus_operators = np.array([A0, A1])
        self.kraus = kraus_operators
        return kraus_operators
    
    def _build_first_causal_state(self)-> list:
        """create the initial causal state

        Returns:
            list: initial causal state
        """
        gamma = np.arctan(self.g*np.sqrt(self.p_bar)/np.sqrt(self.p))
        zero_state = np.array([np.sqrt(self.g**2 * self.p_bar + self.p), np.sqrt(self.p_bar) * np.sqrt(1-self.g**2) * np.exp(1j * (np.pi/2 -gamma))])
        self.causal_states = [zero_state]
        return zero_state
    
    def max_length_filter(self, results: dict, max_length: int)-> dict:
        """filters the raw results 

        Args:
            results (dict): unfiltered results
            max_length (int): maximum length to consider

        Returns:
            dict: filtered results
        """
        filtered_results = {}
        for i in range(self.sequence_length):
            filtered_results['0'*i+'1'] = 0
        filtered_results['0'*(max_length)] = 0
        for key in results.keys():
            if len(key) > max_length:
                filtered_results['0'*(max_length)] += results[key]
            else:
                filtered_results[key] = results[key]
        return filtered_results
    
    def get_fname(self):
        """returns the string for the file name

        Returns:
            str: file name
        """
        return '_sequence_{}_p_{}_q1_{}_q2_{}_shots_{}_max_depth_{}_sample_size_{}_method_{}'.format(
            self.sequence_to_amplify, self.p, self.q1, self.q2, self.shots, self.max_depth_range[-1], self.sample_size, self.method
        )

if __name__ == "__main__":
    pass