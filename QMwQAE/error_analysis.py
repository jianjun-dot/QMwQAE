from typing import Callable, Iterable
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

def load_data(path_to_file: str)-> np.ndarray:
    """small function to load the data

    Args:
        path_to_file (str): relative/absolute path to the file 

    Returns:
        np.ndarray: imported data
    """
    data = np.genfromtxt(path_to_file, delimiter = ',', skip_header = 1)
    return data

def cal_N_q(depth: int, depth_range: list, shots = 100) -> int:
    """calculates the number of calls to the algorithm

    Args:
        depth (int): number of Grover iterators used
        depth_range (str, optional): list of Grover iterators used
        shots (int, optional): number of shots. Defaults to 100.

    Returns:
        int: number of oracle calls
    """

    depth_index = depth_range.index(depth)
    curr_range = depth_range[:depth_index+1]
    assert curr_range[-1] == depth
    return shots * np.sum([2*m+1 for m in curr_range])

def fit_two_data(fit_function: Callable, process: str, path: str, fname1: str, fname2: str):
    """Get the best fit and plot for two different data set

    Args:
        fit_function (Callable): function to use for the best fit
        process (str): the random process
        path (str): path to the folder containing the file
        fname1 (str): file name of the first data set
        fname2 (str): file name of the second data set
    """
    log_depth_range1,classical_fits1, quantum_fits1, log_Nq_range1,log_classical_std1, log_quantum_std1, quantum_fits_Nq1, classical_fits_Nq1, quantum_r_squared_1, classical_r_squared_1 = fit_data(fit_function, process, path, fname1, start = 2, method = "LIS", visualize = False)
    log_depth_range2, _, quantum_fits2, log_Nq_range2, _, log_quantum_std2, quantum_fits_Nq2, _, quantum_r_squared_2, classical_r_squared_2= fit_data(fit_function, process, path, fname2, start = 1, method = "EIS", visualize = False)
    plt.clf()
    plt.figure(figsize=(6.5,6))
    # plt.rcParams.update({'font.size': 12})
# =============================================================================
#     plt.plot(log_Nq_range1, log_classical_std1, 'o--', color = "C0", label = "classical data")
#     plt.plot(log_Nq_range1, fit_function(log_Nq_range1, *classical_fits_Nq1), '-', color = "C0", label = "classical fit: a = {:.3f}".format(classical_fits_Nq1[0]))
#     plt.plot(log_Nq_range1, log_quantum_std1, 'o--',color = "C1", label = "LIS data")
#     plt.plot(log_Nq_range1, fit_function(log_Nq_range1, *quantum_fits_Nq1), '-' ,color = "C1",label = "LIS fit: a = {:.3f}".format(quantum_fits_Nq1[0]))
#     plt.plot(log_Nq_range2, log_quantum_std2, 'o--',color = "C2", label = "EIS data")
#     plt.plot(log_Nq_range2, fit_function(log_Nq_range2, *quantum_fits_Nq2), '-' ,color = "C2",label = "EIS fit: a = {:.3f}".format(quantum_fits_Nq2[0]))
#     
#     plt.xlabel("ln($N_q$)", fontsize = 15)
#     plt.ylabel("ln($\epsilon$)", fontsize = 15)
# =============================================================================
    plt.loglog(np.exp(log_Nq_range1), np.exp(log_classical_std1), 'o--', color = "C0", label = "classical data")
    plt.loglog(np.exp(log_Nq_range1), np.exp(fit_function(log_Nq_range1, *classical_fits_Nq1)), '-', color = "C0", label = "classical fit: a = {:.3f}".format(classical_fits_Nq1[0]))
    plt.loglog(np.exp(log_Nq_range1), np.exp(log_quantum_std1), 'o--',color = "C1", label = "LIS data")
    plt.loglog(np.exp(log_Nq_range1), np.exp(fit_function(log_Nq_range1, *quantum_fits_Nq1)), '-' ,color = "C1",label = "LIS fit: a = {:.3f}".format(quantum_fits_Nq1[0]))
    plt.loglog(np.exp(log_Nq_range2), np.exp(log_quantum_std2), 'o--',color = "C2", label = "EIS data")
    plt.loglog(np.exp(log_Nq_range2), np.exp(fit_function(log_Nq_range2, *quantum_fits_Nq2)), '-' ,color = "C2",label = "EIS fit: a = {:.3f}".format(quantum_fits_Nq2[0]))
    
    
    plt.legend()
    # plt.title(process)
    plt.xlabel("$N_q$", fontsize = 15)
    plt.ylabel("Error", fontsize = 15)
    plt.tight_layout()
    # plt.show()
    plt.savefig("Nq_EIS_LIS_"+fname1.replace("csv", "png"))
    
    
    print('classical r2: {}'.format(classical_r_squared_1))
    print('LIS r2: {}'.format(quantum_r_squared_1))
    print('EIS r2: {}'.format(quantum_r_squared_2))

# =============================================================================
#     plt.clf()
#     plt.figure(figsize=(6.5,6))
#     #plt.loglog(np.exp(log_depth_range1), np.exp(log_classical_std1), 'o--', color = "C0", label = "classical data")
#     #plt.loglog(np.exp(log_depth_range1), np.exp(fit_function(log_depth_range1, *classical_fits1)), '-', color = "C0", label = "classical fit: a = {:.6f}".format(classical_fits1[0]))
#     plt.loglog(np.exp(log_depth_range1), np.exp(log_quantum_std1), 'o--',color = "C1", label = "LIS data")
#     plt.loglog(np.exp(log_depth_range1), np.exp(fit_function(log_depth_range1, *quantum_fits1)), '-' ,color = "C1",label = "quantum fit: a = {:.6f}".format(quantum_fits1[0]))
#     plt.loglog(np.exp(log_depth_range2), np.exp(log_quantum_std2), 'o--',color = "C2", label = "EIS data")
#     plt.loglog(np.exp(log_depth_range2), np.exp(fit_function(log_depth_range2, *quantum_fits2)), '-' ,color = "C2",label = "quantum fit: a = {:.6f}".format(quantum_fits2[0]))
#     plt.legend()
#     # plt.title(process)
#     plt.xlabel("Number of Grover iterator")
#     plt.ylabel("Standard error")
#     plt.tight_layout()
#     plt.show()
#     plt.savefig("EIS_LIS_"+ fname1.replace("csv", "png"))
# =============================================================================

def fit_data(fit_function: Callable, process: str, path: str, fname: str, start:int, method:str, visualize = True):
    """get the best fit for the data set

    Args:
        fit_function (Callable): the best fit function
        process (str): name of the random process
        path (str): path to the folder of the data
        fname (str): file name
        start (int): starting index of the first usable data to remove anomaly
        method (str): schdule used
        visualize (bool, optional): plot the graph. Defaults to True.

    Returns:
        list of np.ndarray: list of the calculated values 
    """
    data = load_data(path+fname)
    depth_range = data[start:,0]
    depth_range = [int(depth) for depth in depth_range]
    Nq_range = [cal_N_q(depth, data[:,0].tolist()) for depth in depth_range]
    classical_std = data[start:,2]
    quantum_std = data[start:,5]

    log_classical_std = transform_function(classical_std)
    log_quantum_std = transform_function(quantum_std)
    log_depth_range = transform_function(depth_range)
    log_Nq_range = transform_function(Nq_range)
    classical_fits, _ = curve_fit(fit_function, log_depth_range, log_classical_std)
    quantum_fits, _ = curve_fit(fit_function, log_depth_range, log_quantum_std)

    classical_fits_Nq, _ = curve_fit(fit_function, log_Nq_range, log_classical_std)
    quantum_fits_Nq, _ = curve_fit(fit_function, log_Nq_range, log_quantum_std)
    
    quantum_r_squared = calc_r_square_value(fit_function, log_quantum_std , log_Nq_range, quantum_fits_Nq)
    classical_r_squared = calc_r_square_value(fit_function, log_classical_std, log_Nq_range, classical_fits_Nq)

    if visualize:
        plt.clf()
        plt.plot(log_Nq_range, log_classical_std, 'o--', color = "C0", label = "classical data")
        plt.plot(log_Nq_range, fit_function(log_Nq_range, *classical_fits_Nq), '-', color = "C0", label = "classical fit: a = {:.6f}".format(classical_fits_Nq[0]))
        plt.plot(log_Nq_range, log_quantum_std, 'o--',color = "C1", label = "quantum data")
        plt.plot(log_Nq_range, fit_function(log_Nq_range, *quantum_fits_Nq), '-' ,color = "C1",label = "quantum fit: a = {:.6f}".format(quantum_fits_Nq[0]))
        plt.legend()
        plt.title(process)
        plt.xlabel("log($N_q$)")
        plt.ylabel("log($\sigma$)")
        plt.tight_layout()
        plt.show()
        plt.savefig("Nq_"+fname.replace("csv", "png"))
    
    return log_depth_range, classical_fits, quantum_fits, log_Nq_range, log_classical_std, log_quantum_std, quantum_fits_Nq, classical_fits_Nq, quantum_r_squared, classical_r_squared


def calc_r_square_value(best_fit_fn: Callable, y_data: Iterable[float], x_data: Iterable[float], best_fit_params: Iterable[float]) -> float:
    """calculate the r2 value of the best fit

    Args:
        best_fit_fn (Callable): best fit function
        y_data (Iterable[float]): list of y values
        x_data (Iterable[float]): list of x values
        best_fit_params (Iterable[float]): list of best fit parameters

    Returns:
        float: r2 value
    """
    residuals = y_data - best_fit_fn(x_data, *best_fit_params)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data-np.mean(y_data))**2)
    r_squared = 1 - (ss_res/ss_tot)
    return r_squared

def transform_function(x:np.ndarray) -> np.ndarray:
    """transform the array of floats

    Args:
        x (np.ndarray): data vector

    Returns:
        np.ndarray: array of transformed data vector
    """
    return np.log(x)

def linear_function(x:np.ndarray, a: float, b: float) -> np.ndarray:
    """linear function

    Args:
        x (np.ndarray): data vector
        a (float): coefficient
        b (float): constant

    Returns:
        np.ndarray: dependent data vector
    """
    return a*x + b

def plot_data(path: str, fname: str):
    """plots the probability against number of grover iterators

    Args:
        path (str): path to the file
        fname (str): file name
    """
    data = load_data(path+fname)
    max_depth_range = data[:,0]
    classical_estimates = data[:,1]
    classical_std = data[:,2]
    quantum_estimates = data[:,4]
    quantum_std = data[:,5]
    true_prob_line = data[:,-1]
    
    plt.figure(figsize=(5.5,5))
    plt.errorbar(max_depth_range, quantum_estimates, yerr = quantum_std, fmt = 'o-', capsize = 5, label = 'quantum')
    plt.errorbar(max_depth_range, classical_estimates, yerr = classical_std, fmt = 'o-', capsize = 5, label = 'classical')
    plt.plot(max_depth_range, true_prob_line, 'o-', label = 'true')
    plt.legend()
    plt.xlabel('Number of Grover iterators')
    plt.ylabel('Probability')
    # plt.title('p = {}, initial state = {}, sequence = {}'.format(0.4, 0,'0000'))
    plt.tight_layout()
    plt.show()
    plt.savefig(fname.replace("csv", "png"))

def plot_error_trend(path: str, fname: str):
    """plots the standard error against number of grover iterators

    Args:
        path (str): path to the file
        fname (str): file name
    """

    data = load_data(path+fname)
    max_depth_range = data[:,0]
    classical_std = data[:,2]
    quantum_std = data[:,5]
    
    plt.figure(figsize=(5.5,5))
    plt.loglog(max_depth_range, quantum_std, 'o-',label = 'quantum')
    plt.loglog(max_depth_range, classical_std,'o-', label = 'classical')
    plt.legend()
    plt.xlabel('Number of Grover iterators')
    plt.ylabel('Standard deviation')
    # plt.title('p = {}, initial state = {}, sequence = {}'.format(0.4, 0,'0000'))
    # plt.title('p = {}, $q_1$ = {}, $q_2$ = {}, sequence = {}, shots = {}'.format(0.1, 0.1, 0.2, '0000', 100))
    plt.tight_layout()
    plt.show()
    # plt.savefig(fname.replace("csv", "png"))
    
def plot_absolute_error(path: str, fname: str):
    """plots the absolute error against the number of causal states

    Args:
        path (str): path to the file
        fname (str): file name
    """
    data = load_data(path+fname)
    max_depth_range = data[:,0]
    classical_estimates = data[:,1]
    quantum_estimates = data[:,4]
    true_prob_line = data[:,-1]
    
    classical_error = np.abs(classical_estimates - true_prob_line)
    quantum_error = np.abs(quantum_estimates - true_prob_line)
    plt.figure(figsize=(5.5,5))
    plt.plot(max_depth_range, quantum_error, 'o-',label = 'quantum')
    plt.plot(max_depth_range, classical_error,'o-', label = 'classical')
    plt.legend()
    plt.xlabel('Number of Grover iterators')
    plt.ylabel('Absolute error')
    # plt.title('p = {}, initial state = {}, sequence = {}'.format(0.4, 0,'0000'))
    # plt.title('p = {}, $q_1$ = {}, $q_2$ = {}, sequence = {}, shots = {}'.format(0.1, 0.1, 0.2, '0000', 100))
    plt.tight_layout()
    plt.show()
    # plt.savefig(fname.replace("csv", "png"))
    
    

def main():
    relative_path_1 = "../data/perturbed_coin/error_analysis/quantum_v_classical_sampling_for_sample_0000_p_0.1_shots_100_max_depth_30_sample_size_1000_method_LIS.csv"
    relative_path_2 = "../data/perturbed_coin/error_analysis/quantum_v_classical_sampling_perturbed_coin_sequence_0000_p_0.1_shots_100_max_depth_36_sample_size_1000_method_PIS.csv"
    second_index = relative_path_1.find("/", 8)
    process = relative_path_1[8:second_index]
    path = "../data/"+process+ "/error_analysis/"
    print(path)
    process = process.replace("_", " ")
    print(process)
    fname1 = relative_path_1[len(path):]
    fname2 = relative_path_2[len(path):]
    fit_two_data(linear_function, process, path, fname1, fname2)
# =============================================================================
#     relative_path_1 = "data/nemo/error_analysis/nemo_quantum_v_classical_sampling_for_sequence_0000_p_0.2_sample_size_1000_shots_100_max_depth_30_method_LIS.csv"
#     second_index = relative_path_1.find("/", 5)
#     process = relative_path_1[5:second_index]
#     path = "data/"+process+ "/error_analysis/"
#     print(path)
#     process = process.replace("_", " ")
#     fname = relative_path_1[len(path):]
#     # plot_data(path, fname, process)
#     # plot_error_trend(path, fname, process)
#     plot_absolute_error(path, fname, process)
# =============================================================================
    
    

if __name__ == "__main__":
    main()
