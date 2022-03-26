import numpy as np
from scipy.linalg import sqrtm


def eigenvector_compute(A, runs = 1000):
    eigen_left = np.eye(A.shape[1], dtype = float) # * 1/A.shape[1]
    eigen_right = eigen_left.copy()  * 1/A.shape[1]

    for _ in range(runs):
        temp1 = np.zeros(eigen_left.shape, dtype = float)
        temp2 = np.zeros(eigen_left.shape, dtype = float)
        for j in range(A.shape[0]):
            temp1 += (A[j].transpose()).dot(eigen_left.dot(A[j]))
            temp2 += A[j].dot(eigen_right.dot(A[j].transpose()))
        eigen_left = temp1.copy()
        eigen_right = temp2.copy()
    return eigen_left, eigen_right

def psi_constructor(eigen_left, eigen_right):
    W_r  = np.sqrt(eigen_right) # eigen_right is diagonal matrix
    svd_eigen_left = np.linalg.svd(eigen_left)
    W_l = svd_eigen_left[0].dot(np.diag(np.sqrt(svd_eigen_left[1]/np.trace(eigen_left))).dot(svd_eigen_left[2].transpose()))
    # W_l = sqrtm(eigen_left)
    basis = np.eye(eigen_left.shape[0])
    states = np.zeros(eigen_left.shape, dtype = float)
    for i in range(states.shape[0]):
        states[i] = W_l.transpose().dot(basis[i].reshape(states.shape[0], 1)).reshape(states.shape[0],)
    psi = 0
    for i in range(states.shape[0]):
        psi += W_r[i,i] * np.kron(basis[i].reshape(states.shape[0],1), np.conjugate(states[i]).reshape(states.shape[0],1))
    psi = psi.reshape(eigen_left.shape)
    singular_values = np.linalg.svd(psi)
    entropy = 0
    for i in range(len(singular_values[1])):
        entropy += singular_values[1][i]**2 * np.log2(singular_values[1][i]**2)
    return -entropy, W_l


def main(p, process):

    ######################
    # Nemo process definitions
    if process == "nemo":
        T = np.array([[p, 0, 1],
                    [1-p, 0, 0],
                    [0,1,0]], dtype = float)

        A = np.zeros((T.shape[0], T.shape[0], T.shape[1]), dtype = float)
        A[0] = np.array([
            [np.sqrt(p), 0, 1/np.sqrt(2)],
            [0, 0, 0],
            [0,0,0]
        ], dtype = float)

        A[1] = np.array([
            [0, 0, 1/np.sqrt(2)],
            [np.sqrt(1-p), 0, 0],
            [0,1,0]
        ], dtype = float)

    ####################
    # perturbed coin definitions
    if process == "perturbed_coin":
        T = np.array([[p, 1-p],
                [1-p, p]], dtype = float)

        A = np.zeros((T.shape[0], T.shape[0], T.shape[1]), dtype = float)

        A[0] = np.array([
            [np.sqrt(p), np.sqrt(1-p)],
            [0,0]
        ], dtype = float)

        A[1] = np.array([
            [0,0],
            [np.sqrt(1-p), np.sqrt(p)]
        ], dtype = float)

    eigen_left, eigen_right = eigenvector_compute(A)

    print('Vl')
    print(eigen_left)
    print("Vr")
    print(eigen_right)
    entropy, W_l = psi_constructor(eigen_left, eigen_right)

    kraus = np.zeros(A.shape, dtype = float)
    for i in range(A.shape[0]):
        kraus[i] = W_l.dot(A[i].dot(np.linalg.inv(W_l)))
    # print("psi \n", psi)
    return entropy, kraus

if __name__ == "__main__":
    # process = input("choose process (nemo, perturbed_coin): ")
    # p = float(input("choose p: "))
    process = "perturbed_coin"
    p = 0.3
    # entropy_list = []
    # prob_list = [x/1000 for x in range(1, 1000)]
    # for p in prob_list:
    #     entropy = main(p, process)
    #     entropy_list.append(entropy)
    entropy, kraus = main(p, process)
    print("K0:\n", kraus[0])
    print("K1:\n", kraus[1])
    verification = kraus[0].transpose().dot(kraus[0]) + kraus[1].transpose().dot(kraus[1])
    print(verification)
    # plt.clf()
    # plt.plot(prob_list, entropy_list)
    # plt.xlabel("probability")
    # plt.ylabel("entropy")
    # plt.title(process)
    # plt.show()



        