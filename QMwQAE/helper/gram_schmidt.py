import numpy as np

class Gram_schmidt(object):
    def __init__(self, dim: int, start_vectors = []):
        """implements the gram schmidt process

        Args:
            dim (int): number of dimensions
            start_vectors (list, optional): given starting vectors. Defaults to [].
        """
        self.dim = dim
        self.vector_list = []
        self.num_start_vectors = len(start_vectors)
        if not start_vectors:
            return
        else:
            for vec in start_vectors:
                self.vector_list.append(vec)

    def next_vector(self, dim: int):
        """generate the next orthogonal vector

        Args:
            dim (int): number of dimensions
        """
        if not self.vector_list:
            new_vector = np.random.rand(dim, )
            new_vector = new_vector/np.linalg.norm(new_vector)
            self.vector_list.append(new_vector)
            return
        while True:
            new_vector = np.random.rand(dim, )
            for _ in range(3):
                for vec in self.vector_list:
                    new_vector = new_vector - vec.dot(new_vector)/np.linalg.norm(vec) * vec
            if np.linalg.norm(new_vector) != 0.:
                new_vector = new_vector/np.linalg.norm(new_vector)
                self.vector_list.append(new_vector)
                return
    def generate_vectors(self):
        """generate next orthogonal vector
        """
        while len(self.vector_list) < self.dim:
            self.next_vector(self.dim)

    def build_unitary(self) -> np.ndarray:
        """creates the unitary matrix

        Returns:
            np.ndarray: unitary matrix
        """
        unitary = np.zeros((self.dim, self.dim), dtype = float)
        for i in range(self.num_start_vectors):
            unitary[:,i*2] = self.vector_list[i]
        for j in range(self.dim -self.num_start_vectors):
            unitary[:,j*2-1] = self.vector_list[j + self.num_start_vectors]
        initial_det = np.linalg.det(unitary)
        if initial_det < 0:
            unitary[:,1] = unitary[:,1] * -1
        self.unitary = unitary
        return unitary


if __name__ == "__main__":
    dim = 3
    first_vector = np.array([1,0,0], dtype = float)
    second_vector = np.array([0,1,0], dtype = float)
    all_vectors = [first_vector]
    my_gs = Gram_schmidt(dim, all_vectors)
    my_gs.generate_vectors()
    print(my_gs.vector_list)
    for i in range(dim):
        for j in range(1, dim):
            print(i,j)
            if i == j:
                print(my_gs.vector_list[i].dot(my_gs.vector_list[j]))
            else:
                print(my_gs.vector_list[i].dot(my_gs.vector_list[j]))
                
    # print(my_gs.vector_list[0].dot(my_gs.vector_list[1]))
    