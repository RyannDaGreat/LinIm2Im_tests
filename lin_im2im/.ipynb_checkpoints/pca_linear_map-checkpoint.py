import numpy as np
from pca import aligned_pca
import faiss
import time

def faiss_search_single(faiss_thing,matrix,index):
    return faiss_thing.search(matrix[index:index+1,:],1)
def faiss_search(faiss_thing,matrix):
    import numpy as np
    dists,indices=zip(*[faiss_search_single(faiss_thing,matrix,i) for i in range(len(matrix))])
    dists=np.concatenate(dists)
    indices=np.concatenate(indices)
    return dists,indices
    
def debug_print(*args):
    pass
#     print(*args,flush=True)

    
class PCALinMapping:
    """
    Learn a supervised or unsupervised linear mapping between the PCA embedding of two domains.
    See https://arxiv.org/pdf/2007.12568.pdf for details.
    Options in args:
      n_components: Number of PCA components (eigenvectors) to use - the dimension of the PCA representation.
      pairing: 'paired' = Skip ICP and just compute Q (i.e. supervised). Otherwise - perform ICP iterations.
      matching: ICP matching method: 'nn' = Regular nearest-neighbors. 'cyc-nn' = use cycle-consistent pairs only.
      transform_type: 'orthogonal' = constrain the linear transformation to be orthogonal. 'linear' = least squares.
      n_iters: Max number of ICP iterations
    """
    def __init__(self, args=None):
        self.args = args
        self.fitted = False
        self.pca_b = self.pca_a = self.Q = None

    
    def fit(self, x_a, x_b, res=None):
        """
        Perform PCA on the two domains and learn the linear transformation.
        :param x_a: Samples from A [m, d] (rows = samples)
        :param x_b: Samples from B [m, d] (rows = samples)
        :param res: Optional GPU resource for faiss. Can be used if called multiple times.
        """
        print('Got {} samples in A and {} in B.'.format(x_a.shape[0], x_b.shape[0]))
        t0 = time.time()
        self.pca_a, self.pca_b = aligned_pca(x_a, x_b, comps=self.args.n_components)
        z_a = self.pca_a.transform(x_a)
        z_b = self.pca_b.transform(x_b)
        print('PCA representations: ', z_a.shape, z_b.shape, 'took:', time.time()-t0)

        Q = np.eye(self.args.n_components, dtype=np.float32)

        if res is None:
            res = faiss.StandardGpuResources()
        nbrs_b = faiss.GpuIndexFlatL2(res, self.args.n_components)
        nbrs_b.add(z_b)

        print('Learning {} transformation using {} sets:'.format(self.args.transform_type, self.args.pairing))
        for it in range(self.args.n_iters):
            t0 = time.time()

            # Step 1 - Matching
            if self.args.pairing == 'paired':
                if it > 0:
                    break
                assert z_a.shape == z_b.shape
                A, B = z_a, z_b
            else:
                print('Iter {}: '.format(it), end='')
                # Find nearest-neighbors to z_A Q in B:
                debug_print("DEBUG",line_number())
                temp=z_a @ Q
                debug_print("DEBUG",line_number())
#                 pseudo_terminal()
                d_qa_to_b, i_qa_to_b = faiss_search(nbrs_b,temp)
#                 d_qa_to_b, i_qa_to_b = nbrs_b.search(temp, 1) #I have to change this; FAISS segfaults on my computer when you feed the search function long matrices for some reason. Idk why. But feeding it small matrices doesn't crash it so I made a function to make this work
                debug_print("DEBUG",line_number())
                i_qa_to_b = i_qa_to_b.squeeze()
                debug_print("DEBUG",line_number())

                if self.args.matching == 'nn':
                    debug_print("DEBUG",line_number())
                    A = z_a
                    debug_print("DEBUG",line_number())
                    B = z_b[i_qa_to_b]
                    debug_print("DEBUG",line_number())
                    print('Found {} NNs. Mean NN l2 = {:.3f}. '.format(len(np.unique(i_qa_to_b)),
                                                                       np.mean(d_qa_to_b)), end='')
                else:
                    debug_print("DEBUG",line_number())
                    # Find nearest-neighbors in the reverse direction, for cycle-consistency:
                    sel_b = np.unique(i_qa_to_b)
                    debug_print("DEBUG",line_number())
                    assert len(sel_b) > 100, 'Only {} unique NNs'.format(len(sel_b))
                    debug_print("DEBUG",line_number())
                    nbrs_aQ = faiss.GpuIndexFlatL2(res, self.args.n_components)
                    debug_print("DEBUG",line_number())
                    nbrs_aQ.add(z_a @ Q)
                    debug_print("DEBUG",line_number())
                    _d_iqb_to_a, _i_iqb_to_a = faiss_search(nbrs_aQ,z_b[sel_b])
#                     _d_iqb_to_a, _i_iqb_to_a = nbrs_aQ.search(z_b[sel_b], 1)
                    
                    debug_print("DEBUG",line_number())
                    i_iqb_to_a = -np.ones(shape=[z_b.shape[0]], dtype=int)
                    debug_print("DEBUG",line_number())
                    i_iqb_to_a[sel_b] = _i_iqb_to_a.squeeze()
                    debug_print("DEBUG",line_number())
                    # Check for cycle-consistency
                    debug_print("DEBUG",line_number())
                    cyc_consistent_a = i_iqb_to_a[i_qa_to_b] == np.arange(len(i_qa_to_b))
                    debug_print("DEBUG",line_number())
                    if np.count_nonzero(cyc_consistent_a) < 1000:
                        debug_print("DEBUG",line_number())
                        print('(only {} consisten pairs) '.format(np.count_nonzero(cyc_consistent_a)), end='')
                        debug_print("DEBUG",line_number())
                        cyc_consistent_a = np.ones_like(cyc_consistent_a)
                        debug_print("DEBUG",line_number())
                    A = z_a[cyc_consistent_a]
                    debug_print("DEBUG",line_number())
                    B = z_b[i_qa_to_b[cyc_consistent_a]]
                    debug_print("DEBUG",line_number())
                    print('{} B-NNs / {} consistent, mean NN l2 = {:.3f}. '.format(len(sel_b),
                        np.count_nonzero(cyc_consistent_a), np.mean(d_qa_to_b[cyc_consistent_a])), end='')

            # Step 2 - Mapping (updating Q):
            prev_Q = Q
            debug_print("DEBUG",line_number())
            if self.args.transform_type == 'orthogonal':
                debug_print("DEBUG",line_number())
                U, S, V = np.linalg.svd(A.T @ B)
                debug_print("DEBUG",line_number())
                Q = U @ V
                debug_print("DEBUG",line_number())
            else:
                debug_print("DEBUG",line_number())
                Q = np.linalg.inv(A.T @ A) @ A.T @ B
                debug_print("DEBUG",line_number())

            debug_print("DEBUG",line_number())
            if np.allclose(Q, prev_Q):
                debug_print("DEBUG",line_number())
                print('Converged - terminating ICP iterations.')
                debug_print("DEBUG",line_number())
                break

            debug_print("DEBUG",line_number())
            print('took {:.2f} sec.'.format(time.time()-t0))
            debug_print("DEBUG",line_number())

        debug_print("DEBUG",line_number())
        self.fitted = True
        debug_print("DEBUG",line_number())
        self.Q = Q
        debug_print("DEBUG",line_number())
        return self

    def transform_a_to_b(self, x_a, _Q=None, n_comps=None):
        """
        Apply the learned linear transformation
        :param x_a: The samples to be transformed
        :param _Q: Option to provide the transformation matrix
        :param n_comps: Use only the first n_comps PCA coefficients
        :return: T(x_a)
        """
        assert self.fitted or _Q is not None
        n_comps = n_comps or self.pca_a.components_.shape[0]
        Q = self.Q if _Q is None else _Q
        mu_a = self.pca_a.mean_.reshape(1, -1)
        mu_b = self.pca_b.mean_.reshape(1, -1)
        z_a = (x_a - mu_a) @ self.pca_a.components_[:n_comps].T
        z_ab = z_a @ Q[:n_comps, :n_comps]
        return z_ab @ self.pca_b.components_[:n_comps] + mu_b

    def reconstruct_a(self, x_a, n_comps=None):
        """
        Represent x_a in its PCA space and reconstruct it back (just to see the PCA reconstruction quality)
        :param x_a: Samples from A
        :param n_comps: Use only the first n_comps PCA coefficients
        :return: Reconstructed samples
        """
        n_comps = n_comps or self.pca_a.components_.shape[0]
        mu_a = self.pca_a.mean_.reshape(1, -1)
        return (x_a - mu_a) @ self.pca_a.components_[:n_comps].T @ self.pca_a.components_[:n_comps] + mu_a

    def reconstruct_b(self, x_b, n_comps=None):
        """
        Same as reconstruct_a, but for domain B
        """
        n_comps = n_comps or self.pca_b.components_.shape[0]
        mu_b = self.pca_b.mean_.reshape(1, -1)
        return (x_b - mu_b) @ self.pca_b.components_[:n_comps].T @ self.pca_b.components_[:n_comps] + mu_b

