from .base_metric import BilinearMixin, _TripletsClassifierMixin
import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils import check_array


class OASIS(BilinearMixin, _TripletsClassifierMixin):
  """
  Key params:

  n_iter: Max number of iterations. If n_iter > n_triplets,
  a random sampling of seen triplets takes place to feed the model.

  c: Passive-agressive param. Controls trade-off bewteen remaining
        close to previous W_i-1 OR minimizing loss of current triplet

  seed: For random sampling

  shuffle: If True will shuffle the triplets given to fit.
  """

  def __init__(
          self,
          preprocessor=None,
          n_iter=10,
          c=1e-6,
          random_state=None,
          ):
    super().__init__(preprocessor=preprocessor)
    self.components_ = None  # W matrix
    self.d = 0  # n_features
    self.n_iter = n_iter  # Max iterations
    self.c = c  # Trade-off param
    self.random_state = check_random_state(random_state)

  def fit(self, triplets, shuffle=True, random_sampling=False,
          custom_order=None):
    """
    Fit OASIS model

    Parameters
    ----------
    triplets : (n x 3 x d) array of samples
    shuffle : Whether the triplets should be suffled beforehand
    random_sampling: Sample triplets, with repetition, uniform probability.
    custom_order : User's custom order of triplets to feed oasis.
    """
    # Currently prepare_inputs makes triplets contain points and not indices
    triplets = self._prepare_inputs(triplets, type_of_inputs='tuples')

    # TODO: (Same as SCML)
    # This algorithm is built to work with indices, but in order to be
    # compliant with the current handling of inputs it is converted
    # back to indices by the following fusnction. This should be improved
    # in the future.
    # Output: indices_to_X, X = unique(triplets)
    triplets, X = self._to_index_points(triplets)

    self.d = X.shape[1]  # (n_triplets, d)
    self.n_triplets = triplets.shape[0]  # (n_triplets, 3)

    self.shuffle = shuffle  # Shuffle the trilplets
    self.random_sampling = random_sampling
    # Get the order in wich the algoritm will be fed
    if custom_order is not None:
      self.indices = self._check_custom_order(custom_order)
    else:
      self.indices = self._get_random_indices(self.n_triplets,
                                              self.n_iter,
                                              self.shuffle,
                                              self.random_sampling)

    self.components_ = np.identity(
        self.d) if self.components_ is None else self.components_

    i = 0
    while i < self.n_iter:
        current_triplet = X[triplets[self.indices[i]]]
        loss = self._loss(current_triplet)
        vi = self._vi_matrix(current_triplet)
        fs = self._frobenius_squared(vi)
        # Global GD or Adjust to tuple
        tau_i = np.minimum(self.c, loss / fs)

        # Update components
        self.components_ = np.add(self.components_, tau_i * vi)
        i = i + 1

    return self

  def partial_fit(self, new_triplets):
    """
    self.components_ already defined, we reuse previous fit
    """
    self.fit(new_triplets)

  def _frobenius_squared(self, v):
    """
    Returns Frobenius norm of a point, squared
    """
    return np.trace(np.dot(v, v.T))

  def _loss(self, triplet):
    """
    Loss function in a triplet
    """
    S = -1 * self.score_pairs([[triplet[0], triplet[1]],
                              [triplet[0], triplet[2]]])
    return np.maximum(0, 1 - S[0] + S[1])

  def _vi_matrix(self, triplet):
    """
    Computes V_i, the gradient matrix in a triplet
    """
    # (pi+ - pi-)
    diff = np.subtract(triplet[1], triplet[2])  # Shape (, d)
    result = []

    # For each scalar in first triplet, multiply by the diff of pi+ and pi-
    for v in triplet[0]:
        result.append(v * diff)

    return np.array(result)  # Shape (d, d)

  def _to_index_points(self, o_triplets):
    """
    Takes the origial triplets, and returns a mapping of the triplets
    to an X array that has all unique point values.

    Returns:

    X: Unique points across all triplets.

    triplets: Triplets-shaped values that represent the indices of X.
    Its guaranteed that shape(triplets) = shape(o_triplets[:-1])
    """
    shape = o_triplets.shape  # (n_triplets, 3, n_features)
    X, triplets = np.unique(np.vstack(o_triplets), return_inverse=True, axis=0)
    triplets = triplets.reshape(shape[:2])  # (n_triplets, 3)
    return triplets, X

  def _get_random_indices(self, n_triplets, n_iter, shuffle=True,
                          random=False):
    """
    Generates n_iter indices in (0, n_triplets).

    If not random:

    If n_iter = n_triplets, then the resulting array will include
    all values in range(0, n_triplets). If shuffle=True, then this
    array is shuffled.

    If n_iter > n_triplets, it will ensure that all values in
    range(0, n_triplets) will be included. Then a random sampling
    is executed to fill the gap. If shuffle=True, then the final
    array is shuffled. The sampling may contain duplicates.

    If n_iter < n_triplets, then a random sampling takes place.
    The final array does not contains duplicates. The shuffle
    param has no effect.

    If random:

    A random sampling is made in any case, generating n_iters values
    that may include duplicates. The shuffle param has no effect.
    """
    rng = self.random_state
    if random:
      return rng.randint(low=0, high=n_triplets, size=n_iter)
    else:
      if n_iter < n_triplets:
        return rng.choice(n_triplets, n_iter, replace=False)
      else:
        array = np.arange(n_triplets)  # All triplets will be included
        if n_iter > n_triplets:
          array = np.concatenate([array, rng.randint(low=0,
                                 high=n_triplets,
                                 size=(n_iter-n_triplets))])
        if shuffle:
          rng.shuffle(array)
        return array

  def get_indices(self):
    """
    Returns an array containing indices of triplets, the order in
    which the algorithm was feed.
    """
    return self.indices

  def _check_custom_order(self, custom_order):
    """
    Checks that the custom order is in fact a list or numpy array,
    and has n_iter values in between (0, n_triplets)
    """
    return check_array(custom_order, ensure_2d=False,
                       allow_nd=True, copy=False,
                       force_all_finite=True, accept_sparse=True,
                       dtype=None, ensure_min_features=self.n_iter,
                       ensure_min_samples=0)
