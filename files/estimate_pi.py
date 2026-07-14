
import numpy as np
import matplotlib.pyplot as plt

def gen_2d_pos(num:int) -> np.ndarray:
    """
    Generate 2d positions in ] 0, 1 [.

    Parameters
    ----------
    num : int
        Number of samples.

    Returns
    -------
    np.ndarray
        ``num`` 2d positions in ] 0, 1 [.

    """
    return np.random.rand(num, 2)

def estim_pi_one_pop(num:int) -> float:
    """
    Estimate pi throug ] 0, 1 [ random position sampling using one only
    population.

    Parameters
    ----------
    num : int
        Number of samples.

    Returns
    -------
    estimation : float
        Pi estimation.

    """
    # sampling
    samples = gen_2d_pos(num)

    # turn into distances
    dist = np.sum(samples**2, axis=1)**0.5

    # count the number of samples which fall in the [0, 1] distance area
    distrib = np.bincount(dist <= 1, minlength=2)

    # Compute the estimation
    estimation = 4 * distrib[1] / num

    return estimation

def estim_pi_multi_pop(num_samples:int, num_populations:int,
                       return_populations:bool=False) -> float:
    """
    Estimate pi throug ] 0, 1 [ random position sampling using distribution
    over multiple populations.

    Parameters
    ----------
    num_samples : int
        Number of samples per population.
    num_populations : int
        Number of population.
    return_populations : bool, optional
        If are also returned the per population pi estimate. The default is
        False.

    Returns
    -------
    estimations : dict
        Pi estimations, with estimations.mean, estimations.median,
        estimations.std.
    populations : np.ndarray, optional
        Per population pi estimate. Returned if return_populations = True.

    """
    # sampling over multiple populations
    samples = np.random.rand(num_populations, num_samples, 2)

    # turn into distances -> (num_populations, num_samples)
    dist = np.sum(samples**2, axis=2)**0.5

    # count the number of samples which fall in the [0, 1] distance area per
    # population
    distrib = np.zeros((num_populations, 2), dtype=int)
    for i in range(0, num_populations, 1):
        distrib[i] = np.bincount(dist[i] <= 1, minlength=2)

    # Compute the estimation
    populations = 4 * distrib[:, 1] / num_samples

    # Compute statistics
    estimations = {'mean':np.mean(populations),
                   'median':np.median(populations),
                   'std':np.std(populations)}

    if return_populations:
        return estimations, populations

    return estimations

def show_distribution(populations:np.ndarray, nbins:int=10) -> None:
    """
    Display the distribution of pi estimation trhough multiple populations.

    Parameters
    ----------
    populations : np.ndarray
        Per population pi estimate.
    nbins : int, optional
        Number of bins. The default is 10.

    Returns
    -------
    None

    """
    mean = np.mean(populations)
    median = np.median(populations)

    counts, edges = np.histogram(populations, bins=nbins)
    width = edges[1]-edges[0]

    centers = (edges[1:]+edges[:-1])/2

    plt.figure()
    plt.grid(True, which='both', zorder=1)
    plt.bar(centers, counts, width=width*0.75)

    plt.xlabel(r'$\pi$ Estimation', fontsize=12)
    plt.ylabel('# Counts', fontsize=12)

    plt.vlines(mean, 0, np.max(counts)*1.05, color='green', label='Average')
    plt.vlines(median, 0, np.max(counts)*1.05, color='red', label='Median')
    plt.legend(fontsize=12)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
