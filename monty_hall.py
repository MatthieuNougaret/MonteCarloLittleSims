# Monthy-all simulation

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def simulations(n_draw:int, n_portes:int) -> np.ndarray:
    """
    Run a generalised Monty Hall simulation and return the win rates
    for both the 'keep' and 'switch' strategies.

    In each round:
      1. A winning door is placed at random among n_portes doors.
      2. The contestant picks a door at random.
      3. The host reveals one losing door that is neither the winner
         nor the contestant's current pick.
      4. The contestant switches to one of the remaining closed doors
         (chosen at random if several are available).

    Parameters
    ----------
    n_draw : int
        Number of rounds to simulate.
    n_portes : int
        Total number of doors (>= 3).

    Returns
    -------
    rate : np.ndarray, shape (2,)
        rate[0] – win prortion when keeping the first choice.
        rate[1] – win prortion when switching after the reveal.

    """
    # [keep_wins, switch_wins]
    rate = np.array([0, 0])

    # door indices [0, 1, ..., n_portes-1]
    kernel = np.arange(n_portes)
    for i in range(n_draw):
        # doors hiding the prize
        winner = np.random.randint(0, n_portes)
        # contestant's first pick
        choice = np.random.randint(0, n_portes)
        # Keep strategy: count a win before any switch occurs
        if winner == choice:
            rate[0] += 1

        # Host reveal: open a door that is neither winner nor chosen
        reveal = kernel[(kernel != winner)&(kernel != choice)]
        if len(reveal) > 1:
            # Multiple eligible doors: the host picks one at random
            reveal = int(reveal[np.random.randint(0, len(reveal))])
        else:
            reveal = int(reveal[0])

        # Switch strategy: move to any remaining closed door
        choice = kernel[(kernel != choice)&(kernel != reveal)]
        if len(choice) > 1:
            # Several doors left: pick one at random
            choice = int(choice[np.random.randint(0, len(choice))])
        else:
            choice = int(choice[0])

        # Switch strategy results: count a win after switching
        if winner == choice:
            rate[1] += 1

    return rate / n_draw

def multidoor_simulation(n_draw:int, num_doors:np.ndarray) -> np.ndarray:
    """
    Run Monty-Hall simulations for various number of doors.

    Parameters
    ----------
    n_draw : int
        Number of rounds to simulate.
    num_doors : np.ndarray
        Vector with the various number of doors (>= 3).

    Returns
    -------
    rates : np.ndarray
        2 columns vector with the proportion of wins. Shape: (num_doors, 2).
        rates[:, 0] – win prortion when keeping the first choice.
        rates[:, 1] – win prortion when switching after the reveal.

    """
    rates = np.zeros((len(num_doors), 2))
    for i in tqdm(range(len(num_doors))):
        rates[i] = simulations(n_draw, num_doors[i])

    return rates

def distrib_one_case(results:np.ndarray, n_draw:int, n_doors:int,
                     nbins:int=10, figsize:tuple=(12, 6)) -> None:
    """
    Diplay the win proportion of ``keep`` and ``switch`` strategy

    Parameters
    ----------
    results : np.ndarray
        2 columns vector with the proportion of wins. Shape: (n_simulations, 2).
        rates[:, 0] – win prortion when keeping the first choice.
        rates[:, 1] – win prortion when switching after the reveal.
    n_draw : int
        Number of rounds to simulate.
    n_doors : int
        Total number of doors (>= 3).
    nbins : int, optional
        Number of bins. The default is 10.
    figsize : tuple, optional
        Figure size (width, height). The default is (12, 6).

    Returns
    -------
    None

    """
    means = np.mean(results, axis=0)
    medians = np.median(results, axis=0)

    counts_k, edges_k = np.histogram(results[:, 0], bins=nbins)
    width_k = edges_k[1]-edges_k[0]
    centers_k = (edges_k[1:]+edges_k[:-1])/2

    counts_s, edges_s = np.histogram(results[:, 1], bins=nbins)
    width_s = edges_s[1]-edges_s[0]
    centers_s = (edges_s[1:]+edges_s[:-1])/2

    h = max([np.max(counts_k), np.max(counts_s)]) * 1.05

    title = 'Win proportion distribution for: '
    plt.figure(figsize=figsize)
    plt.title(title + f'{n_doors} doors with {n_draw} simulation(s)',
              fontsize=12)

    plt.grid(True, which='both', zorder=1)
    plt.bar(centers_k, counts_k, width=width_k*0.75, color='tab:blue',
            alpha=0.75, label='Keep strategy', zorder=2)

    plt.bar(centers_s, counts_s, width=width_s*0.75, color='tab:orange',
            alpha=0.75, label='Switch strategy', zorder=2)

    plt.xlabel('Win Proportion', fontsize=12)
    plt.ylabel('# Counts', fontsize=12)

    plt.vlines(means[0], 0, h, color='green', label='Average keep')
    plt.vlines(medians[0], 0, h, color='red', label='Median keep')
    plt.vlines(means[1], 0, h, color='purple', label='Average switch')
    plt.vlines(medians[1], 0, h, color='k', label='Median switch')

    plt.legend(fontsize=12)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

def multidoor_simulation_plot(results:np.ndarray, door_numbs:np.ndarray
                              ) -> None:
    """
    Display the curves of the ``keep`` and ``switch`` strategy in function of
    the number of doors.

    Parameters
    ----------
    results : np.ndarray
        2 columns vector with the proportion of wins. Shape: (num_doors, 2).
        rates[:, 0] – win prortion when keeping the first choice.
        rates[:, 1] – win prortion when switching after the reveal.
    door_numbs : np.ndarray
        Vector with the various number of doors (>= 3).

    Returns
    -------
    None

    """
    plt.figure(figsize=(12, 6))
    plt.grid(True, which='both', zorder=1)
    plt.plot(door_numbs, results[:, 0], 'ko:', zorder=2, label='Keep')
    plt.plot(door_numbs, results[:, 1], 'b^:', zorder=2, label='Switch')
    plt.xlabel('Number of doors', fontsize=14)
    plt.ylabel('Win probability', fontsize=14)
    plt.legend(title='Strategy', fontsize=14, title_fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()
