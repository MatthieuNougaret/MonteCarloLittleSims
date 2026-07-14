import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def simulate_one_room(n_people:int) -> int:
    """
    Function to generate ``n_people`` random birthday and compute the
    number birthday sampled more than one times.

    Parameters
    ----------
    n_people : int
        Number of random birthday.

    Returns
    -------
    num_collision : int
        Number of birthday which were sampled more than one time.

    """
    rand = np.random.randint(0, 366, n_people)
    hist = np.bincount(rand, minlength=366)
    num_collision = int(np.sum(hist > 1))
    return num_collision

def simulate_multi_room(n_people:int, n_room:int) -> np.ndarray:
    """
    Function to simulate multiple rooms and compute the number of birthday
    collisions for each room.

    Parameters
    ----------
    n_people : int
        Number of random birthday per room.
    n_room : int
        Number of rooms to simulate.

    Returns
    -------
    num_collision : np.ndarray
        Array containing the number of birthday collisions for each room.

    """
    rand = np.random.randint(0, 366, size=(n_room, n_people))
    results = np.zeros((n_room, 366))
    for i in range(n_room):
        results[i] = np.bincount(rand[i], minlength=366)

    num_collision = np.sum(results > 1, axis=1)
    return num_collision

def simulate_multi_room_people(peoples:np.ndarray, n_room:int) -> np.ndarray:
    """
    Function to simulate multiple rooms for varying group sizes and compute 
    the probability of having at least one birthday collision.

    Parameters
    ----------
    peoples : np.ndarray
        Array containing different numbers of people per room to test.
    n_room : int
        Number of rooms to simulate for each group size.

    Returns
    -------
    rate : np.ndarray
        Array containing the collision probabilities corresponding to each
        group size.

    """
    rate = np.zeros(len(peoples))
    for i in range(len(peoples)):
        results = simulate_multi_room(int(peoples[i]), n_room)
        rate[i] = np.bincount(results > 0, minlength=2)[1]/n_room

    return rate

def show_rates_peoples(peoples:np.ndarray, rates:np.ndarray) -> None:
    """
    Function to plot the probability of obtaining a birthday collision 
    against the number of people per room.

    Parameters
    ----------
    peoples : np.ndarray
        Array containing the numbers of people per room.
    rates : np.ndarray
        Array containing the corresponding collision probabilities.

    Returns
    -------
    None

    """
    plt.figure(figsize=(12, 6))
    plt.grid(True, which='both', zorder=1)
    plt.plot(peoples, rates, 'b^--')
    plt.xlabel('Number of people per room', fontsize=14)
    plt.ylabel('Birthday collision proportion', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()
