import random
import simpy
from matplotlib import pyplot as plt
import numpy as np
import math


RANDOM_SEED = 41
NEW_REQUESTS = 1000  # Total number of customers
INTERVAL_REQUESTS = 0.9  # Generate new customers roughly every x seconds
MIN_PATIENCE = 9998  # Min. customer patience
MAX_PATIENCE = 9999  # Max. customer patience
PROCESSING_TIME = 1

WAIT = []
ARRIVE = []
LONG_SHORT = []

def source(env, number, interval, counter):
    """Source generates customers randomly"""
    for i in range(number):
        c = customer(env, 'Request%02d' % i, counter, time_in_bank=PROCESSING_TIME)
        env.process(c)
        t = random.expovariate(1.0 / interval)
        yield env.timeout(t)


def customer(env, name, counter, time_in_bank):
    """Customer arrives, is served and leaves."""
    arrive = env.now

    with counter.request() as req:
        patience = random.uniform(MIN_PATIENCE, MAX_PATIENCE)
        # Wait for the counter or abort at the end of our tether
        results = yield req | env.timeout(patience)

        wait = env.now - arrive

        global WAIT
        WAIT.append(wait)

        if req in results:
            # We got to the counter
            tib = random.expovariate(1.0 / (pick_ls()*time_in_bank))
            yield env.timeout(tib)

        else:
            # We reneged
            print('%7.4f %s: RENEGED after %6.3f' % (env.now, name, wait)) #this should not be called


# Setup and start the simulation
#random.seed(RANDOM_SEED)
env = simpy.Environment()

def pick_ls():
    random_index = random.randint(0, len(LONG_SHORT) - 1)
    long_or_short = LONG_SHORT[random_index]
    LONG_SHORT.pop(random_index)
    return long_or_short


def calc_mean_wait(): # calculate mean service time (wait-time)
    global WAIT
    calc = np.mean(WAIT)
    WAIT = []
    return calc

def long_short():
    global LONG_SHORT

    long = 3
    short = 1
    amount_short = int(0.75*NEW_REQUESTS)
    amount_long = int(0.25*NEW_REQUESTS)

    long_list = [long for _ in range(amount_long)]
    short_list = [short for _ in range(amount_short)]

    LONG_SHORT = long_list + short_list


# Start processes and run
def create_graph():
    cap_list = [1, 2, 4]                        #capacity
    colour = ['blue', 'red', 'yellow', 'green'] #colours for the graphs
    sample_size = 2500                          #itterations/simulations per requist

    request_range = [num / 1000 for num in range(1150) if num >= 1000]

    for cap in cap_list:
        average_rho_total = []
        mean_wait_total = []

        global PROCESSING_TIME
        PROCESSING_TIME = cap

        for request_amount in request_range:

            mean_wait = []

            for _ in range(sample_size):
                long_short()
                counter = simpy.Resource(env, capacity=cap)
                env.process(source(env, NEW_REQUESTS, request_amount, counter))
                env.run() #simulate for one

                mean_wait.append(calc_mean_wait())

            mean_arrival = 1 / request_amount  # could be set for a calculation
            mean_service = 1 / PROCESSING_TIME  # could be set for a calculation

            calc_rho = mean_arrival / (cap * mean_service)

            average_rho_total.append(calc_rho)
            mean_wait_total.append(mean_wait)



        x = average_rho_total
        y = [np.mean(i) for i in mean_wait_total]
        plt.plot(x, y, color=colour[cap-1]) # plots average line

        confidence95 = [1.96 * (np.std(j) / math.sqrt(NEW_REQUESTS)) for j in mean_wait_total]

        lower_bound = list(np.array(y) - np.array(confidence95))
        upper_bound = list(np.array(y) + np.array(confidence95))
        plt.fill_between(x, (lower_bound), (upper_bound), color=colour[cap-1], alpha=0.1) # plots standard deviation

    plt.title('M/M/n long tail mean waiting time versus the system load')
    plt.xlabel('rho')
    plt.ylabel('mean waiting time')
    plt.show()

    plt.show()

create_graph()