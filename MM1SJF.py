import random
import simpy
from matplotlib import pyplot as plt
import numpy as np
import bisect
import math


RANDOM_SEED = 41
NEW_REQUESTS = 1000  # Total number of customers
INTERVAL_REQUESTS = 0.9  # Generate new customers roughly every x seconds
MIN_PATIENCE = 9998  # Min. customer patience
MAX_PATIENCE = 9999  # Max. customer patience

WAIT = []
ARRIVE = []
QUEUE = []

def source(env, number, interval, counter):
    """Source generates customers randomly"""
    for i in range(number):
        c = customer(env, 'Request%02d' % i, counter, time_in_bank=1.0)
        env.process(c)
        t = random.expovariate(1.0 / interval)
        yield env.timeout(t)


def customer(env, name, counter, time_in_bank):
    """Customer arrives, is served and leaves."""
    arrive = env.now
    #print('%7.4f %s: Request recieved' % (arrive, name))


    with counter.request() as req:

        patience = random.uniform(MIN_PATIENCE, MAX_PATIENCE)

        tib = random.expovariate(1.0 / time_in_bank)

        if len(counter.queue) == 1:
            QUEUE.append(tib)
        elif len(counter.queue) > 1:
            bisect.insort(QUEUE, tib)

        # Wait for the counter or abort at the end of our tether
        results = yield req | env.timeout(patience)

        wait = env.now - arrive

        if len(QUEUE) >= 1:
            tib = QUEUE[0]
            QUEUE.pop(0)

        WAIT.append(wait)

        if req in results:
            # We got to the counter

            yield env.timeout(tib)


        else:
            # We reneged
            print('%7.4f %s: RENEGED after %6.3f' % (env.now, name, wait))#this should not be called


# Setup and start the simulation
#print('Bank renege')
#random.seed(RANDOM_SEED)
env = simpy.Environment()

def calc_mean_wait():
    global WAIT
    calc = sum(WAIT)/NEW_REQUESTS
    #print(calc)
    #print()
    WAIT = []
    return calc

# Start processes and run
def create_graph():
    cap = 1
    sample_size = 2500
    request_range = [num / 1000 for num in range(1150) if num >= 1000]
    print(request_range)


    average_rho_total = []
    mean_wait_total = []

    for request_amount in request_range:
        mean_wait = []

        for _ in range(sample_size):
            global QUEUE
            QUEUE = []
            counter = simpy.Resource(env, capacity=cap)
            env.process(source(env, NEW_REQUESTS, request_amount, counter))
            env.run()

            mean_wait.append(calc_mean_wait())

        mean_arrival = 1 / request_amount  # could be set for a calculation
        mean_service = 1  # could be set for a calculation

        calc_rho = mean_arrival / (cap * mean_service)

        average_rho_total.append(calc_rho)
        mean_wait_total.append(mean_wait)


    x = average_rho_total
    y = [np.mean(i) for i in mean_wait_total]
    plt.plot(x, y, color='red') # plots average line

    confidence95 = [1.96*(np.std(j)/math.sqrt(NEW_REQUESTS)) for j in mean_wait_total]

    lower_bound = list(np.array(y) - np.array(confidence95))
    upper_bound = list(np.array(y) + np.array(confidence95))

    plt.fill_between(x, lower_bound, upper_bound, color='red', alpha=0.1)

    plt.title('M/M/1 SJF mean waiting time versus the system load')
    plt.xlabel('rho')
    plt.ylabel('mean waiting time')
    plt.show()

create_graph()