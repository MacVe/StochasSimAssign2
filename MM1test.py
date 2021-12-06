import random
import simpy
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np



RANDOM_SEED = 41
NEW_REQUESTS = 10  # Total number of customers
INTERVAL_REQUESTS = 1.1  # Generate new customers roughly every x seconds
MIN_PATIENCE = 9998  # Min. customer patience
MAX_PATIENCE = 9999  # Max. customer patience
PROCESSING_TIME = 1

WAIT = []
ARRIVE = []

def source(env, number, interval, counter):
    """Source generates customers randomly"""
    print(f'  Queued events: {counter.queue}')
    for i in range(number):
        c = customer(env, 'Request%02d' % i, counter, time_in_bank=PROCESSING_TIME)
        env.process(c)
        t = random.expovariate(1.0 / interval)

        yield env.timeout(t)


def customer(env, name, counter, time_in_bank):
    """Customer arrives, is served and leaves."""
    arrive = env.now
    #print('%7.4f %s: Request recieved' % (arrive, name))



    with counter.request() as req:
        patience = random.uniform(MIN_PATIENCE, MAX_PATIENCE)
        # Wait for the counter or abort at the end of our tether
        results = yield req | env.timeout(patience)

        wait = env.now - arrive

        global WAIT
        WAIT.append(wait)


        global ARRIVE
        ARRIVE.append(arrive)

        if req in results:
            # We got to the counter
            #print('%7.4f %s: Waited %6.3f' % (env.now, name, wait))

            tib = random.expovariate(1.0 / time_in_bank)
            yield env.timeout(tib)
            #print('%7.4f %s: Completed' % (env.now, name))

        else:
            # We reneged
            print('%7.4f %s: RENEGED after %6.3f' % (env.now, name, wait))


# Setup and start the simulation
#print('Bank renege')
#random.seed(RANDOM_SEED)
env = simpy.Environment()


def calc_mean_arri():
    global ARRIVE
    #print(ARRIVE)
    ARRIVE = [i-ARRIVE[0] for i in ARRIVE] #Arrival appears to start later after each iteration randomly
    calc = sum(ARRIVE)/NEW_REQUESTS
    #print(ARRIVE)
    #print(calc)
    ARRIVE = []
    return calc



def calc_mean_wait():
    global WAIT
    calc = sum(WAIT)/NEW_REQUESTS
    #print(calc)
    #print()
    WAIT = []
    return calc

# Start processes and run
def create_graph():
    #cap_list = [1, 2, 4]
    cap = 1
    sample_size = 1
    request_range = [num / 1000 for num in range(1500) if num >= 1000]
    average_rho_total = []
    mean_wait_total = []

    for request_amount in tqdm(request_range):
        # average_rho = []
        mean_wait = []

        for _ in range(sample_size):
            counter = simpy.Resource(env, capacity=cap)
            env.process(source(env, NEW_REQUESTS, request_amount, counter))
            env.run()  # simulate for one

            # mean_arrival = 1/request_amount   #could be set for a calculation
            # mean_service = 1                  #could be set for a calculation

            # calc_rho = mean_arrival/(cap*mean_service)

            # average_rho.append(calc_rho)
            mean_wait.append(calc_mean_wait())

        mean_arrival = 1 / request_amount  # could be set for a calculation
        mean_service = 1 / PROCESSING_TIME  # could be set for a calculation

        calc_rho = mean_arrival / (cap * mean_service)

        average_rho_total.append(calc_rho)
        mean_wait_total.append(mean_wait)

    print(average_rho_total)
    x = average_rho_total
    y = [np.mean(i) for i in mean_wait_total]
    plt.plot(x, y, color='blue')  # plots average line

    std_dev = [np.std(j) for j in mean_wait_total]
    lower_bound = list(np.array(y) - np.array(std_dev))
    upper_bound = list(np.array(y) + np.array(std_dev))
    plt.fill_between(x, (lower_bound), (upper_bound), color='blue', alpha=0.1)  # plots standard deviation

    plt.show()

create_graph()