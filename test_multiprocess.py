from multiprocessing import Pool, TimeoutError
import time
import os

def f(x):
    time.sleep(3)
    return x*x

if __name__ == '__main__':
    # start 4 worker processes
    with Pool(processes=14) as pool:
        startt=time.time()
        # print "[0, 1, 4,..., 81]"
        print(pool.map(f, range(10)))
        print("time1:",time.time()-startt)
        startt=time.time()
        # print same numbers in arbitrary order
        for i in pool.imap_unordered(f, range(10)):
            print(i)
            # print(pool.map(f, range(10)))
        print("time2:", time.time() - startt)