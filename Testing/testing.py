from threading import Thread
from pathos.multiprocessing import ProcessingPool as Pool
rez = []

def square(x, results):
    z = 0
    for y in range(x):
        z += y
    results.append(z)
    return z


def callback(t):
    rez.append(t)

def f(x):
    z = 0
    for y in range(x):
        z += y
    return z


if __name__ == '__main__':
    import time

    start = time.time()
    threads = []
    results = []
    num = 10000

    start = time.time()
    results = []
    res = [square(x, results) for x in range(num)]
    print(results)
    end = time.time()
    print(end - start)

    start = time.time()


    with Pool(5) as p:
        rez = []
        results = p.amap(f, range(num))
        results = results.get()

    print(results)
    end = time.time()
    print(end - start)

pool = Pool(cpu_count())
results = pool.amap(self.__build_forest, range(self.__n_estimators))
self.__trees = results.get()

