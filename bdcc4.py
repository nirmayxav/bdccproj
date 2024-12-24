import random
import time
import matplotlib.pyplot as plt
import cProfile
from multiprocessing import Pool
from threading import Thread
from dask import delayed, compute


def merge_sort(data):
    if len(data) <= 1:
        return data
    mid = len(data) // 2
    left = merge_sort(data[:mid])
    right = merge_sort(data[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    while left and right:
        if left[0] <= right[0]:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))
    result.extend(left or right)
    return result


def parallel_merge_sort_multiprocessing(data):
    if len(data) <= 1:
        return data
    mid = len(data) // 2
    with Pool(2) as pool:
        left, right = pool.map(merge_sort, [data[:mid], data[mid:]])
    return merge(left, right)

def parallel_merge_sort_threading(data):
    if len(data) <= 1:
        return data
    mid = len(data) // 2
    left, right = [], []
    left_thread = Thread(target=lambda: left.extend(merge_sort(data[:mid])))
    right_thread = Thread(target=lambda: right.extend(merge_sort(data[mid:])))
    left_thread.start()
    right_thread.start()
    left_thread.join()
    right_thread.join()
    return merge(left, right)

def parallel_merge_sort_dask(data):
    if len(data) <= 1:
        return data
    mid = len(data) // 2
    left = delayed(merge_sort)(data[:mid])
    right = delayed(merge_sort)(data[mid:])
    return compute(delayed(merge)(left, right))[0]

def measure_runtime(runtime_name, parallel_function, data):
    print(f"Running for {runtime_name}")

   
    start = time.time()
    merge_sort(data)
    sequential_time = time.time() - start

   
    start = time.time()
    parallel_function(data)
    parallel_time = time.time() - start

    # Profiling
    print(f"Profiling Sequential Execution for {runtime_name}")
    cProfile.runctx("merge_sort(data)", globals(), locals())
    print(f"Profiling Parallel Execution for {runtime_name}")
    cProfile.runctx("parallel_function(data)", globals(), locals())

    return sequential_time, parallel_time


def visualize_results(results):
    flavors = list(results.keys())
    seq_times = [results[flavor]["sequential"] for flavor in flavors]
    par_times = [results[flavor]["parallel"] for flavor in flavors]

    plt.figure(figsize=(10, 6))
    plt.bar(flavors, seq_times, color='blue', alpha=0.7, label='Sequential Execution')
    plt.bar(flavors, par_times, color='orange', alpha=0.7, label='Parallel Execution', bottom=seq_times)
    plt.ylabel("Execution Time (seconds)")
    plt.xlabel("Runtime Flavor")
    plt.title("Execution Time Comparison Across Runtimes")
    plt.legend()
    plt.show()


if __name__ == "__main__":
  
    data = [random.randint(0, 100000) for _ in range(10**6)]

    
    results = {}

    
    results["CPython"] = measure_runtime("CPython", parallel_merge_sort_multiprocessing, data)
    results["PyPy"] = measure_runtime("PyPy", parallel_merge_sort_multiprocessing, data)
    results["Jython"] = measure_runtime("Jython", parallel_merge_sort_threading, data)
    results["Dask"] = measure_runtime("Dask", parallel_merge_sort_dask, data)

 
    visualize_results({
        runtime: {"sequential": seq, "parallel": par} for runtime, (seq, par) in results.items()
    })
