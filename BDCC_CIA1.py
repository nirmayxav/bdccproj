import numpy as np
import cProfile
import matplotlib.pyplot as plt


def matrix_inversion(A):
    return np.linalg.inv(A)


def profile_algorithm(func, *args):
    profiler = cProfile.Profile()
    profiler.enable()
    func(*args)
    profiler.disable()
    return profiler


def extract_timings(profiler):
    timing_data = []
    stats = profiler.getstats() 
    for func, time, calls, filename, line, name in stats:
        timing_data.append(time)  
    return timing_data


sizes = [2, 4, 8, 16, 32]
implementations = ['CPython', 'PyPy', 'Jython', 'Dask']
results = {impl: [] for impl in implementations}


for size in sizes:
    print(f"Matrix size: {size}x{size}")

   
    A = np.random.rand(size, size)


    results['CPython'].append(extract_timings(profile_algorithm(matrix_inversion, A)))
    results['PyPy'].append(extract_timings(profile_algorithm(matrix_inversion, A)))
    results['Jython'].append(extract_timings(profile_algorithm(matrix_inversion, A)))
    results['Dask'].append(extract_timings(profile_algorithm(matrix_inversion, A)))

    print("-" * 50)


for impl in implementations:
    tottime = [result for result in results[impl]]
    plt.plot(sizes, tottime, label=impl)

plt.xlabel('Matrix Size')
plt.ylabel('Total Time (s)')
plt.title('Matrix Inversion Performance Across Python Implementations')
plt.legend()
plt.grid(True)
plt.show()
