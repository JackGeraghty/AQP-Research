import multiprocessing as mp
import os

def f(d):
    print(f"Hello {os.getpid()}")

def main():
    pool = mp.Pool(mp.cpu_count()-1)
    d = {}
    for i in range(5):
        pool.apply_async(f, args=(d,))
    pool.close()
    pool.join()
    
if __name__ == "__main__":
     main()