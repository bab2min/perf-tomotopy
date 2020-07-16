class Timer:
    watch = {}

    def __init__(self, name):
        import time
        self._name = name
    
    def __enter__(self):
        import time
        self._start = time.perf_counter()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        end = time.perf_counter()
        Timer.watch[self._name] = Timer.watch.get(self._name, 0) + (end - self._start)

    @staticmethod
    def print():
        print('==== Timer ====')
        for k, v in Timer.watch.items():
            print('{}:\t{:.4f} s'.format(k, v))
        print()