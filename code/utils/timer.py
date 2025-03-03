import time

class Timer:
    def __init__(self):
        self.start_time = None

    def tic(self):
        self.start_time = time.time()

    def toc(self):
        if self.start_time is None:
            raise ValueError("Timer not started. Call tic() to start the timer.")
        return time.time() - self.start_time
