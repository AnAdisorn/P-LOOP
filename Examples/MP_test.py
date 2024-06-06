# %%
import numpy as np
from multiprocessing import Process, Queue
from threading import Thread
import time


# %%
class Manager(Thread):

    def __init__(self, num_works, num_workers):
        super(Manager, self).__init__()
        self.job_queue = Queue()
        self.check_queue = Queue()
        self.result_queue = []

        self.num_works = num_works
        self.works = np.arange(self.num_works) + 1
        self.count = 0

        self.worker = Worker
        self.num_workers = num_workers
        self.workers = []

        self._init_result_queue()
        self._init_works()
        self._init_workers()

    def _init_result_queue(self):
        # Create Queue to communicate with workers
        for _ in range(self.num_workers):
            self.result_queue.append(Queue(1))

    def _init_works(self):
        # Add works to the Job Queue
        for i in self.works:
            self.job_queue.put(i)

    def _init_workers(self):
        # Recruit the workers
        for i in range(self.num_workers):
            self.workers.append(
                self.worker(
                    submit_queue=self.check_queue,
                    job_queue=self.job_queue,
                    result_queue=self.result_queue[i],
                    idx=i,
                )
            )

    def run(self):
        # Start the workers
        for worker in self.workers:
            worker.start()

        jobdone = False
        while not jobdone:
            # Check result of works from Check Queue
            job_idx, idx = self.check_queue.get()
            worker = self.workers[idx]

            # Random chance for job is complete
            if np.random.rand() > 0.5:
                result = "completed"
                self.count += 1
            else:
                result = "failed"
            self.result_queue[idx].put(result)
            print(f"job({job_idx}) is {result} by {idx}, count={self.count}")

            # Check end condition
            if self.count == self.num_works:
                jobdone = True
        exit(0)


class Worker(Process):

    def __init__(self, submit_queue, job_queue, result_queue, idx):
        super(Worker, self).__init__()
        self.submit_queue = submit_queue
        self.job_queue = job_queue
        self.result_queue = result_queue
        self.idx = idx

    def run(self):
        while not self.job_queue.empty():
            # There is more job in the queue, grab it
            job_idx = self.job_queue.get()
            print(f"{self.idx} start job({job_idx})")

            # Start working
            work_state = "in_process"
            start = time.time()
            while work_state == "in_process":
                # Simulate working time
                time.sleep(np.random.rand() * 5)

                # Submit work to be checked by Manager
                self.submit_queue.put([job_idx, self.idx])

                result = self.result_queue.get()
                if result == "completed":
                    work_state = "done"

            print(f"job({job_idx}) took {time.time()-start}s by {self.idx}")

        print(f"{self.idx} found no more works, leave the office")
        exit(0)


# %%
if __name__ == "__main__":
    manager = Manager(num_works=20, num_workers=5)
    start = time.time()
    manager.start()
    manager.join()
    print(f"Time taken for this whole job: {time.time()-start}s")
