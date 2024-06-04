#pragma once

#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>


//------------------------------------------------------------------------------
// ThreadWorker

using TaskFn = std::function<void(int worker_index)>;

class ThreadWorker {
public:
    ThreadWorker(int worker_index, int cpu_id_affinity = -1);
    ~ThreadWorker();

    void QueueTask(TaskFn task);

    int32_t GetActiveTaskCount() const { return ActiveTasks; }

    void WaitForTasks(int max_active_tasks = 0);

protected:
    int WorkerIndex = -1;
    int CpuIdAffinity = -1;

    std::shared_ptr<std::thread> Thread;
    std::atomic<bool> Terminated = ATOMIC_VAR_INIT(false);

    std::atomic<int32_t> ActiveTasks = ATOMIC_VAR_INIT(0);

    std::vector<TaskFn> Tasks;

    std::mutex TaskLock, DoneLock;
    std::condition_variable TaskCondition, DoneCondition;

    void Loop();
};


//------------------------------------------------------------------------------
// WorkerPool

class WorkerPool {
public:
    ~WorkerPool() {
        Stop();
    }

    void Start(int worker_count = 0, bool use_thread_affinity = true);
    void Stop();

    void WaitForTasks();

    int GetWorkerCount() const { return Workers.size(); }

    // max_active_tasks: 0 means no limit. Otherwise, block until queue is short
    void QueueTask(TaskFn task, int max_active_tasks = 0, bool round_robin = false);

private:
    std::mutex Lock;
    std::vector<std::shared_ptr<ThreadWorker>> Workers;
    int RoundRobinIndex = 0;
};
