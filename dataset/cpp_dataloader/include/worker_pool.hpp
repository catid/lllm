#pragma once

#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>

using TaskFn = std::function<void()>;


//------------------------------------------------------------------------------
// ThreadWorker

class ThreadWorker {
public:
    ThreadWorker(int cpu_id_affinity = -1);
    ~ThreadWorker();

    void QueueTask(TaskFn task);

    const int32_t GetActiveTaskCount() const { return ActiveTasks; }

protected:
    int CpuIdAffinity = -1;

    std::shared_ptr<std::thread> Thread;
    std::atomic<bool> Terminated = ATOMIC_VAR_INIT(false);

    std::atomic<int32_t> ActiveTasks = ATOMIC_VAR_INIT(0);

    std::vector<TaskFn> Tasks;

    std::mutex Lock;
    std::condition_variable Condition;

    void Loop();
};


//------------------------------------------------------------------------------
// WorkerPool

class WorkerPool {
public:
    ~WorkerPool();

    void Start(int worker_count = 0, bool use_thread_affinity = true);

    int GetWorkerCount() const { return Workers.size(); }

    void QueueTask(TaskFn task);

private:
    std::vector<std::shared_ptr<ThreadWorker>> Workers;
};
