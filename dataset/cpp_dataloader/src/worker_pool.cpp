#include "worker_pool.hpp"

#include "tools.hpp"

#include <cstdlib>


//------------------------------------------------------------------------------
// ThreadWorker

ThreadWorker::ThreadWorker(int worker_index, int cpu_id_affinity)
{
    WorkerIndex = worker_index;
    CpuIdAffinity = cpu_id_affinity;

    Terminated = false;
    Thread = std::make_shared<std::thread>(&ThreadWorker::Loop, this);
}

ThreadWorker::~ThreadWorker()
{
    Terminated = true;
    {
        std::unique_lock<std::mutex> lock(Lock);
        Condition.notify_one();
    }
    JoinThread(Thread);
    Thread = nullptr;
}

void ThreadWorker::Loop()
{
    if (CpuIdAffinity >= 0) {
        set_thread_affinity(CpuIdAffinity);
    }

    while (!Terminated) {
        std::vector<TaskFn> todo;

        {
            std::unique_lock<std::mutex> lock(Lock);
            Condition.wait(lock, [this]{ return Terminated || Tasks.size() > 0; });
            if (Terminated) {
                break;
            }
            todo.swap(Tasks);
        }

        for (auto& task : todo) {
            task(WorkerIndex);
            if (Terminated) {
                break;
            }
            --ActiveTasks;
        }
    }
}

void ThreadWorker::QueueTask(TaskFn task)
{
    std::unique_lock<std::mutex> lock(Lock);
    ++ActiveTasks;
    Tasks.emplace_back(std::move(task));
    Condition.notify_one();
}

void ThreadWorker::WaitForTasks(int max_active_tasks)
{
    while (!Terminated && ActiveTasks > max_active_tasks) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}


//------------------------------------------------------------------------------
// WorkerPool

void WorkerPool::Start(int worker_count, bool use_thread_affinity)
{
    const int num_logical_cores = std::thread::hardware_concurrency();
    if (worker_count <= 0) {
        worker_count = num_logical_cores;
    }

    int affinity = 0;
    for (int worker_index = 0; worker_index < worker_count; ++worker_index) {
        auto worker = std::make_shared<ThreadWorker>(
            worker_index,
            use_thread_affinity ? affinity : -1);
        Workers.emplace_back(std::move(worker));

        affinity += 2;
        if (affinity >= num_logical_cores) {
            // Avoid putting two workers on the same physical core.
            // For e.g. 8 logical cores:
            // 0, 2, 4, 8, 10->1, 3, 5, 7, 9->0, 2, 4, ...
            // Note this assumes an even number of logical cores.
            affinity--;
            affinity -= num_logical_cores;
        }
    }
}

void WorkerPool::Stop()
{
    Workers.clear();
}

void WorkerPool::WaitForTasks()
{
    for (auto& worker : Workers) {
        worker->WaitForTasks();
    }
}

void WorkerPool::QueueTask(TaskFn task, int max_active_tasks)
{
    // Find the least busy worker.
    int best_active_tasks = -1;
    int best_i = -1;

    for (;;) {
        for (int i = 0; i < Workers.size(); ++i) {
            int active_tasks = Workers[i]->GetActiveTaskCount();

            // Any idle worker is fine.
            if (active_tasks <= 0) {
                Workers[i]->QueueTask(task);
                return;
            }

            if (best_active_tasks < 0 || active_tasks < best_active_tasks) {
                best_active_tasks = active_tasks;
                best_i = i;
            }
        }

        if (max_active_tasks <= 0 || best_active_tasks < max_active_tasks) {
            break;
        }

        // Wait and try to find a less busy worker
        std::this_thread::sleep_for(std::chrono::milliseconds(10 + rand() % 10));
    }

    if (best_i != -1) {
        Workers[best_i]->QueueTask(task);
    }
}
