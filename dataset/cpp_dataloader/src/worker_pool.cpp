#include "worker_pool.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

static void set_thread_affinity(int cpu_id) { static
    // Set thread affinity
    #ifdef _WIN32
        HANDLE thread_handle = GetCurrentThread();
        DWORD_PTR thread_affinity_mask = 1LL << cpu_id;
        SetThreadAffinityMask(thread_handle, thread_affinity_mask);
    #else
        cpu_set_t cpu_set;
        CPU_ZERO(&cpu_set);
        CPU_SET(cpu_id, &cpu_set);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set), &cpu_set);
    #endif
}


//------------------------------------------------------------------------------
// ThreadWorker

ThreadWorker::ThreadWorker(int cpu_id_affinity)
{
    CpuIdAffinity = cpu_id_affinity;

    Terminated = false;
    Thread = std::make_shared<std::thread>(&ThreadWorker::Loop, this);
}

ThreadWorker::~ThreadWorker()
{
    Terminated = true;
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
            task();
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


//------------------------------------------------------------------------------
// WorkerPool

void WorkerPool::Start(int worker_count, bool use_thread_affinity)
{
    const int num_logical_cores = std::thread::hardware_concurrency();
    if (worker_count <= 0) {
        worker_count = num_logical_cores;
    }

    int affinity = 0;
    for (int i = 0; i < worker_count; ++i) {
        auto worker = std::make_shared<ThreadWorker>(use_thread_affinity ? affinity : -1);
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

WorkerPool::~WorkerPool()
{
    Workers.clear();
}

void WorkerPool::QueueTask(TaskFn task)
{
    // Find the least busy worker.
    int best_active_tasks = -1;
    int best_i = -1;

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

    Workers[best_i]->QueueTask(task);
}
