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
        std::unique_lock<std::mutex> lock(TaskLock);
        TaskCondition.notify_all();
    }
    {
        std::unique_lock<std::mutex> lock(DoneLock);
        DoneCondition.notify_all();
    }
    JoinThread(Thread);
    Thread = nullptr;
}

void ThreadWorker::Loop()
{
    if (CpuIdAffinity >= 0) {
        set_thread_affinity(CpuIdAffinity);
    }

    std::vector<TaskFn> todo;

    while (!Terminated) {
        // Wait for new work to be submitted
        {
            std::unique_lock<std::mutex> lock(TaskLock);
            TaskCondition.wait(lock, [this]{ return Terminated || Tasks.size() > 0; });
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
        todo.clear();

        // Notify all waiting threads that new work can be submitted
        {
            std::unique_lock<std::mutex> lock(DoneLock);
            DoneCondition.notify_all();
        }
    }
}

void ThreadWorker::QueueTask(TaskFn task)
{
    std::unique_lock<std::mutex> lock(TaskLock);
    ++ActiveTasks;
    Tasks.emplace_back(std::move(task));
    TaskCondition.notify_all();
}

void ThreadWorker::WaitForTasks(int max_active_tasks) {
    std::unique_lock<std::mutex> lock(DoneLock);
    DoneCondition.wait(lock, [this, max_active_tasks]{ return Terminated || ActiveTasks <= max_active_tasks; });
}


//------------------------------------------------------------------------------
// WorkerPool

void WorkerPool::Start(int worker_count, bool use_thread_affinity)
{
    std::lock_guard<std::mutex> lock(Lock);

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
    std::lock_guard<std::mutex> lock(Lock);

    Workers.clear();
}

void WorkerPool::WaitForTasks()
{
    std::lock_guard<std::mutex> lock(Lock);

    for (auto& worker : Workers) {
        worker->WaitForTasks();
    }
}

void WorkerPool::QueueTask(TaskFn task, int max_active_tasks, bool round_robin)
{
    std::lock_guard<std::mutex> lock(Lock);

    // Pick first round-robin worker
    int next_task_index = 0;
    if (round_robin) {
        next_task_index = RoundRobinIndex++;
        if (RoundRobinIndex >= (int)Workers.size()) {
            RoundRobinIndex = 0;
        }
    }

    for (;;) {
        // Find the least busy worker.
        int best_active_tasks = -1;
        int best_i = -1;

        for (int i = 0; i < (int)Workers.size(); ++i) {
            // Pick next worker index
            int task_index = next_task_index++;
            if (task_index >= (int)Workers.size()) {
                task_index = 0;
            }

            int active_tasks = Workers[task_index]->GetActiveTaskCount();

            // Any idle worker is fine.
            if (active_tasks <= 0) {
                Workers[task_index]->QueueTask(task);
                return;
            }

            if (best_active_tasks < 0 || active_tasks < best_active_tasks) {
                best_active_tasks = active_tasks;
                best_i = task_index;
            }
        }

        // If there is no limit (0) or the best worker has fewer than the limit:
        if (max_active_tasks == 0 || best_active_tasks < max_active_tasks) {
            // Queue the task on the best worker
            Workers[best_i]->QueueTask(task);
            return;
        }

        // Wait and try to find a less busy worker
        std::this_thread::sleep_for(std::chrono::milliseconds(10 + rand() % 10));
    }
}
