#include "worker_pool.hpp"

#include "tools.hpp"

bool TestSingleTask() {
    WorkerPool pool;
    pool.Start(4, false);

    bool task_completed = false;
    pool.QueueTask([&](int /*worker_index*/) {
        task_completed = true;
    });

    pool.WaitForTasks();
    pool.Stop();

    if (!task_completed) {
        LOG_ERROR() << "TestSingleTask failed";
        return false;
    }
    LOG_INFO() << "TestSingleTask passed";
    return true;
}

bool TestMultipleTasks() {
    WorkerPool pool;
    pool.Start();

    std::vector<int> results;
    std::mutex results_mutex;

    const int num_tasks = 100000 * pool.GetWorkerCount();
    for (int i = 0; i < num_tasks; ++i) {
        pool.QueueTask([&, i](int /*worker_index*/) {
            std::lock_guard<std::mutex> lock(results_mutex);
            results.push_back(i);
        });
    }

    pool.WaitForTasks();
    pool.Stop();

    if ((int)results.size() != num_tasks) {
        LOG_ERROR() << "TestMultipleTasks failed";
        return false;
    }
    LOG_INFO() << "TestMultipleTasks passed";
    return true;
}

int main() {
    if (!TestSingleTask()) {
        return -1;
    }
    if (!TestMultipleTasks()) {
        return -1;
    }

    LOG_INFO() << "All tests passed";
    return 0;
}
