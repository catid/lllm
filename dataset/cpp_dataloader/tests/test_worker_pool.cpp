#include <iostream>
#include <cassert>

#include "worker_pool.hpp"

bool TestSingleTask() {
    WorkerPool pool;
    pool.Start(4, false);

    bool task_completed = false;
    pool.QueueTask([&]() {
        task_completed = true;
    });

    pool.WaitForTasks();
    pool.Stop();

    if (!task_completed) {
        std::cout << "TestSingleTask failed" << std::endl;
        return false;
    }
    std::cout << "TestSingleTask passed" << std::endl;
    return true;
}

bool TestMultipleTasks() {
    WorkerPool pool;
    pool.Start();

    std::vector<int> results;
    std::mutex results_mutex;

    const int num_tasks = 10000000;
    for (int i = 0; i < num_tasks; ++i) {
        pool.QueueTask([&, i]() {
            std::lock_guard<std::mutex> lock(results_mutex);
            results.push_back(i);
        });
    }

    pool.WaitForTasks();
    pool.Stop();

    if (results.size() != num_tasks) {
        std::cout << "TestMultipleTasks failed" << std::endl;
        return false;
    }
    std::cout << "TestMultipleTasks passed" << std::endl;
    return true;
}

int main() {
    if (!TestSingleTask()) {
        return -1;
    }
    if (!TestMultipleTasks()) {
        return -1;
    }

    std::cout << "All tests passed" << std::endl;
    return 0;
}
