#pragma once

#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>

using TaskFn = std::function<void()>;

class ThreadPool {
public:
    ThreadPool(int worker_count = 8);
    ~ThreadPool();

    void QueueTask(TaskFn task);

private:
    std::vector<std::thread> Workers;
    std::vector<TaskFn> Tasks;

    std::mutex Lock;
    std::condition_variable Condition;
    std::atomic<bool> Terminated = ATOMIC_VAR_INIT(false);
};
