#pragma once
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory>


namespace clmt {
    template <typename T>
    class threadSafeQueue {
    private:
        mutable std::mutex mut;
        std::queue<T> data;
        std::condition_variable data_cond;
    public:
        threadSafeQueue() {}
        threadSafeQueue(const threadSafeQueue& other) {
            std::lock_guard<std::mutex> lk(mut);
            data = other.data;
        }
        void push(T x) {
            std::lock_guard<std::mutex> lk(mut);
            data.push(x);
            data_cond.notify_all();
        }
        void waitPop(T& value) {
            std::lock_guard<std::mutex> lk(mut);
            data_cond.wait(lk, [this] {return !data.empty(); });
            value = data.front();
            data.pop();
        }

        std::shared_ptr<T> waitPop() {
            std::lock_guard<std::mutex> lk(mut);
            data_cond.wait(lk, [this] {return !data.empty(); });
            std::shared_ptr<T> result = std::make_shared<T>(data.front());
            data.pop();
            return result;
        }
        bool tryPop(T& value) {
            std::lock_guard<std::mutex> lk(mut);
            if (data.empty()) {
                return false;
            }
            value = data.front();
            data.pop();
            return true;
        }
        std::shared_ptr<T> tryPop() {
            std::lock_guard<std::mutex> lk(mut);
            if (data.empty()) {
                return std::make_shared<T>();
            }
            std::shared_ptr<T> result = std::make_shared<T>(data.front());
            data.pop();
            return result;
        }
        bool empty()const {
            std::lock_guard<std::mutex> lk(mut);
            return data.empty();
        }
    };

}