#ifndef NETWORK_TRAINER_H
#define NETWORK_TRAINER_H

#include "neurons_network.h"
#include "training_data.h"
#include <condition_variable>
#include <future>
#include <queue>
#include <thread>
#include <variant>
#include <vector>

using Result = std::variant<bool, double>;

class NetworkTrainer
{
public:
    NetworkTrainer(size_t number_threads);
    ~NetworkTrainer() noexcept;

    void train_network(NeuronsNetwork &network, const std::vector<std::vector<TrainingData>> &datas_chunks, const std::vector<TrainingData> &test_datas,
                       const TrainingParams &training_params);

    int test_network(NeuronsNetwork &network, const std::vector<TrainingData> &datas);

    void thread_work();
private:
    static double train_network_with_data(NeuronsNetwork &network, const TrainingData &datas);
    bool is_prediction_good(const Vector<float> &expected, const Vector<float> &actual);
    static unsigned int map_network_output_to_res(const Vector<float> &output);

    std::vector<std::thread> thread_pool;
    std::mutex tasks_mutex;
    std::queue<std::packaged_task<Result()>> tasks;
    bool shutdown = false;
    std::condition_variable cv;

    template<class T>
    void add_task(T&& fct);
};

template<class T>
void NetworkTrainer::add_task(T&& fct) {
    std::unique_lock<std::mutex> lock(tasks_mutex);
    tasks.emplace(std::forward<T>(fct));
    lock.unlock();
    cv.notify_one();
}

#endif // NETWORK_TRAINER_H
