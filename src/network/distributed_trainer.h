#pragma once

#include <grpcpp/grpcpp.h>
#include <vector>
#include <memory>
#include <stdexcept>
#include <spdlog/spdlog.h>
#include "../core/ml_model.h"

namespace distributed_ml {
class DistributedTrainer {
public:
    // Enhanced constructor with validation
    explicit DistributedTrainer(
        std::shared_ptr<MLModel> model, 
        const std::vector<std::string>& worker_addresses);
    
    // Distribute training across workers with error handling
    [[nodiscard]] bool distributed_train(
        const Eigen::MatrixXd& X, 
        const Eigen::VectorXd& y,
        size_t max_retries = 3);
    
    // Aggregate model parameters from workers with robust error handling
    [[nodiscard]] bool aggregate_parameters();

    // Get current training status
    [[nodiscard]] std::string get_status() const;

private:
    std::shared_ptr<MLModel> model_;
    std::vector<std::string> worker_addresses_;
    std::shared_ptr<spdlog::logger> logger_;
    
    // Internal error tracking
    enum class TrainingStatus {
        NOT_STARTED,
        IN_PROGRESS,
        COMPLETED,
        FAILED
    };
    TrainingStatus current_status_ = TrainingStatus::NOT_STARTED;

    // Validate worker addresses
    [[nodiscard]] bool validate_worker_addresses() const;

    // Split data for distributed training
    [[nodiscard]] std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> 
    split_data(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
};
