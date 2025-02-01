#include "distributed_trainer.h"
#include <algorithm>
#include <thread>
#include <chrono>

namespace distributed_ml {
    DistributedTrainer::DistributedTrainer(
        std::shared_ptr<MLModel> model, 
        const std::vector<std::string>& worker_addresses)
        : model_(std::move(model)), 
          worker_addresses_(worker_addresses),
          logger_(spdlog::default_logger()) {
        
        if (!model_) {
            throw std::invalid_argument("MLModel cannot be null");
        }

        if (!validate_worker_addresses()) {
            throw std::invalid_argument("Invalid worker addresses");
        }
    }

    bool DistributedTrainer::validate_worker_addresses() const {
        return std::all_of(worker_addresses_.begin(), worker_addresses_.end(), 
            [](const std::string& addr) {
                // Basic validation: non-empty and contains ':' (host:port)
                return !addr.empty() && addr.find(':') != std::string::npos;
            });
    }

    std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> 
    DistributedTrainer::split_data(
        const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        
        const size_t num_workers = worker_addresses_.size();
        const size_t rows = X.rows();
        
        std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> splits;
        
        for (size_t i = 0; i < num_workers; ++i) {
            size_t start = i * (rows / num_workers);
            size_t end = (i == num_workers - 1) ? rows : (i + 1) * (rows / num_workers);
            
            splits.emplace_back(
                X.block(start, 0, end - start, X.cols()),
                y.segment(start, end - start)
            );
        }
        
        return splits;
    }

    bool DistributedTrainer::distributed_train(
        const Eigen::MatrixXd& X, 
        const Eigen::VectorXd& y,
        size_t max_retries) {
        
        if (X.rows() != y.size()) {
            logger_->error("Input data dimensions mismatch");
            return false;
        }

        current_status_ = TrainingStatus::IN_PROGRESS;
        
        try {
            auto data_splits = split_data(X, y);
            
            for (size_t retry = 0; retry < max_retries; ++retry) {
                try {
                    // Simulate distributed training
                    for (const auto& [split_X, split_y] : data_splits) {
                        model_->train(split_X, split_y);
                    }
                    
                    if (aggregate_parameters()) {
                        current_status_ = TrainingStatus::COMPLETED;
                        logger_->info("Distributed training successful");
                        return true;
                    }
                }
                catch (const std::exception& e) {
                    logger_->warn("Training attempt {} failed: {}", retry + 1, e.what());
                    std::this_thread::sleep_for(std::chrono::seconds(1 << retry));
                }
            }
            
            current_status_ = TrainingStatus::FAILED;
            logger_->error("Distributed training failed after {} retries", max_retries);
            return false;
        }
        catch (const std::exception& e) {
            current_status_ = TrainingStatus::FAILED;
            logger_->error("Unexpected error in distributed training: {}", e.what());
            return false;
        }
    }

    bool DistributedTrainer::aggregate_parameters() {
        try {
            // Placeholder for actual parameter aggregation
            // In a real implementation, this would involve gRPC calls
            // to synchronize model parameters across workers
            
            logger_->info("Aggregating model parameters");
            
            // Example: Simple averaging of model parameters
            // This is a very simplified mock implementation
            return true;
        }
        catch (const std::exception& e) {
            logger_->error("Parameter aggregation failed: {}", e.what());
            return false;
        }
    }

    std::string DistributedTrainer::get_status() const {
        switch (current_status_) {
            case TrainingStatus::NOT_STARTED: return "Not Started";
            case TrainingStatus::IN_PROGRESS: return "In Progress";
            case TrainingStatus::COMPLETED: return "Completed";
            case TrainingStatus::FAILED: return "Failed";
            default: return "Unknown";
        }
    }
} // namespace distributed_ml
