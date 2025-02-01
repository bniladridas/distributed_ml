#include "ml_model.h"
#include <vector>

namespace distributed_ml {

MLModel::MLModel(int input_dim, int output_dim) 
    : input_dim_(input_dim), output_dim_(output_dim),
      weights_(input_dim, output_dim), 
      bias_(output_dim) {
    // Initialize weights and bias to zero or random values
    weights_.setZero();
    bias_.setZero();
}

std::vector<double> MLModel::get_parameters() const {
    std::vector<double> params;
    // Flatten weights and bias into a single vector
    for (int i = 0; i < weights_.rows(); ++i) {
        for (int j = 0; j < weights_.cols(); ++j) {
            params.push_back(weights_.coeff(i, j));
        }
    }
    for (int i = 0; i < bias_.size(); ++i) {
        params.push_back(bias_.coeff(i));
    }
    return params;
}

void MLModel::set_parameters(const std::vector<double>& params) {
    // Reconstruct weights and bias from the parameter vector
    int weight_size = input_dim_ * output_dim_;
    
    for (int i = 0; i < input_dim_; ++i) {
        for (int j = 0; j < output_dim_; ++j) {
            weights_.coeffRef(i, j) = params[i * output_dim_ + j];
        }
    }

    for (int i = 0; i < output_dim_; ++i) {
        bias_.coeffRef(i) = params[weight_size + i];
    }
}

// Default implementations for pure virtual methods
void MLModel::train(const Eigen::MatrixXd& /*input*/, const Eigen::VectorXd& /*target*/) {
    // Default implementation does nothing, as this is an abstract base class
}

Eigen::VectorXd MLModel::predict(const Eigen::VectorXd& /*input*/) const {
    // Default implementation returns an empty vector
    return Eigen::VectorXd();
}

} // namespace distributed_ml
