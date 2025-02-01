#ifndef DISTRIBUTED_ML_MODEL_H
#define DISTRIBUTED_ML_MODEL_H

#include <vector>
#include <Eigen/Dense>

namespace distributed_ml {

class MLModel {
protected:
    int input_dim_;
    int output_dim_;
    Eigen::MatrixXd weights_;
    Eigen::VectorXd bias_;

public:
    // Constructor
    MLModel(int input_dim, int output_dim);

    // Virtual destructor for proper inheritance
    virtual ~MLModel() = default;

    // Pure virtual methods to be implemented by derived classes
    virtual void train(const Eigen::MatrixXd& input, const Eigen::VectorXd& target) = 0;
    virtual Eigen::VectorXd predict(const Eigen::VectorXd& input) const = 0;

    // Methods for parameter management
    std::vector<double> get_parameters() const;
    void set_parameters(const std::vector<double>& params);
};

} // namespace distributed_ml

#endif // DISTRIBUTED_ML_MODEL_H
