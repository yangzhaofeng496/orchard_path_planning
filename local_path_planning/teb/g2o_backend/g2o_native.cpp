#include <cmath>
#include <iostream>
#include <memory>

#include <Eigen/Core>
#include <g2o/core/base_dynamic_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

class TebVectorVertex final : public g2o::BaseDynamicVertex<Eigen::VectorXd> {
 public:
  TebVectorVertex(Eigen::VectorXd lower, Eigen::VectorXd upper)
      : lower_(std::move(lower)), upper_(std::move(upper)) {
    setDimension(static_cast<int>(lower_.size()));
  }

  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }

 protected:
  bool setDimensionImpl(int dimension) override {
    _estimate = Eigen::VectorXd::Zero(dimension);
    return true;
  }
  void setToOriginImpl() override { _estimate.setZero(); }
  void oplusImpl(const double* update) override {
    Eigen::Map<const Eigen::VectorXd> delta(update, dimension());
    _estimate += delta;
    _estimate = _estimate.cwiseMax(lower_).cwiseMin(upper_);
  }

 private:
  Eigen::VectorXd lower_;
  Eigen::VectorXd upper_;
};

class PythonResidualEdge final
    : public g2o::BaseUnaryEdge<-1, Eigen::VectorXd, TebVectorVertex> {
 public:
  PythonResidualEdge(py::object residual, int residual_dimension)
      : residual_(std::move(residual)) {
    setDimension(residual_dimension);
    setMeasurement(Eigen::VectorXd::Zero(residual_dimension));
    setInformation(Eigen::MatrixXd::Identity(residual_dimension, residual_dimension));
  }

  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }

  void computeError() override {
    const auto* vertex = static_cast<const TebVectorVertex*>(_vertices[0]);
    _error = evaluate(vertex->estimate());
  }

  void linearizeOplus() override {
    auto* vertex = static_cast<TebVectorVertex*>(_vertices[0]);
    const Eigen::VectorXd original = vertex->estimate();
    const Eigen::VectorXd base = evaluate(original);
    const double epsilon = 1e-6;
    _jacobianOplusXi.resize(dimension(), original.size());
    for (Eigen::Index column = 0; column < original.size(); ++column) {
      Eigen::VectorXd perturbed = original;
      perturbed[column] += epsilon;
      vertex->setEstimate(perturbed);
      _jacobianOplusXi.col(column) = (evaluate(perturbed) - base) / epsilon;
    }
    vertex->setEstimate(original);
    _error = base;
  }

 private:
  Eigen::VectorXd evaluate(const Eigen::VectorXd& state) const {
    py::object value = residual_(state);
    return value.cast<Eigen::VectorXd>();
  }
  py::object residual_;
};

py::dict optimize(const Eigen::VectorXd& x0, const Eigen::VectorXd& lower,
                  const Eigen::VectorXd& upper, py::object residual,
                  int residual_dimension, int max_iterations) {
  if (x0.size() == 0 || x0.size() != lower.size() || x0.size() != upper.size())
    throw std::invalid_argument("invalid TEB vector/bounds dimensions");

  g2o::SparseOptimizer optimizer;
  using Block = g2o::BlockSolverX;
  auto linear = std::make_unique<g2o::LinearSolverEigen<Block::PoseMatrixType>>();
  auto block = std::make_unique<Block>(std::move(linear));
  optimizer.setAlgorithm(
      new g2o::OptimizationAlgorithmLevenberg(std::move(block)));
  optimizer.setVerbose(false);

  auto* vertex = new TebVectorVertex(lower, upper);
  vertex->setId(0);
  vertex->setEstimate(x0.cwiseMax(lower).cwiseMin(upper));
  optimizer.addVertex(vertex);

  auto* edge = new PythonResidualEdge(std::move(residual), residual_dimension);
  edge->setId(1);
  edge->setVertex(0, vertex);
  optimizer.addEdge(edge);

  optimizer.initializeOptimization();
  const int iterations = optimizer.optimize(max_iterations);
  edge->computeError();

  py::dict result;
  result["x"] = vertex->estimate();
  result["fun"] = edge->chi2();
  result["nit"] = iterations;
  result["success"] = iterations >= 0 && vertex->estimate().allFinite();
  result["status"] = iterations >= 0 ? 0 : 1;
  result["message"] = iterations >= 0 ? "g2o optimization completed"
                                        : "g2o optimization failed";
  return result;
}

PYBIND11_MODULE(_g2o_teb_native, module) {
  module.doc() = "Native g2o driver for the orchard TEB residual graph";
  module.def("optimize", &optimize, py::arg("x0"), py::arg("lower"),
             py::arg("upper"), py::arg("residual"),
             py::arg("residual_dimension"), py::arg("max_iterations"));
}
