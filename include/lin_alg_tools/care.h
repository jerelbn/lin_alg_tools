#pragma once

#include <Eigen/Dense>
#include "lin_alg_tools/schur.h"

using namespace Eigen;


template<int M, int N> // M: size of Q, N: size of R
class CareSolver
{

private:

  typedef Matrix<double, M, M> MatrixM;
  typedef Matrix<double, N, N> MatrixN;
  typedef Matrix<double, M*2, M*2> MatrixM2;

  MatrixM U11_, U21_;
  MatrixM2 H_, Z_;
  SchurSolver<M*2> schur_solver_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CareSolver() : schur_solver_(true, true) {}

  void solve(MatrixM& P, const MatrixM& A, const Matrix<double,M,N>& B, const MatrixM& Q, const MatrixN& R)
  {
    // Construct H
    H_.template block<M,M>(0,0) = A;
    H_.template block<M,M>(0,M) = -B * R.inverse() * B.transpose();
    H_.template block<M,M>(M,0) = -Q;
    H_.template block<M,M>(M,M) = -A.transpose();

    // Schur decomposition of H
    schur_solver_.solve(H_.data(), Z_.data());

    // Find P that solves CARE
    U11_ = Z_.template block<M,M>(0,0);
    U21_ = Z_.template block<M,M>(M,0);
    P = U11_.transpose().householderQr().solve(U21_.transpose());
  }

};
