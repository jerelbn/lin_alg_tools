#include <iostream>
#include <eigen3/Eigen/Eigen>
#include <gtest/gtest.h>
#include "lin_alg_tools/schur.h"
#include "lin_alg_tools/care.h"

using namespace std;
using namespace Eigen;

typedef Eigen::Matrix<double, 6, 6> Matrix6d;



TEST(LinAlgToolsTest, SchurSolver)
{
  // Create a matrix
  Matrix3d A;
  A << -1.0, -8.0,  0.0,
       -1.0,  1.0, -5.0,
        3.0,  0.0,  2.0;

  SchurSolver<3> solver(true, true);
  solver.solve(A.data());
  Matrix3d Q(solver.getQ());
  int* sdim_ptr(solver.getSdim());

  // True answers
  Matrix3d U_true;
  U_true << -2.07741801, -1.78375524,  1.35592292,
             6.87127273, -2.07741801,  2.50039274,
                     0.,          0.,  6.15483601;

  Matrix3d Q_true;
  Q_true << -0.19098244,  0.90472430,  0.38078819,
            -0.80279692,  0.07926254, -0.59096071,
            -0.56483875, -0.41855871,  0.71117212;
  int sdim_true = 2;

  // GTEST check that error is less than some threshold
  EXPECT_LE((A - U_true).norm(), 1e-3);
  EXPECT_LE((Q - Q_true).norm(), 1e-3);
  EXPECT_LE(sdim_true - *sdim_ptr, 1e-3);
}

TEST(LinAlgToolsTest, CareSolver)
{
  // Build up some matrices
  Matrix6d A = Matrix6d::Zero();
  A.block<3,3>(0,3) = Matrix3d::Identity();

  Matrix<double,6,4> B;
  B <<         0.,          0.,          0.,  0.,
               0.,          0.,          0.,  0.,
               0.,          0.,          0.,  0.,
       3.94997654,  0.12801372, -3.00607926,  0.,
       6.16461742, -2.91511635,  0.07454642,  0.,
       8.19551358,  0.95984792,  0.62731905,  0.;

  Matrix6d Q = 0.001 * Matrix6d::Identity();
  Matrix4d R = 0.01 * Matrix4d::Identity();

  // Solve the CARE
  Matrix6d P;
  CareSolver<6,4> care_solver;
  care_solver.solve(P, A, B, Q, R);

  // Check if CARE is near zero
  double are_val = (A.transpose() * P + P * A - P * B * R.inverse() * B.transpose() * P + Q)(0);
  EXPECT_NEAR(are_val, 0.0, 1e-3);
}
