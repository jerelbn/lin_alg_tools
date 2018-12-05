// For algorithm details, refer to https://en.wikipedia.org/wiki/Schur_decomposition and 
//                                 http://www.netlib.org/lapack/explore-3.1.1-html/dgees.f.html
#pragma once

#include <iostream>


// Select type and function needed for eigenvalue sorting in DGEES
typedef int select_t(double*, double*);
inline int select_(double *ar, double *ai) { return *ar < 0.0; }

// dgees_ is a symbol in the LAPACK library files
extern "C"
{
  extern int dgees_(char*, char*, select_t, int*, double*, int*, int*, double*, double*, double*, int*, double*, int*, int*, int*);
}


// Main class for the Schur decomposition solver
template<int N> // number of rows or columns of the square matrix to be decomposed
class SchurSolver
{

public:

  SchurSolver() : N_(N)
  {
    // No sorting or schur vector calculation
    jobvs_ = 'N';
    sort_ = 'N';
  }

  SchurSolver(const bool& compute_schur_vectors, const bool& sort_eigenvalues) : N_(N)
  {
    // Save flag for Schur vector computation
    if (compute_schur_vectors)
      jobvs_ = 'V';
    else
      jobvs_ = 'N';

    // Save flag for eigenvalue sorting
    if (sort_eigenvalues)
      sort_ = 'S';
    else
      sort_ = 'N';
  }


  void solve(double* U)
  {
    // calculate the Schur Decomposition using the DGEES subroutine
    dgees_(&jobvs_, &sort_, select_, &N_, U, &N_, &sdim_, &wr_[0], &wi_[0], &Q_[0][0], &N_, &work_[0], &lwork_, &bwork_[0], &info_);
    if (info_ != 0)
    {
      std::cout << "Error in SchurSolver: DGEES returned error code " << info_ << std::endl;
      std::abort();
    }
  }


  void solve(double* U, double* Q)
  {
    // calculate the Schur Decomposition using the DGEES subroutine
    dgees_(&jobvs_, &sort_, select_, &N_, U, &N_, &sdim_, &wr_[0], &wi_[0], Q, &N_, &work_[0], &lwork_, &bwork_[0], &info_);
    if (info_ != 0)
    {
      std::cout << "Error in SchurSolver: DGEES returned error code " << info_ << std::endl;
      std::abort();
    }
  }


  double* getQ() { return &Q_[0][0]; }
  int* getSdim() { return &sdim_; }

private:

  char jobvs_; // compute Schur vectors? 'N' no, 'V' yes
  char sort_; // sort eigenvalues? 'N' no, 'S' yes
  int sdim_; // number eigenvalues after sorting for which select_ is true (negative eigenvalues block size)
  int N_; // order of input matrix
  double wr_[N]; // real eigenvalues
  double wi_[N]; // imaginary eigenvalues
  double Q_[N][N]; // memory for orthogonal matrix of schur vectors
  int lwork_ = 4 * N; // workspace size
  double work_[4 * N]; // workspace
  int bwork_[4 * N]; // logical sort workspace
  int info_; // diagnostics

};
