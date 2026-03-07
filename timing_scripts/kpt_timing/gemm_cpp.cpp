// #include <armadillo>
// #include <complex>
// #include <iostream>
// #include <chrono>

// // Function to apply exponential propagator of the HS transformation
// arma::cx_mat apply_exponential(arma::cx_mat& phi, const arma::cx_mat& VHS, int exp_nmax) {
//     arma::cx_mat Temp = arma::zeros<arma::cx_mat>(phi.n_rows, phi.n_cols);
//     Temp = phi;

//     for (int n = 1; n <= exp_nmax; ++n) {
//         Temp = (VHS * Temp) / static_cast<double>(n);
//         phi += Temp;
//     }

//     return phi;
// }

// // Function to perform gemm on multiple walkers
// void gemm(int nwalkers, std::vector<arma::cx_mat>& phia, std::vector<arma::cx_mat>& phib, std::vector<arma::cx_mat>& VHS, int exp_nmax) {
//     for (int iw = 0; iw < nwalkers; ++iw) {
//         phia[iw] = apply_exponential(phia[iw], VHS[iw], exp_nmax);
//         phib[iw] = apply_exponential(phib[iw], VHS[iw], exp_nmax);
//     }
// }

// int main() {
//     int nwalkers = 10;
//     int nbsf = 26;
//     int nk = 27;
//     int nup = 4, ndown = 4;
//     int exp_nmax = 6;

//     // Generate random matrices
//     std::vector<arma::cx_mat> phia(nwalkers);
//     std::vector<arma::cx_mat> phib(nwalkers);
//     std::vector<arma::cx_mat> VHS(nwalkers);

//     for (int iw = 0; iw < nwalkers; ++iw) {
//         phia[iw] = arma::randu<arma::cx_mat>(nk * nbsf, nk * nup);
//         phib[iw] = arma::randu<arma::cx_mat>(nk * nbsf, nk * ndown);
//         VHS[iw] = arma::randu<arma::cx_mat>(nk * nbsf, nk * nbsf);
//     }

//     auto start = std::chrono::high_resolution_clock::now();

//     gemm(nwalkers, phia, phib, VHS, exp_nmax);

//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsed = end - start;
//     std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

//     return 0;
// }
// #include <armadillo>
// #include <iostream>
// #include <chrono>

// // Function to apply exponential propagator of the HS transformation
// arma::mat apply_exponential(arma::mat& phi, const arma::mat& VHS, int exp_nmax) {
//     arma::mat Temp = arma::zeros<arma::mat>(phi.n_rows, phi.n_cols);
//     Temp = phi;

//     for (int n = 1; n <= exp_nmax; ++n) {
//         Temp = (VHS * Temp) / static_cast<double>(n);
//         phi += Temp;
//     }

//     return phi;
// }

// // Function to perform gemm on multiple walkers
// void gemm(int nwalkers, std::vector<arma::mat>& phia, std::vector<arma::mat>& phib, std::vector<arma::mat>& VHS, int exp_nmax) {
//     for (int iw = 0; iw < nwalkers; ++iw) {
//         phia[iw] = apply_exponential(phia[iw], VHS[iw], exp_nmax);
//         phib[iw] = apply_exponential(phib[iw], VHS[iw], exp_nmax);
//     }
// }

// int main() {
//     int nwalkers = 10;
//     int nbsf = 26;
//     int nk = 27;
//     int nup = 4, ndown = 4;
//     int exp_nmax = 6;

//     // Generate random matrices
//     std::vector<arma::mat> phia(nwalkers);
//     std::vector<arma::mat> phib(nwalkers);
//     std::vector<arma::mat> VHS(nwalkers);

//     for (int iw = 0; iw < nwalkers; ++iw) {
//         phia[iw] = arma::randu<arma::mat>(nk * nbsf, nk * nup);
//         phib[iw] = arma::randu<arma::mat>(nk * nbsf, nk * ndown);
//         VHS[iw] = arma::randu<arma::mat>(nk * nbsf, nk * nbsf);
//     }

//     auto start = std::chrono::high_resolution_clock::now();

//     gemm(nwalkers, phia, phib, VHS, exp_nmax);

//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsed = end - start;
//     std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

//     return 0;
// }

#include <armadillo>
#include <iostream>
#include <complex>
#include <chrono>

using namespace std::chrono;

// Function to apply exponential propagator (as an example)
arma::cx_mat apply_exponential(arma::cx_mat& phi, const arma::cx_mat& VHS, int exp_nmax) {
    arma::cx_mat Temp = phi;

    for (int n = 1; n <= exp_nmax; ++n) {
        Temp = VHS * Temp / static_cast<double>(n);
        phi += Temp;
    }

    return phi;
}

// Function to perform gemm on multiple walkers
void gemm(int nwalkers, std::vector<arma::cx_mat>& phia, std::vector<arma::cx_mat>& phib, std::vector<arma::cx_mat>& VHS, int exp_nmax) {
    for (int iw = 0; iw < nwalkers; ++iw) {
        phia[iw] = apply_exponential(phia[iw], VHS[iw], exp_nmax);
        phib[iw] = apply_exponential(phib[iw], VHS[iw], exp_nmax);
    }
}

int main() {
    int nwalkers = 10;
    int nbsf = 26;
    int nk = 27;
    int nup = 4, ndown = 4;
    int exp_nmax = 6;
    int n_trials = 100;

    // Create vectors to store the complex matrices for phia, phib, and VHS
    std::vector<arma::cx_mat> phia(nwalkers);
    std::vector<arma::cx_mat> phib(nwalkers);
    std::vector<arma::cx_mat> VHS(nwalkers);

    // Generate random cx_matrices
    for (int iw = 0; iw < nwalkers; ++iw) {
        phia[iw] = arma::randu<arma::cx_mat>(nk * nbsf, nk * nup);
        phib[iw] = arma::randu<arma::cx_mat>(nk * nbsf, nk * ndown);
        VHS[iw] = arma::randu<arma::cx_mat>(nk * nbsf, nk * nbsf);
    }

    // Armadillo vector to store the elapsed times for each trial
    arma::vec elapsed_times(n_trials);

    // Run the gemm function 100 times and record the elapsed time for each run
    for (int i = 0; i < n_trials; ++i) {
        auto start = high_resolution_clock::now();

        // Call the gemm function
        gemm(nwalkers, phia, phib, VHS, exp_nmax);

        auto end = high_resolution_clock::now();
        double elapsed_time = duration_cast<duration<double>>(end - start).count();
        elapsed_times(i) = elapsed_time;
    }

    // Calculate the mean and variance of the elapsed times
    double mean_time = arma::mean(elapsed_times);
    double std_dev_time = arma::stddev(elapsed_times);

    std::cout << "Mean of elapsed times: " << mean_time << " seconds" << std::endl;
    std::cout << "Std deviation of elapsed times: " << std_dev_time << " seconds" << std::endl;

    return 0;
}