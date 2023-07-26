#include <vector>
#include <iostream>
#include <fmt/core.h>
#include <fmt/color.h>
#include <cryptopp/des.h>
#include <cryptopp/base64.h>
#include <mpi.h>
#include <fstream>
#include <string>

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int rank_id;
    int nprocs;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    
    MPI_Finalize();
}
