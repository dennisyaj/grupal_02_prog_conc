#include <vector>
#include <iostream>
#include <fmt/core.h>
#include <fmt/color.h>
#include <cryptopp/des.h>
#include <cryptopp/base64.h>
#include <mpi.h>
#include <fstream>
#include <string>
#include <iostream>
#include <filesystem>

#define MAX_ITEMS 256
using namespace std;

vector<vector<int>> cargarImagenPGM(const string &imagenPath) {

    ifstream archivo(std::filesystem::current_path().parent_path().string() + "\\" + imagenPath, ios::binary);
    if (!archivo) {
        return {};
    }

    string tipo;
    int ancho, alto, max_valor;
    //para extraer valores
    archivo >> tipo >> ancho >> alto >> max_valor;

    vector<vector<int>> imagen(alto, vector<int>(ancho));

    for (int i = 0; i < alto; i++) {
        for (int j = 0; j < ancho; j++) {
            int valor;
            archivo >> valor;
            imagen[i][j] = valor;
        }
    }
    return imagen;
}

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int rank_id;
    int nprocs;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (rank_id == 0) {
        fmt::println("{}", rank_id);
        vector<vector<int>> imagen = cargarImagenPGM("intersecciones.pgm");
    }


    MPI_Finalize();
}
