#include <vector>
#include <iostream>
#include <fmt/core.h>
#include <mpi.h>
#include <fstream>
#include <string>
#include <iostream>
#include <filesystem>

using namespace std;

vector<int> cargarImagenPGM(const string &imagenPath) {
    ifstream archivo(std::filesystem::current_path().parent_path().string() + "/" + imagenPath, ios::binary);
    if (!archivo) {
        return {};
    }

    string tipo;
    int ancho, alto, max_valor;
    //para extraer valores
    archivo >> tipo >> ancho >> alto >> max_valor;

    vector<int> imagen(alto * ancho);

    for (int i = 0; i < alto; i++) {
        for (int j = 0; j < ancho; j++) {
            int valor;
            archivo >> valor;
            imagen[i * ancho + j] = valor;
        }
    }
    return imagen;
}

void imprimirHistograma(vector<int> histograma) {
    fmt::println("Histograma MPi");
    for (int i = 0; i < 256; ++i) {
        fmt::println("{}: {}", i, histograma[i]);
    }
}

vector<int> contar_datos(const vector<int> &data) {
    vector<int> datos(256, 0); // El histograma tiene 256 elementos
    for (int pixel: data) {
        datos[pixel]++;
    }
    return datos;
}

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int rank_id;
    int nprocs;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);//identificador de cada proceso
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);//numero total de procesos

    vector<int> imagen;
    int tamanio, bloque;

    if (rank_id == 0) {
        imagen = cargarImagenPGM("intersecciones.pgm");
        tamanio = imagen.size();
        bloque = tamanio / nprocs;

        for (int nRank = 1; nRank < nprocs; nRank++) {
            MPI_Send(&tamanio, 1, MPI_INT, nRank, 0, MPI_COMM_WORLD);
            MPI_Send(&bloque, 1, MPI_INT, nRank, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&tamanio, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&bloque, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // vector local para almacenar los datos
    vector<int> local_data(bloque);

    // Distribuir los datos
    MPI_Scatter(&imagen[0], bloque, MPI_INT, &local_data[0], bloque, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> histograma_local = contar_datos(local_data);

    vector<int> histograma_global(256, 0);

    MPI_Reduce(&histograma_local[0], &histograma_global[0], 256,
               MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);// Recolectar los histogramas locales

    if (rank_id == 0) {
        imprimirHistograma(histograma_global);
    }

    MPI_Finalize();
    return 0;
}
