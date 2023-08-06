#include <vector>
#include <iostream>
#include <fmt/core.h>
#include <mpi.h>
#include <fstream>
#include <string>
#include <filesystem>

using namespace std;
#define MAX_ITEMS 256
using namespace std;

vector<vector<int>> cargarImagenPGM(const string &imagenPath) {

    ifstream archivo(std::filesystem::current_path().parent_path().string() + "/" + imagenPath, ios::binary);
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

vector<int> histograma_serial(const vector<vector<int>> &imagen) {
    vector<int> histograma(MAX_ITEMS, 0);

    for (const auto &fila: imagen) {
        for (int pixel: fila) {
            histograma[pixel]++;
        }
    }
    return histograma;
}

vector<int> histograma_omp(const vector<vector<int>> &imagen) {

    vector<int> histograma(MAX_ITEMS, 0);

#pragma omp parallel
    {
        std::vector<int> local(MAX_ITEMS, 0);
#pragma omp for
        for (const auto &fila: imagen) {
            for (int pixel: fila) {
                local[pixel]++;
            }
        }

#pragma omp critical
        {
            for (int i = 0; i < MAX_ITEMS; i++) {
                histograma[i] += local[i];
            }
        }
    }
    return histograma;
}

vector<int> cargarImagenPGMV2(const string &imagenPath) {
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
    int desde = 0;
    int intervalos = 32;
    for (int n = intervalos; n <= 256; n += intervalos) {
        int estrellas = 0;
        for (int i = desde; i < n; i++) {
            estrellas += histograma[i];
        }
        string lines = "";
        for (int j = 0; j < estrellas / 4000; j++) {
            lines += "*";
        }
        fmt::println("{} - {}: \t{}\t|{}", desde, n, estrellas, lines);
        desde = n + 1;
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
        imagen = cargarImagenPGMV2("intersecciones.pgm");
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

    // Recolectar los histogramas locales
    MPI_Reduce(&histograma_local[0], &histograma_global[0], 256,
               MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank_id == 0) {
        fmt::println("Histograma MPI");
        imprimirHistograma(histograma_global);
    }
//////////////////////////////////version serial y version omp////////////////////////
    if (rank_id == 0) {
        vector<vector<int>> imagen = cargarImagenPGM("intersecciones.pgm");

        if (imagen.empty()) {
            fmt::print("Error al abrir el archivo");
            return 1;
        }

        {
            vector<int> histograma = histograma_serial(imagen);
            fmt::println("Histograma Serial");
            imprimirHistograma(histograma);
        }

        {
            vector<int> histograma = histograma_omp(imagen);
            fmt::println("Histograma OMP");
            imprimirHistograma(histograma);
        }

    }
    MPI_Finalize();
    return 0;
}
