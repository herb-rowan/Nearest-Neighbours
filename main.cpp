#include <mpi.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>
#include <time.h>
#include <stdlib.h>
#include <typeinfo>



// function to read in a list of 3D coordinates from an .xyz file
// input: the name of the file
std::vector < std::vector < double > > read_xyz_file(std::string filename, int& N, double& L){


  // open the file
  std::ifstream xyz_file(filename);

  // read in the number of atoms
  xyz_file >> N;
  
  // read in the cell dimension
  xyz_file >> L;
  
  // now read in the positions, ignoring the atomic species
  std::vector < std::vector < double > > positions;
  std::vector < double> pos = {0, 0, 0};
  std::string dummy; 
  for (int i=0;i<N;i++){
    xyz_file >> dummy >> pos[0] >> pos[1] >> pos[2];
    positions.push_back(pos);           
  }
  
  // close the file
  xyz_file.close();
  
  return positions;
  
}

int bruteForce(int argc, char **argv){
    MPI_Init(&argc, &argv);
    std::string filename = argv[1];
    int vectorPrinter = atoi(argv[3]);
    int timePrinter = atoi(argv[4]);
    // std::cout << "time printer is" << timePrinter<< std::endl;
    double cutoff = 9.0;
    double cutoff_squared = cutoff * cutoff;
    double L;
    int N;
    
    std:: vector < std::vector < double > > positions = read_xyz_file(filename, N, L);
    std::vector < double > neighbour_vector = std::vector < double > (N, 0.0);
    double start_time = MPI_Wtime();
    for (int i = 0; i<N; i++){
        for (int j = i+1; j<N; j++){
            if (i == j) continue;
            double distance_squared = 0;
            std:: vector < double > position_i = positions[i];
            std:: vector < double > position_j = positions[j];
            distance_squared = ((position_i[0] - position_j[0]) * (position_i[0] - position_j[0])) + ((position_i[1] - position_j[1]) * (position_i[1] - position_j[1])) + ((position_i[2] - position_j[2]) * (position_i[2] - position_j[2]));
            if (distance_squared < cutoff_squared){
                neighbour_vector[i]++;
                neighbour_vector[j]++;
            }
        }

    }
    double end_time = MPI_Wtime();
    MPI_Finalize();
    
    if (vectorPrinter == 1){
        for (int i = 0; i<N; i++){
            std::cout << neighbour_vector[i] <<",";
        }
    }
    if (timePrinter == 1){
        std::cout << "Time taken: " << end_time - start_time << std::endl;
    }
    return 0;


}

int bruteForceMPIChunk(int argc, char **argv){
    MPI_Init(&argc, &argv);
    std::string filename = argv[1];
    int vectorPrinter = atoi(argv[3]);
    int timePrinter = atoi(argv[4]);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double cutoff = 9.0;
    double cutoff_squared = cutoff * cutoff;
    double L;
    int N;
    int chunkSize = N / size; // Calculate the base chunk size
    int remainder = N % size; // Calculate the remainder
    int localChunkSize = chunkSize + (rank < remainder ? 1 : 0);

    
    std:: vector < std::vector < double > > positions = read_xyz_file(filename, N, L);
    std::vector < double > neighbour_vector = std::vector < double > (N, 0.0);
    double start_time = MPI_Wtime();
    for (int i = rank*localChunkSize; i<(rank+1)*localChunkSize; i+=1){
        for (int j = i+1; j<N; j++){
            if (i == j) continue;
            double distance_squared = 0;
            std:: vector < double > position_i = positions[i];
            std:: vector < double > position_j = positions[j];
            distance_squared = ((position_i[0] - position_j[0]) * (position_i[0] - position_j[0])) + ((position_i[1] - position_j[1]) * (position_i[1] - position_j[1])) + ((position_i[2] - position_j[2]) * (position_i[2] - position_j[2]));
            if (distance_squared < cutoff_squared){
                neighbour_vector[i]++;
                neighbour_vector[j]++;
            }
            
        }


    }

    std::vector<double> global_neighbour_vector(N, 0);
    MPI_Allreduce(neighbour_vector.data(), global_neighbour_vector.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();
    MPI_Finalize();
    if (rank == 0){
        if (vectorPrinter == 1){
            for (int i = 0; i<N; i++){
                std::cout << global_neighbour_vector[i] <<",";
            }
        }
        if (timePrinter == 1){
            std::cout << "Time taken: " << end_time - start_time << std::endl;
        }
    }
    return 0;

}

int bruteForceMPIDealer(int argc, char **argv){
    MPI_Init(&argc, &argv);
    std::string filename = argv[1];
    int vectorPrinter = atoi(argv[3]);
    int timePrinter = atoi(argv[4]);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double cutoff = 9.0;
    double cutoff_squared = cutoff * cutoff;
    double L;
    int N;
    
    std:: vector < std::vector < double > > positions = read_xyz_file(filename, N, L);
    std::vector < double > neighbour_vector = std::vector < double > (N, 0.0);
    double start_time = MPI_Wtime();
    for (int i = rank; i<N; i+=size){
        for (int j = i+1; j<N; j++){
            if (i == j) continue;
            double distance_squared = 0;
            std:: vector < double > position_i = positions[i];
            std:: vector < double > position_j = positions[j];
            distance_squared = ((position_i[0] - position_j[0]) * (position_i[0] - position_j[0])) + ((position_i[1] - position_j[1]) * (position_i[1] - position_j[1])) + ((position_i[2] - position_j[2]) * (position_i[2] - position_j[2]));
            if (distance_squared < cutoff_squared){
                neighbour_vector[i]++;
                neighbour_vector[j]++;
            }
            
        }


    }

    std::vector<double> global_neighbour_vector(N, 0);
    MPI_Allreduce(neighbour_vector.data(), global_neighbour_vector.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();
    MPI_Finalize();
    if (rank == 0){
        if (vectorPrinter == 1){
            for (int i = 0; i<N; i++){
                std::cout << global_neighbour_vector[i] << ",";
            }
        }
        if (timePrinter == 1){
            std::cout << "Time taken: " << end_time - start_time <<","<< std::endl;
        }
    }
    return 0;

}
int cellList(int argc, char**argv){
    MPI_Init(&argc, &argv);
    std::string filename = argv[1];
    int vectorPrinter = atoi(argv[3]);
    int timePrinter = atoi(argv[4]);
    double cutoff = 9.0;
    double cutoff_squared = pow(cutoff,2);
    double L;
    int N;
    
    std:: vector < std::vector < double > > positions = read_xyz_file(filename, N, L);
    std::vector < double > neighbour_vector = std::vector < double > (N, 0.0);
    int nCells = std::ceil(L/cutoff);
    double nCellsCubed = pow(nCells,3);
    std::vector < std::vector < bool > > BoolCheckedCell = std::vector < std::vector < bool > > (nCellsCubed, std::vector < bool > (nCellsCubed, false));
    std::vector < std::vector < int > > cellVector = std::vector < std::vector < int > > (nCellsCubed, std::vector < int > ());
    
    double start_time = MPI_Wtime();
    for (int i = 0; i<N; i++){
        int x = std::floor(positions[i][0]/(cutoff+0.001));
        int y = std::floor(positions[i][1]/(cutoff+0.001));
        int z = std::floor(positions[i][2]/(cutoff+0.001));
        int cellIndex = x + y*nCells + z*pow(nCells,2);
        if ((cellIndex >= 0) and (cellIndex < cellVector.size())){
            cellVector[cellIndex].push_back(i);
        }
        
    }
        
        for(int x_i = 0; x_i < nCells; x_i++){
            for(int y_i = 0; y_i < nCells; y_i++){
                for(int z_i = 0; z_i < nCells; z_i++){
                    for (int x_j = x_i -1; x_j< x_i+2; x_j++){
                        for(int y_j = y_i -1; y_j< y_i+2; y_j++){
                            for(int z_j = z_i -1; z_j< z_i+2; z_j++){
                                if (x_j < 0 or x_j >= nCells or y_j < 0 or y_j >= nCells or z_j < 0 or z_j >= nCells){
                                    continue;
                                }
                                // if (BoolCheckedCell[x_i + y_i*nCells + z_i*pow(nCells,2)][x_j + y_j*nCells + z_j*pow(nCells,2)] == true){
                                //     continue;
                                // }
                                for (int i = 0; i<cellVector[x_i + y_i*nCells + z_i*pow(nCells,2)].size(); i++){
                                    for (int j = 0; j<cellVector[x_j + y_j*nCells + z_j*pow(nCells,2)].size(); j++){
                                        if ((x_i + y_i*nCells + z_i*pow(nCells,2))==(x_j + y_j*nCells + z_j*pow(nCells,2)) && i == j) {
                                            continue;
                                        }
                                        int atom1 = cellVector[x_i + y_i*nCells + z_i*pow(nCells,2)][i];
                                        int atom2 = cellVector[x_j + y_j*nCells + z_j*pow(nCells,2)][j];
                                        double distance_squared = 0;
                                        std:: vector < double > position_i = positions[atom1];
                                        std:: vector < double > position_j = positions[atom2];
                                        distance_squared = ((position_i[0] - position_j[0]) * (position_i[0] - position_j[0])) + ((position_i[1] - position_j[1]) * (position_i[1] - position_j[1])) + ((position_i[2] - position_j[2]) * (position_i[2] - position_j[2]));
                                        if (distance_squared < cutoff_squared){
                                            neighbour_vector[atom1]++;
                                            // neighbour_vector[atom2]++;
                                        }
                                    }
                                }
                                BoolCheckedCell[x_i + y_i*nCells + z_i*pow(nCells,2)][x_j + y_j*nCells + z_j*pow(nCells,2)] = true;
                                BoolCheckedCell[x_j + y_j*nCells + z_j*pow(nCells,2)][x_i + y_i*nCells + z_i*pow(nCells,2)] = true;
                            }
                        }
                    }
                }
            }
        }
    
    double end_time = MPI_Wtime();
    MPI_Finalize();
    
    if (vectorPrinter == 1){
        for (int i = 0; i<N; i++){
            std::cout << neighbour_vector[i] << ",";
        }
    }
    if (timePrinter == 1){
        std::cout << "Time taken: " << end_time - start_time << std::endl;
    }
    return 0;

}
int cellListMPI(int argc, char**argv){
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::string filename = argv[1];
    int vectorPrinter = atoi(argv[3]);
    int timePrinter = atoi(argv[4]);
    double cutoff = 9.0;
    double cutoff_squared = pow(cutoff,2);
    double L;
    int N;
    
    std:: vector < std::vector < double > > positions = read_xyz_file(filename, N, L);
    std::vector < double > neighbour_vector = std::vector < double > (N, 0.0);
    int nCells = std::ceil(L/cutoff);
    double nCellsCubed = pow(nCells,3);
    std::vector < std::vector < bool > > BoolCheckedCell = std::vector < std::vector < bool > > (nCellsCubed, std::vector < bool > (nCellsCubed, false));
    std::vector < std::vector < int > > cellVector = std::vector < std::vector < int > > (nCellsCubed, std::vector < int > ());
    
    double start_time = MPI_Wtime();
    for (int i = 0; i<N; i++){
        int x = std::floor(positions[i][0]/(cutoff+0.001));
        int y = std::floor(positions[i][1]/(cutoff+0.001));
        int z = std::floor(positions[i][2]/(cutoff+0.001));
        int cellIndex = x + y*nCells + z*pow(nCells,2);
        if ((cellIndex >= 0) and (cellIndex < cellVector.size())){
            cellVector[cellIndex].push_back(i);
        }
        
    }
        
        for(int x_i = 0; x_i < nCells; x_i++){
            for(int y_i = 0; y_i < nCells; y_i++){
                for(int z_i = 0; z_i < nCells; z_i++){
                    for (int x_j = x_i -1; x_j< x_i+2; x_j++){
                        for(int y_j = y_i -1; y_j< y_i+2; y_j++){
                            for(int z_j = z_i -1; z_j< z_i+2; z_j++){
                                if (x_j < 0 or x_j >= nCells or y_j < 0 or y_j >= nCells or z_j < 0 or z_j >= nCells){
                                    continue;
                                }
                                // if (BoolCheckedCell[x_i + y_i*nCells + z_i*pow(nCells,2)][x_j + y_j*nCells + z_j*pow(nCells,2)] == true){
                                //     continue;
                                // }
                                for (int i = rank; i<cellVector[x_i + y_i*nCells + z_i*pow(nCells,2)].size(); i+=size){
                                    for (int j = 0; j<cellVector[x_j + y_j*nCells + z_j*pow(nCells,2)].size(); j++){
                                        if ((x_i + y_i*nCells + z_i*pow(nCells,2))==(x_j + y_j*nCells + z_j*pow(nCells,2)) && i == j) {
                                            continue;
                                        }
                                        int atom1 = cellVector[x_i + y_i*nCells + z_i*pow(nCells,2)][i];
                                        int atom2 = cellVector[x_j + y_j*nCells + z_j*pow(nCells,2)][j];
                                        double distance_squared = 0;
                                        std:: vector < double > position_i = positions[atom1];
                                        std:: vector < double > position_j = positions[atom2];
                                        distance_squared = ((position_i[0] - position_j[0]) * (position_i[0] - position_j[0])) + ((position_i[1] - position_j[1]) * (position_i[1] - position_j[1])) + ((position_i[2] - position_j[2]) * (position_i[2] - position_j[2]));
                                        if (distance_squared < cutoff_squared){
                                            neighbour_vector[atom1]++;
                                            // neighbour_vector[atom2]++;
                                        }
                                    }
                                }
                                BoolCheckedCell[x_i + y_i*nCells + z_i*pow(nCells,2)][x_j + y_j*nCells + z_j*pow(nCells,2)] = true;
                                BoolCheckedCell[x_j + y_j*nCells + z_j*pow(nCells,2)][x_i + y_i*nCells + z_i*pow(nCells,2)] = true;
                            }
                        }
                    }
                }
            }
        }
    
    double end_time = MPI_Wtime();

    std::vector<double> global_neighbour_vector(N, 0);
    MPI_Allreduce(neighbour_vector.data(), global_neighbour_vector.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Finalize();
    if (rank == 0){
        if (vectorPrinter == 1){
            for (int i = 0; i<N; i++){
                std::cout << global_neighbour_vector[i] <<",";
            }
        }
        if (timePrinter == 1){
            std::cout << "Time taken: " << end_time - start_time << std::endl;
        }
    }
    
    return 0;

}
int cellListMPIDoubleIncrement(int argc, char**argv){
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::string filename = argv[1];
    int vectorPrinter = atoi(argv[3]);
    int timePrinter = atoi(argv[4]);
    double cutoff = 9.0;
    double cutoff_squared = pow(cutoff,2);
    double L;
    int N;
    
    std:: vector < std::vector < double > > positions = read_xyz_file(filename, N, L);
    std::vector < double > neighbour_vector = std::vector < double > (N, 0.0);
    int nCells = std::ceil(L/cutoff);
    double nCellsCubed = pow(nCells,3);
    std::vector < std::vector < bool > > BoolCheckedCell = std::vector < std::vector < bool > > (nCellsCubed, std::vector < bool > (nCellsCubed, false));
    std::vector < std::vector < int > > cellVector = std::vector < std::vector < int > > (nCellsCubed, std::vector < int > ());
    
    double start_time = MPI_Wtime();
    for (int i = 0; i<N; i++){
        int x = std::floor(positions[i][0]/(cutoff+0.001));
        int y = std::floor(positions[i][1]/(cutoff+0.001));
        int z = std::floor(positions[i][2]/(cutoff+0.001));
        int cellIndex = x + y*nCells + z*pow(nCells,2);
        if ((cellIndex >= 0) and (cellIndex < cellVector.size())){
            cellVector[cellIndex].push_back(i);
        }
        
    }
        
        for(int x_i = 0; x_i < nCells; x_i++){
            for(int y_i = 0; y_i < nCells; y_i++){
                for(int z_i = 0; z_i < nCells; z_i++){
                    for (int x_j = x_i -1; x_j< x_i+2; x_j++){
                        for(int y_j = y_i -1; y_j< y_i+2; y_j++){
                            for(int z_j = z_i -1; z_j< z_i+2; z_j++){
                                if (x_j < 0 or x_j >= nCells or y_j < 0 or y_j >= nCells or z_j < 0 or z_j >= nCells){
                                    continue;
                                }
                                // if (BoolCheckedCell[x_i + y_i*nCells + z_i*pow(nCells,2)][x_j + y_j*nCells + z_j*pow(nCells,2)] == true){
                                //     continue;
                                // }
                                for (int i = rank; i<cellVector[x_i + y_i*nCells + z_i*pow(nCells,2)].size(); i+=size){
                                    for (int j = i+1; j<cellVector[x_j + y_j*nCells + z_j*pow(nCells,2)].size(); j++){
                                        if ((x_i + y_i*nCells + z_i*pow(nCells,2))==(x_j + y_j*nCells + z_j*pow(nCells,2)) && i == j) {
                                            continue;
                                        }
                                        int atom1 = cellVector[x_i + y_i*nCells + z_i*pow(nCells,2)][i];
                                        int atom2 = cellVector[x_j + y_j*nCells + z_j*pow(nCells,2)][j];
                                        double distance_squared = 0;
                                        std:: vector < double > position_i = positions[atom1];
                                        std:: vector < double > position_j = positions[atom2];
                                        distance_squared = ((position_i[0] - position_j[0]) * (position_i[0] - position_j[0])) + ((position_i[1] - position_j[1]) * (position_i[1] - position_j[1])) + ((position_i[2] - position_j[2]) * (position_i[2] - position_j[2]));
                                        if (distance_squared < cutoff_squared){
                                            neighbour_vector[atom1]++;
                                            neighbour_vector[atom2]++;
                                        }
                                    }
                                }
                                BoolCheckedCell[x_i + y_i*nCells + z_i*pow(nCells,2)][x_j + y_j*nCells + z_j*pow(nCells,2)] = true;
                                BoolCheckedCell[x_j + y_j*nCells + z_j*pow(nCells,2)][x_i + y_i*nCells + z_i*pow(nCells,2)] = true;
                            }
                        }
                    }
                }
            }
        }
    
    double end_time = MPI_Wtime();

    std::vector<double> global_neighbour_vector(N, 0);
    MPI_Allreduce(neighbour_vector.data(), global_neighbour_vector.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Finalize();
    if (rank == 0){
        if (vectorPrinter == 1){
            for (int i = 0; i<N; i++){
                std::cout << global_neighbour_vector[i] <<",";
            }
        }
        if (timePrinter == 1){
            std::cout << "Time taken: " << end_time - start_time << std::endl;
        }
    }
    
    return 0;

}


int main(int argc, char** argv) {
    
    int method = std::atoi(argv[2]);
    
    if (method == 0){
        bruteForce(argc, argv);
    }
    else if (method == 1){
        bruteForceMPIChunk(argc, argv);
    }
    else if (method == 2){
        bruteForceMPIDealer(argc, argv);
    }
    else if (method == 3){
        cellList(argc, argv);
    }
    else if (method == 4){
        cellListMPI(argc, argv);
    }
    else if (method == 5){
        cellListMPIDoubleIncrement(argc, argv);
    }
    else{
        std::cout << "Invalid method chosen" << std::endl;
    }

    
    return 0;
}

