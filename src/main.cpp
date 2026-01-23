#include <iostream>
#include <math.h>
#include <random>
#include <mpi.h>

class MCSimulation {
public:
  MCSimulation(MPI_Comm comm);
  void run();

private:
  int generate_initial_state(int num_particles, double box_length);
  double get_particle_energy(int particle_count, double box_length, int i_particle, double cutoff2);
  double lennard_jones_potential(double rij2);
  double minimum_image_distance(double *r_i, double *r_j, double box_length);
  bool accept_or_reject( double delta_e, double beta );
  double adjust_displacement( int n_trials, int n_accept, double max_displacement );

  std::vector<double> coordinates;

  // MPI information
  int mpi_size, mpi_rank;
  MPI_Comm mpi_comm;
  std::vector<int> mpi_start_index;
  std::vector<int> mpi_end_index;

  // Random number generators
  std::mt19937 mt;
  std::uniform_real_distribution<double> dist;
};

MCSimulation::MCSimulation(MPI_Comm comm) {
  mt = std::mt19937(1);
  //std::random_device rd;
  //mt = std::mt19937(rd());
  dist = std::uniform_real_distribution<double>(0.0, 1.0);

  mpi_comm = comm;
  MPI_Comm_size(mpi_comm, &mpi_size);
  MPI_Comm_rank(mpi_comm, &mpi_rank);
  mpi_start_index.resize(mpi_size);
  mpi_end_index.resize(mpi_size);
}

// Generate an initial set of coordinates
int MCSimulation::generate_initial_state(int num_particles, double box_length) {
  int particles_per_side = std::ceil( std::pow(num_particles, 1.0 / 3.0) );
  double particle_spacing = box_length / particles_per_side;
  for (int iparticle = 0; iparticle < mpi_end_index[mpi_rank] - mpi_start_index[mpi_rank]; ++iparticle) {
    int offset = mpi_start_index[mpi_rank];
    int ix = (iparticle + offset) % particles_per_side;
    int iy = ((iparticle + offset) / particles_per_side) % particles_per_side;
    int iz = (iparticle + offset) / (particles_per_side * particles_per_side);
    coordinates[3*iparticle + 0] = ( ix + ( 0.1 * dist(mt) ) ) * particle_spacing;
    coordinates[3*iparticle + 1] = ( iy + ( 0.1 * dist(mt) ) ) * particle_spacing;
    coordinates[3*iparticle + 2] = ( iz + ( 0.1 * dist(mt) ) ) * particle_spacing;
  }

  return 0;
}

// Evaluate the LJ potential for a given squared distance
double MCSimulation::lennard_jones_potential(double rij2) {
  double sig_by_r2 = 1.0 / rij2;
  double sig_by_r6 = sig_by_r2*sig_by_r2*sig_by_r2;
  double sig_by_r12 = sig_by_r6*sig_by_r6;
  return 4.0 * ( sig_by_r12 - sig_by_r6 );
}

// Compute the minimum image distance between two particles
double MCSimulation::minimum_image_distance(double *r_i, double *r_j, double box_length) {
  double rij[3];
  rij[0] = r_i[0] - r_j[0] - box_length * round( (r_i[0] - r_j[0]) / box_length );
  rij[1] = r_i[1] - r_j[1] - box_length * round( (r_i[1] - r_j[1]) / box_length );
  rij[2] = r_i[2] - r_j[2] - box_length * round( (r_i[2] - r_j[2]) / box_length );

  double rij2 = ( rij[0] * rij[0] ) + ( rij[1] * rij[1] ) + ( rij[2] * rij[2] );
  return rij2;
}

// Compute the energy of a particle
double MCSimulation::get_particle_energy(int particle_count, double box_length, int i_particle, double cutoff2) {
  double e_total = 0.0;
  double e_partial = 0.0;
  int i_particle_rank;
  for ( int irank = 0; irank < mpi_size; ++irank) {
    if ( i_particle >= mpi_start_index[irank] && i_particle < mpi_end_index[irank] ) i_particle_rank = irank;
  }
  std::vector<double> i_position(3);
  if ( mpi_rank == i_particle_rank ) {
    i_position[0] = coordinates[3 * ( i_particle - mpi_start_index[mpi_rank] ) + 0];
    i_position[1] = coordinates[3 * ( i_particle - mpi_start_index[mpi_rank] ) + 1];
    i_position[2] = coordinates[3 * ( i_particle - mpi_start_index[mpi_rank] ) + 2];
  }
  MPI_Bcast(i_position.data(), 3, MPI_DOUBLE, i_particle_rank, mpi_comm);



  for (int j_particle = mpi_start_index[mpi_rank]; j_particle < mpi_end_index[mpi_rank]; ++j_particle) {
    if ( i_particle != j_particle ) {
      double *j_position = &coordinates[3*(j_particle - mpi_start_index[mpi_rank])];
      double rij2 = minimum_image_distance( i_position.data(), j_position, box_length );
      if ( rij2 < cutoff2 ) {
        e_partial += lennard_jones_potential(rij2);
      }
    }
  }

  MPI_Allreduce(&e_partial, &e_total, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);

  return e_total;
}

// Accept or reject a move based on the energy difference and system temperature
bool MCSimulation::accept_or_reject( double delta_e, double beta ) {
  bool accept = false;
  if ( delta_e < 0.0 ) {
    accept = true;
  }
  else {
    double random_number = dist(mt);
    double p_acc = exp(-beta * delta_e);

    if ( random_number < p_acc ) {
      accept = true;
    }
    else {
      accept = false;
    }
  }
  return accept;
}

// Change the acceptance criteria to get the desired rate
double MCSimulation::adjust_displacement( int n_trials, int n_accept, double max_displacement ) {
  double acc_rate = double(n_accept) / double(n_trials);
  double new_displacement = max_displacement;
  if ( acc_rate < 0.38 ) {
    new_displacement *= 0.8;
  }
  else if ( acc_rate > 0.42 ) {
    new_displacement *= 1.2;
  }
  return new_displacement;
}

void MCSimulation::run() {

  double start_simulation_time = MPI_Wtime();
  double total_energy_time = 0.0;
  double total_decision_time = 0.0;


  /******************
  * Parameter setup *
  ******************/

  double reduced_temperature = 0.9;
  double reduced_density = 0.9;
  int n_steps = 100000;
  int freq = 1000;
  int num_particles = 10000;
  double simulation_cutoff = 3.0;
  double max_displacement = 0.1;
  bool tune_displacement = true;
  bool plot = true;

  double box_length = cbrt(num_particles / reduced_density);
  double beta = 1.0 / reduced_temperature;
  double simulation_cutoff2 = simulation_cutoff*simulation_cutoff;
  int n_trials = 0;
  int n_accept = 0;

  int current_start_index = 0;
  for ( int irank = 0; irank < mpi_size; ++irank ) {
    int nparticles_to_calc = num_particles / mpi_size;
    if ( irank < num_particles % mpi_size ) ++nparticles_to_calc;
    mpi_start_index[irank] = current_start_index;
    mpi_end_index[irank] = current_start_index + nparticles_to_calc;
    current_start_index += nparticles_to_calc;
  }

  coordinates = std::vector<double>(3 * (mpi_end_index[mpi_rank] - mpi_start_index[mpi_rank]));

  /*************************
  * Monte Carlo Simulation *
  *************************/
  generate_initial_state(num_particles, box_length);

  // Total energy of the system, relative to the first step
  double total_energy = 0.0;

  // Beginning of main MC iterative loop
  n_trials = 0;
  for (int i_step = 0; i_step < n_steps; ++i_step) {
    n_trials += 1;
    int i_particle;
    if ( mpi_rank == 0 ) {
      i_particle = floor( double(num_particles) * dist(mt) );
    }
    MPI_Bcast(&i_particle, 1, MPI_INT, 0, mpi_comm);
    double random_displacement[3];
    if ( mpi_rank == 0 ) {
      for (int i = 0; i < 3; ++i) {
        random_displacement[i] = ( ( 2.0 * dist(mt) ) - 1.0 ) * max_displacement;
      }
    }
    MPI_Bcast(&random_displacement[0], 3, MPI_DOUBLE, 0, mpi_comm);

    // get the current energy of the test particle
    double start_energy_time = MPI_Wtime();
    double current_energy = get_particle_energy( num_particles, box_length, i_particle, simulation_cutoff2 );
    total_energy_time += MPI_Wtime() - start_energy_time;

    // get the new coordinates of the test particle
    int i_particle_rank;
    for ( int irank = 0; irank < mpi_size; ++irank) {
      if ( i_particle >= mpi_start_index[irank] && i_particle < mpi_end_index[irank] ) i_particle_rank = irank;
    }
    if ( mpi_rank == i_particle_rank ) {
      for (int i = 0; i < 3; ++i) {
        coordinates[3*(i_particle - mpi_start_index[mpi_rank]) + i] += random_displacement[i];
        coordinates[3*(i_particle - mpi_start_index[mpi_rank]) + i] -= box_length * round(coordinates[3*(i_particle - mpi_start_index[mpi_rank]) + i] / box_length);
      }
    }

    // get the new energy of the test particle
    start_energy_time = MPI_Wtime();
    double proposed_energy = get_particle_energy( num_particles, box_length, i_particle, simulation_cutoff2 );
    total_energy_time += MPI_Wtime() - start_energy_time;

    // test whether to accept or reject this step
    double start_decision_time = MPI_Wtime();
    double delta_e = proposed_energy - current_energy;
    bool accept;
    if ( mpi_rank == 0 ) {
      accept = accept_or_reject(delta_e, beta);
    }
    MPI_Bcast(&accept, 1, MPI_CXX_BOOL, 0, mpi_comm);
    if (accept) {
      total_energy += delta_e;
      n_accept += 1;
    }
    else {
      // revert the position of the test particle
      if ( mpi_rank == i_particle_rank ) {
        for (int i = 0; i < 3; ++i) {
          coordinates[3*(i_particle - mpi_start_index[mpi_rank]) + i] -= random_displacement[i];
          coordinates[3*(i_particle - mpi_start_index[mpi_rank]) + i] -= box_length * round(coordinates[3*(i_particle - mpi_start_index[mpi_rank]) + i] / box_length);
        }
      }
    }

    if ( (i_step + 1) % freq == 0 ) {
      if ( mpi_rank == 0 ) {
        std::cout << i_step + 1 << " " << total_energy << '\n';
      }

      if ( tune_displacement ) {
        max_displacement = adjust_displacement(n_trials, n_accept, max_displacement);
        n_trials = 0;
        n_accept = 0;
      }
    }

    total_decision_time += MPI_Wtime() - start_decision_time;
  }

  if ( mpi_rank == 0 ) {
    std::cout << "Total simulation time: " << MPI_Wtime() - start_simulation_time << '\n';
    std::cout << "    Energy time:      " << total_energy_time << '\n';
    std::cout << "    Decision time:    " << total_decision_time << '\n';
  }

}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  MCSimulation mysimulation(MPI_COMM_WORLD);
  mysimulation.run();
  MPI_Finalize();
  return 0;
}

