#include <iostream>
#include <math.h>
#include <random>
#include <mpi.h>

class MCSimulation {
  public:
    MCSimulation();
    void run();

  private:
    int generate_initial_state(int num_particles, double box_length);
    double get_particle_energy(int particle_count, double box_length, int i_particle, double cutoff2);
    double lennard_jones_potential(double rij2);
    double minimum_image_distance(double *r_i, double *r_j, double box_length);
    bool accept_or_reject( double delta_e, double beta );
    double adjust_displacement( int n_trials, int n_accept, double max_displacement );

    std::vector<double> coordinates;

    // Random number generators
    std::mt19937 mt;
    std::uniform_real_distribution<double> dist;
};

MCSimulation::MCSimulation() {
  mt = std::mt19937(1);
  dist = std::uniform_real_distribution<double>(0.0, 1.0);
}

// Generate an initial set of coordinates
int MCSimulation::generate_initial_state(int num_particles, double box_length) {
  int particles_per_side = std::ceil( std::pow(num_particles, 1.0 / 3.0) );
  double particle_spacing = box_length / particles_per_side;
  for (int iparticle = 0; iparticle < num_particles; ++iparticle) {
    int ix = iparticle % particles_per_side;
    int iy = (iparticle / particles_per_side) % particles_per_side;
    int iz = iparticle / (particles_per_side * particles_per_side);
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
  double *i_position = &coordinates[3*i_particle];

  for (int j_particle=0; j_particle < particle_count; ++j_particle) {
    if ( i_particle != j_particle ) {
      double *j_position = &coordinates[3*j_particle];
      double rij2 = minimum_image_distance( i_position, j_position, box_length );
      if ( rij2 < cutoff2 ) {
	e_total += lennard_jones_potential(rij2);
      }
    }
  }

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

  int world_size, my_rank;
  MPI_Comm world_comm = MPI_COMM_WORLD;
  MPI_Comm_size(world_comm, &world_size);
  MPI_Comm_rank(world_comm, &my_rank);

  /******************
  * Parameter setup *
  ******************/

  double reduced_temperature = 0.9;
  double reduced_density = 0.9;
  int n_steps = 500000;
  int freq = 1000;
  int num_particles = 100;
  double simulation_cutoff = 3.0;
  double max_displacement = 0.1;
  bool tune_displacement = true;
  bool plot = true;

  double box_length = cbrt(num_particles / reduced_density);
  double beta = 1.0 / reduced_temperature;
  double simulation_cutoff2 = simulation_cutoff*simulation_cutoff;
  int n_trials = 0;
  int n_accept = 0;
  coordinates = std::vector<double>(3*num_particles);

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
    int i_particle = floor( double(num_particles) * dist(mt) );
    double random_displacement[3];
    for (int i = 0; i < 3; ++i) {
      random_displacement[i] = ( ( 2.0 * dist(mt) ) - 1.0 ) * max_displacement;
    }

    // get the current energy of the test particle
    double start_energy_time = MPI_Wtime();
    double current_energy = get_particle_energy( num_particles, box_length, i_particle, simulation_cutoff2 );
    total_energy_time += MPI_Wtime() - start_energy_time;

    // get the new coordinates of the test particle
    for (int i = 0; i < 3; ++i) {
      coordinates[3*i_particle + i] += random_displacement[i];
      coordinates[3*i_particle + i] -= box_length * round(coordinates[3*i_particle + i] / box_length);
    }

    // get the new energy of the test particle
    start_energy_time = MPI_Wtime();
    double proposed_energy = get_particle_energy( num_particles, box_length, i_particle, simulation_cutoff2 );
    total_energy_time += MPI_Wtime() - start_energy_time;

    // test whether to accept or reject this step
    double start_decision_time = MPI_Wtime();
    double delta_e = proposed_energy - current_energy;
    bool accept = accept_or_reject(delta_e, beta);
    if (accept) {
      total_energy += delta_e;
      n_accept += 1;
    }
    else {
      // revert the position of the test particle
      for (int i = 0; i < 3; ++i) {
	coordinates[3*i_particle + i] -= random_displacement[i];
	coordinates[3*i_particle + i] -= box_length * round(coordinates[3*i_particle + i] / box_length);
      }
    }

    if ( (i_step + 1) % freq == 0 ) {
      if ( my_rank == 0 ) {
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

  if ( my_rank == 0 ) {
    std::cout << "Total simulation time: " << MPI_Wtime() - start_simulation_time << '\n';
    std::cout << "    Energy time:      " << total_energy_time << '\n';
    std::cout << "    Decision time:    " << total_decision_time << '\n';
  }
  
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  MCSimulation mysimulation;
  mysimulation.run();
  MPI_Finalize();
  return 0;
}

