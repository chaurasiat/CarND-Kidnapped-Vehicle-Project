/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles=50;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	// Generate particles with normal distribution with mean on GPS values.
  for (int i = 0; i < num_particles; i++) {

    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;

    particles.push_back(particle);
	}

  // The filter is now initialized.
  is_initialized = true;



}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++) {

    // calculate new state
    if (fabs(yaw_rate) < 0.00001) {  
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    } 
    else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    // adding noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i = 0; i < observations.size(); i++) {
    
    // grab current observation
    LandmarkObs obs = observations[i];

    // init minimum distance to maximum possible
    double min_dist = numeric_limits<double>::max();

    
    int nearest_landmark_id = -1;
    
    for (int j = 0; j < predicted.size(); j++) {
      // grab current prediction
      LandmarkObs pred = predicted[j];
      
      // get distance between current/predicted landmarks
      double cur_dist = dist(obs.x, obs.y, pred.x, pred.y);

      // find the predicted landmark nearest the current observed landmark
      if (cur_dist < min_dist) {
        min_dist = cur_dist;
        nearest_landmark_id = pred.id;
      }
    }

    // set the observation's id to the nearest predicted landmark's id
    observations[i].id = nearest_landmark_id;
  }
	
	
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (int i = 0; i < num_particles; i++) {

    // get the particle x, y coordinates
    double particle_x = particles[i].x;
    double particle_y = particles[i].y;
    double particle_theta = particles[i].theta;

    // create a vector to hold the map landmark locations predicted to be within sensor range of the particle
    vector<LandmarkObs> predictions;

    
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {

      // get id and x,y coordinates
	   LandmarkObs landmrkobj;
      
	   landmrkobj.id=map_landmarks.landmark_list[j].id_i;
      landmrkobj.x = map_landmarks.landmark_list[j].x_f;
      landmrkobj.y=map_landmarks.landmark_list[j].y_f;
      
      // only consider landmarks within sensor range of the particle (rather than using the "dist" method considering a circular 
      // region around the particle, this considers a rectangular region but is computationally faster)
      if (fabs(landmrkobj.x - particle_x) <= sensor_range && fabs(landmrkobj - particle_y) <= sensor_range) {
		  
        // add prediction to vector
        predictions.push_back(landmrkobj);
      }
    }

    // create and populate a copy of the list of observations transformed from vehicle coordinates to map coordinates
    vector<LandmarkObs> transformed_obs;
    for (int j = 0; j < observations.size(); j++) {
		LandmarkObs landmrk;
		landmrk.id=observations[j].id;
      landmrk.x = cos(particle_theta)*observations[j].x - sin(particle_theta)*observations[j].y + particle_x;
      landmrk.y = sin(particle_theta)*observations[j].x + cos(particle_theta)*observations[j].y + particle_y;
      transformed_obs.push_back(landmrk);
    }

    // perform dataAssociation for the predictions and transformed observations on current particle
    dataAssociation(predictions, transformed_obs);

    // reinitialize weight,by default it's zero, which could have resulted in error
    particles[i].weight = 1.0;

	//# calculate normalization term
	double gauss_norm= 1/(2 * M_PI * std_landmark[0] * std_landmark[1]);
	double exponent;

    for (int j = 0; j < transformed_obs.size(); j++) {
      
      // placeholders for observation and associated prediction coordinates
      double ob_x, ob_y, pr_x, pr_y;
      ob_x = transformed_obs[j].x;
      ob_y = transformed_obs[j].y;

      int associated_prediction = transformed_obs[j].id;

      // get the x,y coordinates of the prediction associated with the current observation
      for (int k = 0; k < predictions.size(); k++) {
        if (predictions[k].id == associated_prediction) {
          pr_x = predictions[k].x;
          pr_y = predictions[k].y;
        }
      }

      // calculate weight for this observation with multivariate Gaussian
      double std_x = std_landmark[0];
      double std_y = std_landmark[1];
	  exponent=pow(pr_x-ob_x,2)/(2*pow(std_x, 2)) + (pow(pr_y-ob_y,2)/(2*pow(std_y, 2)));
      //double obs_w = (gauss_norm ) * exp( -( pow(pr_x-ob_x,2)/(2*pow(std_x, 2)) + (pow(pr_y-ob_y,2)/(2*pow(std_y, 2))) ) );
	  double obs_w = (gauss_norm ) * exp( -(exponent));
      // product of this obersvation weight with total observations weight
      particles[i].weight *= obs_w;
    }
  }


}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> resampled_particles;

  // get all of the current weights
  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }

  // generate random starting index for resampling wheel
  uniform_int_distribution<int> intdist(0, num_particles-1);
  int index = intdist(gen);


//   // get max weight
//   double max_weight = numeric_limits<double>::min();
// 	for(int i = 0; i < num_particles; i++) {
// 			if(particles[i].weight > max_weight) {
// 				max_weight = particles[i].weight;
// 			}
// 		}
// 		//
  double max_weight = *max_element(weights.begin(), weights.end());

  
  uniform_real_distribution<double> realdist(0.0, max_weight);

  double beta = 0.0;

  
  for (int i = 0; i < num_particles; i++) {
    beta += realdist(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampled_particles.push_back(particles[index]);
  }

  particles = resampled_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
