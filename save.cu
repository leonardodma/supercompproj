#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <thrust/device_ptr.h>
#include <iostream>
#include <vector>

using namespace std;

struct city
{
    int id;
    float x, y;

    __host__ __device__ float operator()(const city &a, const city &b) const
    {
        float dx = a.x - b.x;
        float dy = a.y - b.y;
        return sqrt(dx * dx + dy * dy);
    }
};

struct best_path
{
    // Define the operator() that will be used by thrust::transform
    int num_interations;
    int city_amount;
    thrust::device_ptr<city> dp;

    best_path(thrust::device_ptr<city> dp_, int num_interations_, int city_amount_) : num_interations(num_interations_), dp(dp_), city_amount(city_amount_){};

    __host__ __device__ float operator()(const int &id) const
    {
        // Get the city vector based on the id, num_interations and the device pointer
        thrust::device_ptr<city> vector_cities_dp = dp + (id * num_interations);

        // Calculate the total distance of the path
        float total_distance = 0;
        for (int i = 0; i < city_amount - 1; i++)
        {
            total_distance += city()(vector_cities_dp[i], vector_cities_dp[i + 1]);
        }

        // Sum the distance of the last city to the first city
        total_distance += city()(vector_cities_dp[city_amount - 1], vector_cities_dp[0]);

        return total_distance;
    }
};

float local_search(thrust::device_ptr<city> dp, int num_interations, int num_cities)
{
    // Crate a device vector to store to store the the indexes: 0, 1, 2, ..., num_interations
    thrust::device_vector<int> indexes(num_interations);
    thrust::sequence(indexes.begin(), indexes.end());

    // Create a device vector to store the distances
    thrust::device_vector<float> distances(num_interations);

    // Use transform to call the best_path operator() for each index
    thrust::transform(indexes.begin(), indexes.end(), distances.begin(), best_path(dp, num_interations, num_cities));

    // Print the distances
    thrust::host_vector<float> h_distances = distances;
    for (int i = 0; i < h_distances.size(); i++)
    {
        cout << h_distances[i] << endl;
    }

    return 0.0;
}

thrust::host_vector<city> read_file(int num_cities)
{
    thrust::host_vector<city> h_vector_cities(num_cities);
    for (int i = 0; i < num_cities; i++)
    {
        float x;
        float y;
        cin >> x;
        cin >> y;
        h_vector_cities[i].id = i;
        h_vector_cities[i].x = x;
        h_vector_cities[i].y = y;
    }

    return h_vector_cities;
}

int main()
{
    int num_cities;
    cin >> num_cities;
    int num_solutions = num_cities * 10;

    // Read cities from file
    thrust::host_vector<city> h_vector_cities = read_file(num_cities);

    // Copy cities to device
    thrust::device_vector<city> d_vector_cities = h_vector_cities;

    // Vector to store all vectors shuffled
    thrust::device_vector<city> all_shuffled_paths(num_cities * num_solutions);

    // Create random number generator
    thrust::default_random_engine generator(10);

    for (int i = 0; i < num_solutions; i++)
    {
        // Current solution
        thrust::device_vector<city> d_vector_cities_current = d_vector_cities;

        // Shuffle cities on device
        thrust::shuffle(d_vector_cities.begin(), d_vector_cities.end(), generator);

        // Copy shuffled cities to all_shuffled_paths
        thrust::copy(d_vector_cities.begin(), d_vector_cities.end(), all_shuffled_paths.begin() + i * num_cities);
    }

    // Get pointer to all_shuffled_paths
    thrust::device_ptr<city> dp = &all_shuffled_paths[0];

    local_search(dp, num_solutions, num_cities);

    /* for (int i = 0; i < num_solutions; i++)
    {
        for (int j = 0; j < num_cities; j++)
        {
            cout << h_vector_all_shuffled_paths[i * num_cities + j].id << " ";
        }
        cout << endl;
    } */

    return 0;
}