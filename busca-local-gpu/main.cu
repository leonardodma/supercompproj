#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <iostream>

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

float total_path_dist(thrust::device_vector<city> d_vector_cities)
{
    // create another vector that is a copy of d_vector_cities, but with the first element moved to the end
    thrust::device_vector<city> d_vector_cities_shifted(d_vector_cities.size());
    thrust::copy(d_vector_cities.begin() + 1, d_vector_cities.end(), d_vector_cities_shifted.begin());
    d_vector_cities_shifted[d_vector_cities_shifted.size() - 1] = d_vector_cities[0];

    // Use transform to compute the distance between each pair of cities
    thrust::device_vector<float> d_vector_distances(d_vector_cities.size());

    // Use transform to compute the distance between each pair of cities
    thrust::transform(d_vector_cities.begin(), d_vector_cities.end(), d_vector_cities_shifted.begin(), d_vector_distances.begin(), city());

    // Use reduce to sum the distances
    return thrust::reduce(d_vector_distances.begin(), d_vector_distances.end(), 0.0f, thrust::plus<float>());
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

    // Create random number generator
    thrust::default_random_engine generator(10);

    // Copy cities to device
    thrust::device_vector<city> d_vector_cities_best = d_vector_cities;
    float best_distance = total_path_dist(d_vector_cities_best);

    for (int i = 0; i < num_solutions; i++)
    {
        // Current solution
        thrust::device_vector<city> d_vector_cities_current = d_vector_cities;

        // Shuffle cities on device
        thrust::shuffle(d_vector_cities.begin(), d_vector_cities.end(), generator);

        // Calculate distance of solution
        float current_distance = total_path_dist(d_vector_cities_current);

        // Swap j and j+1 of d_vector_cities_current to check if the distance is shorter
        for (int j = 0; j < num_cities - 1; j++)
        {
            thrust::swap(d_vector_cities_current[j], d_vector_cities_current[j + 1]);
            float new_distance = total_path_dist(d_vector_cities_current);

            if (new_distance < current_distance)
            {
                current_distance = new_distance;
            }
            else
            {
                thrust::swap(d_vector_cities_current[j], d_vector_cities_current[j + 1]);
            }
        }

        // If the current solution is better than the best solution, update the best solution
        if (current_distance < best_distance)
        {
            d_vector_cities_best = d_vector_cities_current;
            best_distance = current_distance;
        }
    }

    // Copy best solution to host
    thrust::host_vector<city> h_vector_cities_best = d_vector_cities_best;

    // Print best solution
    cout << best_distance;
    cout << " 0" << endl;
    for (int j = 0; j < num_cities; j++)
    {
        cout << h_vector_cities_best[j].id << " ";
    }

    cout << endl;

    return 0;
}