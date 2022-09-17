#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <tuple>
using namespace std;

// Struct to store the coordinates of a city, and its index
struct city
{
    int id;
    float x, y;
};

// Function to calculate the distance between two cities
float dist(city a, city b)
{
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) * 1.0);
}

// Function to calculate the total distance of a path
float total_dist(vector<city> vector_cities)
{
    float total_distance = 0.0;
    for (size_t i = 0; i < vector_cities.size() - 1; i++)
    {
        total_distance += dist(vector_cities[i], vector_cities[i + 1]);
    }

    city first_city = vector_cities[0];
    city last_city = vector_cities[vector_cities.size() - 1];
    total_distance += dist(last_city, first_city);

    return total_distance;
}

// Function to get the heuristic path: the nearest neighbor
tuple<vector<city>, float> get_heuristic_solution(vector<city> vector_cities, int num_cities)
{
    // Declare variables to store the visited cities and the total distance
    vector<int> visited_cities_index = {0};
    vector<city> visited_cities = {vector_cities[0]};
    float total_distance = 0.0;

    // Iterate over the cities
    for (int i = 0; i < num_cities; i++)
    {
        // Get the current city: the last city in the visited cities vector
        int current_city = visited_cities_index.back();

        // Declare variable to store the minor distance
        float minor_distance = 1000000000.0;

        // Get the number of visited cities
        int num_visited_cities = visited_cities_index.size();

        // If the number of visited cities is less than the number of cities: we still have cities to visit
        if (num_visited_cities < num_cities)
        {
            // Declare variable to store the closest city
            int closest_city = 0;

            for (int j = 0; j < num_cities; j++)
            {
                int next_city = j;

                // If the next city is not in the visited cities vector
                if (!count(visited_cities_index.begin(), visited_cities_index.end(), next_city))
                {
                    // Calculate the distance between the current city and the next city
                    float distance = dist(vector_cities[current_city], vector_cities[next_city]);

                    // If the distance is less than the minor distance
                    if (distance < minor_distance)
                    {
                        // Update the minor distance and the closest city
                        minor_distance = distance;
                        closest_city = next_city;
                    }
                }
            }

            // Update the total distance and the visited cities
            visited_cities_index.push_back(closest_city);
            visited_cities.push_back(vector_cities[closest_city]);
            total_distance += minor_distance;
        }
        // Else go back to the first city
        else
        {
            total_distance += dist(vector_cities[current_city], vector_cities[0]);
        }
    }

    return make_tuple(visited_cities, total_distance);
}

// Function to format the output of the program
void format_output(float total_dist, vector<city> cities)
{
    cout << total_dist;
    cout << " 0" << endl;
    for (size_t i = 0; i < cities.size(); i++)
    {
        cout << cities[i].id << " ";
    }
    cout << endl;
}

int main(int argc, char const *argv[])
{
    // Get number of cities
    int num_cities;
    cin >> num_cities;
    city array_cities[num_cities];

    // Get locations of cities using the input
    for (int i = 0; i < num_cities; i++)
    {
        float x;
        float y;
        cin >> x;
        cin >> y;
        city c;
        c.id = i;
        c.x = x;
        c.y = y;
        array_cities[i] = c;
    }

    // Put cities in a vector
    vector<city> vector_cities(array_cities, array_cities + num_cities);

    // Get the heuristic solution
    vector<city> heuristic_solution;
    float heuristic_distance;
    tie(heuristic_solution, heuristic_distance) = get_heuristic_solution(vector_cities, num_cities);

    // Get the best solution using the local search
    int num_solutions = 10 * num_cities;
    vector<city> best_solution = heuristic_solution;
    float best_distance = heuristic_distance;

    // Defining seed
    unsigned seed = 10;
    default_random_engine generator(seed);
    uniform_int_distribution<int> random_index(0, num_cities - 1);

    for (int i = 0; i < num_solutions; i++)
    {
        // Get two random cities
        int random_city_1 = random_index(generator);
        int random_city_2 = random_index(generator);

        // Swap the cities
        swap(best_solution[random_city_1], best_solution[random_city_2]);

        // Calculate the new distance
        float new_distance = total_dist(best_solution);

        // Cerr of the local search
        cerr << "Local: " << new_distance << " ";
        for (size_t i = 0; i < best_solution.size(); i++)
        {
            cerr << best_solution[i].id << " ";
        }
        cerr << endl;

        // If the new distance is less than the best distance
        if (new_distance < best_distance)
        {
            // Update the best distance
            best_distance = new_distance;
        }
        // Else swap the cities again
        else
        {
            swap(best_solution[random_city_1], best_solution[random_city_2]);
        }
    }

    format_output(best_distance, best_solution);

    return 0;
}
