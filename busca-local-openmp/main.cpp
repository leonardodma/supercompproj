#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <omp.h>
using namespace std;

struct city
{
    int id;
    float x, y;
};

struct solution
{
    float dist;
    vector<city> cities;
};

vector<city> read_file(int num_cities)
{
    vector<city> vector_cities;
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
        vector_cities.push_back(c);
    }
    return vector_cities;
}

float dist(city a, city b)
{
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) * 1.0);
}

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

void local_search(solution *local_solution, int num_cities, default_random_engine generator)
{
    vector<city> current_solution = local_solution->cities;
    shuffle(current_solution.begin(), current_solution.end(), generator);
    float current_distance = total_dist(current_solution);

    for (int j = 0; j < num_cities - 1; j++)
    {
        swap(current_solution[j], current_solution[j + 1]);
        float new_distance = total_dist(current_solution);
        if (new_distance < current_distance)
        {
            // Improvement
            current_distance = new_distance;
        }
        else
        {
            // Swab back
            swap(current_solution[j], current_solution[j + 1]);
        }
    }
// https://www.ibm.com/docs/en/zos/2.4.0?topic=processing-pragma-omp-critical
#pragma omp critical
    {
        if (current_distance < local_solution->dist)
        {
            local_solution->dist = current_distance;
            local_solution->cities = current_solution;
        }
    }
}

void format_output(solution best_solution)
{
    cout << best_solution.dist << " 0" << endl;
    for (size_t i = 0; i < best_solution.cities.size(); i++)
    {
        cout << best_solution.cities[i].id << " ";
    }
    cout << endl;
}

int main(int argc, char const *argv[])
{
    // Get number of cities
    int num_cities;
    cin >> num_cities;
    int num_solutions = 10 * num_cities;

    // Defining seed
    unsigned seed = 10;
    default_random_engine generator(seed);

    // Put cities in a vector reading from file
    vector<city> vector_cities = read_file(num_cities);

    // Initialize solution struct
    solution local_solution;
    local_solution.cities = vector_cities;
    local_solution.dist = total_dist(vector_cities);

// Get local search solution parallelized
#pragma omp parallel
    {
#pragma omp for
        for (int i = 0; i < num_solutions; i++)
        {
            local_search(&local_solution, num_cities, generator);
        }
    }

    // Print output
    format_output(local_solution);

    return 0;
}