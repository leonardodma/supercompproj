#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
using namespace std;

struct city
{
    int id;
    float x, y;
};

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
    int num_solutions = 10 * num_cities;

    // Get locations of cities
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

    // Defining seed
    unsigned seed = 10;
    default_random_engine generator(seed);
    // Put cities in a vector
    vector<city> vector_cities(array_cities, array_cities + num_cities);
    vector<city> best_solution = vector_cities;
    float best_distance = total_dist(best_solution);

    for (int i = 0; i < num_solutions; i++)
    {
        vector<city> current_solution = vector_cities;
        shuffle(current_solution.begin(), current_solution.end(), generator);
        float current_distance = total_dist(current_solution);

        for (int j = 0; j < num_cities - 1; j++)
        {
            swap(current_solution[j], current_solution[j + 1]);
            float new_distance = total_dist(current_solution);
            if (new_distance < current_distance)
            {
                current_distance = new_distance;
            }
            else
            {
                swap(current_solution[j], current_solution[j + 1]);
            }
        }

        if (current_distance < best_distance)
        {
            best_distance = current_distance;
            best_solution = current_solution;
        }

        // Cout error
        cerr << "local: " << current_distance << " ";
        for (size_t k = 0; k < current_solution.size(); k++)
        {
            cerr << current_solution[k].id << " ";
        }
        cerr << endl;
    }

    // Print output
    format_output(best_distance, best_solution);

    return 0;
}
