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

// Function to format the output of the program
void format_output(float total_dist, vector<city> cities)
{
    cout << total_dist;
    cout << " 1" << endl;
    for (size_t i = 0; i < cities.size(); i++)
    {
        cout << cities[i].id << " ";
    }
    cout << endl;
}

// Function to calculate the distance between two cities
float dist(city a, city b)
{
    float diffX = b.x - a.x;
    float diffY = b.y - a.y;

    return sqrt(diffX * diffX + diffY * diffY);
}

// Function to calculate the total distance of a path
float get_total_dist(vector<city> vector_cities)
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

vector<city> global_min(vector<city> possible_cities, vector<city> current_path, int &leafs)
{
    // If there are no more possible cities to visit, return the current path (the best path)
    if (possible_cities.size() == 0)
    {
        leafs++;
        return current_path;
    }

    vector<vector<city>> children_paths(possible_cities.size());
    for (size_t i = 0; i < possible_cities.size(); i++)
    {
        // New path will add a city from de possible cities to visit vector
        vector<city> new_path = current_path;
        new_path.push_back(possible_cities[i]);

        // Possible cities do visit will be the same, but without the city that was added to the path
        vector<city> new_possible_cities = possible_cities;
        new_possible_cities.erase(new_possible_cities.begin() + i);

        // Recursive call to the function
        // Put all children paths in a vector
        children_paths[i] = global_min(new_possible_cities, new_path, leafs);
    }

    // Get the best path from the children paths (more economical than returning all paths)
    vector<city> best_children_path = children_paths[0];
    float best_children_dist = get_total_dist(best_children_path);

    for (size_t i = 1; i < children_paths.size(); i++)
    {
        float current_children_dist = get_total_dist(children_paths[i]);
        if (current_children_dist < best_children_dist)
        {
            best_children_path = children_paths[i];
            best_children_dist = current_children_dist;
        }
    }

    return best_children_path;
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
        float x, y;
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
    int leafs = 0;
    vector<city> best_path = global_min(vector_cities, vector<city>(), leafs);
    cerr << "num_neafs " << leafs << endl;
    float total_dist = get_total_dist(best_path);
    format_output(total_dist, best_path);

    return 0;
}
