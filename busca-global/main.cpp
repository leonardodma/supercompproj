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
    cout << " 0" << endl;
    for (size_t i = 0; i < cities.size(); i++)
    {
        cout << cities[i].id << " ";
    }
    cout << endl;
}

// Function to calculate the distance between two cities
float dist(city a, city b)
{
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) * 1.0);
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

// Recursive function to get all the possible paths
void permute(vector<vector<city>> &all_possibile_paths, vector<city> current_path, size_t position)
{
    // If the current path has the same number of cities as the total number of cities, return the current path
    if (position == current_path.size() - 1)
    {
        all_possibile_paths.push_back(current_path);
        return;
    }

    // Iterate over the cities
    for (size_t i = position; i < current_path.size(); i++)
    {
        // Swap the current city with the city in the position
        swap(current_path[i], current_path[position]);

        // Call the function again
        permute(all_possibile_paths, current_path, position + 1);

        // Swap the current city with the city in the position
        swap(current_path[i], current_path[position]);
    }
}

// Function to get all the possible paths
vector<vector<city>> get_all_possible_paths(vector<city> vector_cities)
{
    // Declare a vector to store all the possible paths
    vector<vector<city>> all_possibile_paths;

    // Call the recursive function
    permute(all_possibile_paths, vector_cities, 0);

    return all_possibile_paths;
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
    vector<city> best_path(array_cities, array_cities + num_cities);
    float best_distance = get_total_dist(best_path);

    // Get all the possible paths
    vector<vector<city>> all_possible_paths = get_all_possible_paths(best_path);

    // Iterate over all the possible paths
    for (size_t i = 0; i < all_possible_paths.size(); i++)
    {
        // Get the total distance of the current path
        float total_distance = get_total_dist(all_possible_paths[i]);

        // If the total distance of the current path is less than the best distance, update the best distance and the best path
        if (total_distance < best_distance)
        {
            best_distance = total_distance;
            best_path = all_possible_paths[i];
        }
    }

    format_output(best_distance, best_path);

    return 0;
}
