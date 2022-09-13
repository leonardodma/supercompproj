#include <iostream>
#include <cmath>
#include <algorithm>
using namespace std;

struct city
{
    float x, y;
};

float dist(city a, city b)
{
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) * 1.0);
}

void format_output(float total_dist, vector<int> visited_cities)
{
    cout << total_dist;
    cout << " 0" << endl;
    for (size_t i = 0; i < visited_cities.size(); i++)
    {
        cout << visited_cities[i] << " ";
    }
    cout << endl;
}

int main(int argc, char const *argv[])
{
    int num_cities;
    cin >> num_cities;
    city array_cities[num_cities];

    for (int i = 0; i < num_cities; i++)
    {
        float x;
        float y;
        cin >> x;
        cin >> y;
        city c;
        c.x = x;
        c.y = y;
        array_cities[i] = c;
    }

    vector<city> vector_cities(array_cities, array_cities + num_cities);
    vector<int> visited_cities = {0};
    float total_distance = 0.0;

    for (int i = 0; i < num_cities; i++)
    {
        int current_city = visited_cities.back();
        float minor_distance = 1000000000.0;

        int num_visited_cities = visited_cities.size();
        if (num_visited_cities < num_cities)
        {
            int minor_city = 0;

            for (int j = 0; j < num_cities; j++)
            {
                int next_city = j;

                if (!count(visited_cities.begin(), visited_cities.end(), next_city))
                {
                    float distance = dist(vector_cities[current_city], vector_cities[next_city]);
                    // cout << "Distance from " << current_city << " to " << next_city << " is " << distance << endl;

                    if (distance < minor_distance)
                    {
                        minor_distance = distance;
                        minor_city = next_city;
                    }
                }
            }

            visited_cities.push_back(minor_city);
            total_distance += minor_distance;
        }
        else
        {
            total_distance += dist(vector_cities[current_city], vector_cities[0]);
        }
    }

    format_output(total_distance, visited_cities);

    return 0;
}
