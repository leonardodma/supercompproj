#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <omp.h>
using namespace std;

struct product
{
    int id;
    int weight;
    int price;
};

void print_vector(vector<product> myvector)
{
    for (size_t i = 0; i < myvector.size(); i++)
    {
        cout << myvector[i].id << " ";
    }
    cout << endl;
}

void format_output(vector<product> bag)
{
    int sum_weight = 0;
    int sum_price = 0;

    for (size_t i = 0; i < bag.size(); i++)
    {
        sum_weight += bag[i].weight;
        sum_price += bag[i].price;
    }

    print_vector(bag);
}

bool check_valid(vector<product> selected_products, int max_weight)
{
    int sum_weight = 0;
    for (size_t i = 0; i < selected_products.size(); i++)
    {
        sum_weight += selected_products[i].weight;
    }
    return sum_weight <= max_weight;
}

int get_total_price(vector<product> selected_products)
{
    int sum_price = 0;
    for (size_t i = 0; i < selected_products.size(); i++)
    {
        sum_price += selected_products[i].price;
    }
    return sum_price;
}

vector<product> global_min(vector<product> possible_products, vector<product> current_selected_products, int max_weight)
{
    // If there are no more possible cities to visit, return the current path (the best path)
    if (possible_products.size() == 0)
    {
        return current_selected_products;
    }

    vector<vector<product>> children_products;
    for (size_t i = 0; i < possible_products.size(); i++)
    {
        // New path will add a product from de possible cities to visit vector
        vector<product> new_selected_products = current_selected_products;
        new_selected_products.push_back(possible_products[i]);

        if (check_valid(new_selected_products, max_weight))
        {
            // New possible cities to visit will be the possible cities to visit without the city that was added to the new path
            vector<product> new_possible_products = possible_products;
            new_possible_products.erase(new_possible_products.begin() + i);

            // Recursive call using openmp task
            vector<product> child_products;
#pragma omp task
            {
                child_products = global_min(new_possible_products, new_selected_products, max_weight);
            }
            children_products.push_back(child_products);
        }
    }

    // Check if there are no valid childs
    if (children_products.size() == 0)
    {
        return current_selected_products;
    }

    // Get the best path from the children paths (more economical than returning all paths)
    vector<product> best_children_selection = children_products[0];
    float best_children_total_price = get_total_price(best_children_selection);

    for (size_t i = 1; i < children_products.size(); i++)
    {
        float current_children_selection = get_total_price(children_products[i]);
        if (current_children_selection > best_children_total_price)
        {
            best_children_selection = children_products[i];
            best_children_total_price = current_children_selection;
        }
    }

    return best_children_selection;
}

int main(int argc, char const *argv[])
{
    cout << "Number of threads: " << omp_get_max_threads() << endl;
    int n_max;
    int weight_max;
    cin >> n_max;
    cin >> weight_max;
    product array_products[n_max];

    for (int i = 0; i < n_max; i++)
    {
        int wi;
        int vi;
        cin >> wi;
        cin >> vi;
        product x;
        x.id = i;
        x.weight = wi;
        x.price = vi;
        array_products[i] = x;
    }

    vector<product> all_products(array_products, array_products + n_max);

    // Paralleling the problem using openmp
    // Create master and call the recursive function
    vector<product> bag1;

#pragma omp master
    {
        bag1 = global_min(all_products, vector<product>(), weight_max);
    }

    print_vector(bag1);

    return 0;
}
