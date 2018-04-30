#include <iostream>
#include <fstream>
#include <string>

using namespace std;

struct flower {
    int id;
    float sepal_length;
    float sepal_width;
    float petal_length;
    float petal_width;
    string species;
};



int main() {
    ifstream iris_data("iris.csv");
    
    if(!iris_data.is_open()) cout << "ERROR: File Open" << '\n';
    
    string id;
    string sepal_length;
    string sepal_width;
    string petal_length;
    string petal_width;
    string species;
    
    int i = 1;
    while(iris_data.good()) {
        getline(iris_data, id, ',');
        getline(iris_data, sepal_length, ',');
        getline(iris_data, sepal_width, ',');
        getline(iris_data, petal_length, ',');
        getline(iris_data, petal_width, ',');
        getline(iris_data, species, '\n');
        
        if(i != 1) {
            cout << stoi(id) << ": " << stof(sepal_length) << ", " << stof(sepal_width) << ", " << stof(petal_length) << ", " << stof(petal_width) << ", " << species << '\n';;
        }
        
        i++;
    }
}