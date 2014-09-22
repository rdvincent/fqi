#include <vector>
#include <iterator>
#include <algorithm>
#include <values.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <utility>
using namespace std;

ostream & operator <<(ostream &os, vector<double> d) {
  os << "V(";
  for (int i = 0; i < d.size(); i++) {
    if (i > 0) os << ",";
    os << d[i];
  }
  return os << ")";
}
int main() {
  vector<double> d;
  d.push_back(1.0);
  d.push_back(-1.0);
  d.push_back(2.0);
  cout << d << endl;
}
