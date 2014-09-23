#include <vector>
#include <iterator>
#include <algorithm>
#include <values.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <utility>
#include <cmath>
using namespace std;

ostream & operator <<(ostream &os, const vector<double> &d)
{
  os << "V(";
  for (int i = 0; i < d.size(); i++) {
    if (i > 0) os << ",";
    os << d[i];
  }
  return os << ")";
}

#include "dataset.h"
#include "tuple.h"
#include "random.h"
#include "extra.h"
#include "regressor.h"
#include "policy.h"
#include "domain.h"
#include "fqi.h"

/** Calculate the mean squared error of the output for a data set,
 * given a tree.
 */
double mse(const dataset &ts, const ExtraTree &rf) 
{
  double mse;
  for (int i = 0; i < ts.size(); i++) {
    double tmp = rf.output(ts.data[i].attributes) - ts.data[i].output;
    mse += tmp * tmp;
  }
  return mse / ts.size();
}

/* Read the Parkinson's disease data (classification)
 */
void readparkinsons(dataset &ts) {
  ifstream src("parkinsons.data");
  string line;

  while (getline(src, line)) {
    double tmp[24];
    int i = 0;
    int p1 = 0;
    int p2 = 0;
    while ((p2 = line.find(',', p1)) != string::npos) {
      tmp[i++] = strtod(line.substr(p1, p2).c_str(), NULL);
      p1 = p2 + 1;
    }
    tmp[i++] = strtod(line.substr(p1).c_str(), NULL);
    datum d;
    d.output = tmp[17] > 0.0 ? 1.0 : -1.0;
    for (int j = 1; j < 17; j++) {
      d.attributes.push_back(tmp[j]);
    }
    for (int j = 18; j < 24; j++) {
      d.attributes.push_back(tmp[j]);
    }
    ts.data.push_back(d);
  }
}

/* Read the yacht hydrodynamics data (regression)
 */
void readhydro(dataset &ts) {
  ifstream src("yacht_hydrodynamics.data");
  string line;

  while (getline(src, line)) {
    double tmp[7];
    int i = 0;
    int p1 = 0;
    int p2 = 0;
    while ((p2 = line.find(' ', p1)) != string::npos) {
      tmp[i++] = strtod(line.substr(p1, p2).c_str(), NULL);
      p1 = p2 + 1;
    }
    tmp[i++] = strtod(line.substr(p1).c_str(), NULL);
    datum d;
    d.output = tmp[6];
    for (int j = 0; j < 6; j++) {
      d.attributes.push_back(tmp[j]);
    }
    ts.data.push_back(d);
  }
}

double checkRegression(const dataset &ts, int ntrain, ExtraTree &rf) {
  dataset trainset, testset;
  ts.randomFold(ntrain, trainset, testset);

  rf.train(trainset, false);

  return mse(testset, rf);     // Calculate mean squared error
}

void testRegression(int nfolds, const dataset &ts) {
  int ntrain = ts.size() * (nfolds - 1) / nfolds;

  for (int i = 1; i <= nfolds; i++) {
    cout << "Fold " << i;
    ExtraTree rf(ts.nd(), 30, 2);
    double mse = checkRegression(ts, ntrain, rf);
    cout << " mse " << mse << endl;
    assert(mse < 3.5);
  }
}

class ConfusionMatrix {
private:
  int tp, tn, fp, fn;
  
public:
  ConfusionMatrix(int i_tp = 0, int i_tn = 0, int i_fp = 0, int i_fn = 0) {
    tp = i_tp;
    tn = i_tn;
    fp = i_fp;
    fn = i_fn;
  }

  ConfusionMatrix operator +(const ConfusionMatrix &cm) {
    ConfusionMatrix r(tp + cm.tp,
                      tn + cm.tn,
                      fp + cm.fp,
                      fn + cm.fn);
    return r;
  }

  int total() const { return (tp + tn + fp + fn); }
  double accuracy() const { return (tp + tn) / (double) total(); }
  double precision() const { return (double) tp / (tp + fp); }
  double recall() { return (double) tp / (tp + fn); }
  double specificity() { return (double) tn / (tn + fp); }
      
  void record(double nPred, double nTrue) {
    if (nPred == nTrue) {
      if (nPred > 0) {
        tp += 1;
      }
      else {
        tn += 1;
      }
    }
    else {
      if (nPred > 0) {
        fp += 1;
      }
      else {
        fn += 1;
      }
    }
  }
  friend ostream & operator <<(ostream &os, const ConfusionMatrix &d);
};

ostream & operator <<(ostream &os, const ConfusionMatrix &d) {
  return os << "tp " << d.tp << " tn " << d.tn << " fp " << d.fp << " fn " << d.fn;
}

ConfusionMatrix checkClassification(const dataset &ts, int ntrain, ExtraTree &rf) {
  dataset trainset, testset;
  ts.randomFold(ntrain, trainset, testset);
  ConfusionMatrix cm;

  rf.train(trainset, false);

  for (int i = 0; i < testset.size(); i++) {
    datum item = testset.data[i];
    double pred = rf.output(item.attributes);
    cm.record(pred, item.output);
  }
  return cm;
}

void testClassification(int nfolds, const dataset &ts) {
  int ntrain = ts.size() * (nfolds - 1) / nfolds;
  ConfusionMatrix cmTotal;

  for (int i = 1; i <= nfolds; i++) {
    cout << "Fold " << i << " ";
    ExtraTreeClassification rf(ts.nd(), 100, 2);
    ConfusionMatrix cm = checkClassification(ts, ntrain, rf);
    cout << cm << endl;
    cmTotal = cmTotal + cm;
  }

  cout << "Totals: " << cmTotal << endl;
  cout << "Overall precision: " << cmTotal.precision() << endl;
  cout << "Overall recall: " << cmTotal.recall() << endl;
  cout << "Accuracy: " << cmTotal.accuracy() << endl;
}

#include "getopt.h"

int main(int argc, char **argv) {
  const char *domain = "mc";

  Domain *pd;
  Regressor *pr;
  FQI *fqi;
  dataset ts1;
  dataset ts2;
  int c;
  int tflag = 0;
  int sflag = 0;
  while ((c = getopt(argc, argv, "tsd:")) != -1) {
    switch (c) {
    case 't':
      tflag++;
      break;
    case 's':
      sflag++;
      break;
    case 'd':
      domain = optarg;
      break;
    case '?':
      return 1;
    default:
      break;
    }
  }

  if (tflag) {
    cout << "Testing classification with parkinsons.data." << endl;
    readparkinsons(ts1);
    testClassification(10, ts1);

    cout << "Testing regression with yacht_hydrodynamics.data." << endl;
    readhydro(ts2);
    testRegression(10, ts2);
    return 0;
  }
  else {
    pd = CreateDomain(domain);
    if (sflag) {
      pr = new SingleETRegressor(pd->numActions, pd->numDimensions);
    }
    else {
      pr = new ExtraTreeRegressor(pd->numActions, pd->numDimensions);
    }
    fqi = new FQI(pd, pr, 0.98, 400, 10, 30);
    fqi->run();
  }
  cout << "done!" << endl;
}
