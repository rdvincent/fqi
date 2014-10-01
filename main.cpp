/** \file
 * \brief Main program and testing harness for the FQI and ExtraTree
 * algorithms.
 *
 * Copyright (c) 2008-2014 Robert D. Vincent.
 */
#include <vector>
#include <iterator>
#include <algorithm>
#include <values.h>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <assert.h>
#include <utility>
#include <cmath>
using namespace std;

/**
 * Helper function to print a vector of doubles to a stream.
 */
ostream & operator <<(ostream &os, const vector<double> &d)
{
  os << "V(";
  for (size_t i = 0; i < d.size(); i++) {
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

/**
 * Calculate the mean squared error of the output for a data set,
 * given a tree. Used for testing.
 * \param ts The dataset to test against.
 * \param rf The ExtraTree to test.
 * \return The average mean squared error over the entire \c dataset.
 */
double mse(const dataset &ts, const ExtraTree &rf)
{
  double mse = 0.0;
  for (size_t i = 0; i < ts.size(); i++) {
    double tmp = rf.output(ts.data[i].attributes) - ts.data[i].output;
    mse += tmp * tmp;
  }
  return mse / ts.size();
}

/**
 * Read the Parkinson's disease classification data (Little et al. 2007)
 * from the UCI repository. Used for testing.
 * \param ts The returned dataset.
 */
void readparkinsons(dataset &ts) {
  ifstream src("testing/parkinsons.data");
  string line;

  while (getline(src, line)) {
    double tmp[24];
    int i = 0;
    size_t p1 = 0;
    size_t p2 = 0;
    while ((p2 = line.find(',', p1)) != string::npos) {
      tmp[i++] = strtod(line.substr(p1, p2 - p1).c_str(), NULL);
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

/**
 * Read WDBC (Wisconsin diagnostic breast cancer) data (classification).
 * Used for testing.
 * \param ts The returned dataset.
 */
void readwdbc(dataset &ts) {
  ifstream src("testing/wdbc.data");
  string line;

  while (getline(src, line)) {
    double tmp[100];
    int i = 0;
    int n = 0;
    size_t p1 = 0;
    size_t p2 = 0;
    while ((p2 = line.find(',', p1)) != string::npos) {
      string subs = line.substr(p1, p2 - p1);

      /* index 0 is ignored, index 1 is either 'M' or 'B', the rest are
       * features.
       */
      if (n == 1) {
        tmp[i++] = (subs.compare("M") == 0) ? 1.0 : -1.0;
      }
      else if (n >= 2) {
        tmp[i++] = strtod(subs.c_str(), NULL);
      }
      n++;
      p1 = p2 + 1;
    }
    tmp[i++] = strtod(line.substr(p1).c_str(), NULL);
    datum d;
    d.output =  tmp[0];
    for (int j = 1; j < i; j++) {
      d.attributes.push_back(tmp[j]);
    }
    ts.data.push_back(d);
  }
}

/**
 * Read the yacht hydrodynamics data (regression). Used for testing.
 * \param ts The returned dataset.
 */
void readhydro(dataset &ts) {
  ifstream src("testing/yacht_hydrodynamics.data");
  string line;

  while (getline(src, line)) {
    double tmp[7];
    int i = 0;
    size_t p1 = 0;
    size_t p2 = 0;
    while ((p2 = line.find(' ', p1)) != string::npos) {
      tmp[i++] = strtod(line.substr(p1, p2 - p1).c_str(), NULL);
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

/**
 * Calculate regression results for an ExtraTree. Generates a random fold
 * of the dataset \c ts with \c ntrain elements and a test set containing the
 * remainder of the elements. Used for testing.
 *
 * \param ts The dataset to test against.
 * \param ntrain The number of training examples to use.
 * \param rf The ExtraTree to test.
 * \return The average mean-square error of the regression.
 */
double checkRegression(const dataset &ts, int ntrain, ExtraTree &rf) {
  dataset trainset, testset;
  ts.randomFold(ntrain, trainset, testset);

  rf.train(trainset, false);

  return mse(testset, rf);     // Calculate mean squared error
}

/**
 * Perform an n-fold test using a particular \c dataset. Used for testing.
 * \param nfolds The number of test folds.
 * \param ts The training \c dataset.
 * \return The average mean-squared error over all of the folds.
 */
double testRegression(int nfolds, const dataset &ts) {
  int ntrain = ts.size() * (nfolds - 1) / nfolds;
  double sum = 0.0;

  for (int i = 0; i < nfolds; i++) {
    cout << "Fold: " << i+1 << " ";
    ExtraTree rf(ts.nd(), 100, 5);
    double mse = checkRegression(ts, ntrain, rf);
    cout << "MSE: " << mse << endl;
    sum += mse;
  }
  return (sum / nfolds);
}

/**
 * Represents a standard binary confusion matrix.
 */
class ConfusionMatrix {
private:
  int tp;                       /**< Number of true positives */
  int tn;                       /**< Number of true negatives */
  int fp;                       /**< Number of false positives */
  int fn;                       /**< Number of false negatives */

public:
  /** Constructor for a confusion matrix.
   *
   * \param i_tp Initial number of true positives.
   * \param i_tn Initial number of true negatives.
   * \param i_fp Initial number of false positives.
   * \param i_fn Initial number of false negatives.
   */
  ConfusionMatrix(int i_tp = 0, int i_tn = 0, int i_fp = 0, int i_fn = 0) {
    tp = i_tp;
    tn = i_tn;
    fp = i_fp;
    fn = i_fn;
  }

  /**
   * Add two confusion matrices elementwise.
   * \param cm The right-hand operand of the addition.
   */
  ConfusionMatrix operator +(const ConfusionMatrix &cm) {
    ConfusionMatrix r(tp + cm.tp,
                      tn + cm.tn,
                      fp + cm.fp,
                      fn + cm.fn);
    return r;
  }

  /**
   * Calculate the total number of results.
   */
  int total() const { return (tp + tn + fp + fn); }

  /**
   * Calculate the overall accuracy, which is the percentage of
   * correct results overall.
   */
  double accuracy() const { return (tp + tn) / (double) total(); }

  /**
   * Calculate the overall precision, or the ratio of true positives to
   * all positives.
   */
  double precision() const { return (double) tp / (tp + fp); }

  /**
   * Calculate the recall, or the ratio of true positives to
   * true positives and false negatives.
   */
  double recall() { return (double) tp / (tp + fn); }

  /**
   * Calculate the specificity, or the ratio of true negatives to
   * true negatives plus false positives.
   */
  double specificity() { return (double) tn / (tn + fp); }

  /**
   * Record the result of a classification event.
   * \param nPred The predicted label.
   * \param nTrue The actual label.
   */
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

/**
 * Print a ConfusionMatrix in a human-readable manner.
 * \param os The output stream.
 * \param d The ConfusionMatrix to print.
 */
ostream & operator <<(ostream &os, const ConfusionMatrix &d) {
  return os << "tp " << d.tp << " tn " << d.tn << " fp " << d.fp << " fn " << d.fn;
}

/**
 * Calculate classification results.
 * \param ts The entire training set.
 * \param ntrain The number of training examples to use.
 * \param rf The ExtraTree to use.
 * \return A summary of the classification results.
 */
ConfusionMatrix checkClassification(const dataset &ts, int ntrain, ExtraTree &rf) {
  dataset trainset, testset;
  ts.randomFold(ntrain, trainset, testset);
  ConfusionMatrix cm;

  rf.train(trainset, false);

  for (size_t i = 0; i < testset.size(); i++) {
    datum item = testset.data[i];
    double pred = rf.output(item.attributes);
    cm.record(pred, item.output);
  }
  return cm;
}

/**
 * Perform n-fold validation with a \c dataset.
 * \param nfolds The number of cross-validation folds to perform.
 * \param ts The training dataset.
 * \return The overall accuracy.
 */
double testClassification(int nfolds, const dataset &ts) {
  int ntrain = ts.size() * (nfolds - 1) / nfolds;
  ConfusionMatrix cmTotal;
  cout << "Performing " << nfolds << " folds with ";
  cout << ntrain << " training examples out of " << ts.size() << "." << endl;

  for (int i = 0; i < nfolds; i++) {
    cout << "Fold: " << i+1 << " ";
    ExtraTreeClassification rf(ts.nd(), 51, 2);
    ConfusionMatrix cm = checkClassification(ts, ntrain, rf);
    cout << "Accuracy: " << cm.accuracy() << endl;
    cmTotal = cmTotal + cm;
  }
  cout << "Totals: " << cmTotal << endl;
  cout << "Accuracy: " << setprecision(5) << cmTotal.accuracy() << endl;
  return cmTotal.accuracy();
}

#include "getopt.h"

/**
 * Our main program. Performs simple command-line processing and sets
 * up for either testing the ExtraTree implementations or running the FQI
 * algorithm on the selected domain.
 */
int main(int argc, char **argv) {
  const char *domain = "mc";

  Domain *pd;
  Regressor *pr;
  FQI *fqi;
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
    dataset ts1, ts2, ts3;

    cout << "*** Testing classification with parkinsons.data." << endl;
    readparkinsons(ts1);

    double acc = testClassification(40, ts1);
    /* Original paper reports 91.8% mean accuracy */
    assert(acc > 0.918);

    cout << "*** Testing classification with wdbc.data." << endl;
    readwdbc(ts2);
    assert(testClassification(20, ts2) > 0.94);

    cout << "*** Testing regression with yacht_hydrodynamics.data." << endl;
    readhydro(ts3);
    double mse = testRegression(20, ts3);
    assert(mse < 2.0);
    cout << "MSE: " << setprecision(5) << mse << endl;
    assert(mse < 1.1);

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
