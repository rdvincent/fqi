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

#include "fqi.h"

ostream & operator <<(ostream &os, const vector<double> &d)
{
  os << "V(";
  for (int i = 0; i < d.size(); i++) {
    if (i > 0) os << ",";
    os << d[i];
  }
  return os << ")";
}

/* Our trivial "decision tree test" class.
 */
class dttest {
public:
  int index;
  double value;

  dttest(int i = -1, double v = 0.0) {
    index = i;
    value = v;
  }
} ;

/* Very basic binary decision tree class.
 */
class decisiontree {
public:
  class dttest test;
  decisiontree *left;
  decisiontree *right;
  
  decisiontree(double value = 0.0): test(-1, value) {
    left = NULL;
    right = NULL;
  }

  decisiontree(const dttest &test, decisiontree *left, decisiontree *right) {
    this->test = test;
    this->left = left;
    this->right = right;
  }

  decisiontree(const decisiontree &dt) {
    if (dt.isleaf()) {
      test.value = dt.test.value;
      left = right = NULL;
    }
    else {
      test = dt.test;
      left = new decisiontree(*dt.left);
      right = new decisiontree(*dt.right);
    }
  }

  ~decisiontree() {
    if (!isleaf()) {
      delete left;
      delete right;
    }
  }

  bool isleaf() const {
    return left == NULL;
  }

  double output(const vector<double> &data) const {
    if (isleaf()) {
      return test.value;
    }
    else {
      if (data[test.index] < test.value) {
        return left->output(data);
      }
      else {
        return right->output(data);
      }
    }
  }
};

class ExtraTreeParameters {
public:
  int K;
  int nmin;
  int M;
  int MAXDEPTH;
    
  ExtraTreeParameters(int _K = 10, int _M = 50, int _nmin = 2) {
    K = _K;
    M = _M;
    nmin = _nmin;
    MAXDEPTH = 120;
  }
};

class ExtraTree {
protected:
  int nd;
  vector<decisiontree *> forest;
  ExtraTreeParameters p;

  /**
   * Calculate the value the tree should return at a leaf.
   * This needs to be overridden in derived classes, such as for
   * classification.
   */
  virtual double leafValue(const dataset &ts) const {
    return ts.outputMean();
  }

  /**
   * Test whether a particular dataset is effectively constant. This
   * could actually live in the dataset class, as opposed to being
   * here as a static member. Either choice is defensible, as it is
   * part of the Extratree algorithm definition, but it is really a
   * property of the dataset.
   *
   * A dataset is "constant" if it is empty, or if all items have the 
   * same output, or if all items have the same inputs.
   */
  static bool isConstant(const dataset &ts) {
    if (ts.size() == 0)
      return true;

    bool result = true;
    double ref_output = ts.data[0].output;
    for (int i = 1; i < ts.size(); i++) {
      if (ref_output != ts.data[i].output) {
        result = false;
      }
    }

    if (result) 
      return true;

    result = true;
    vector<double> ref_attr = ts.data[0].attributes;
    for (int i = 1; i < ts.size(); i++) {
      if (ref_attr != ts.data[i].attributes) {
        result = false;
      }
    }
    return (result);
  }

  /*
   * Update a tree by creating a new copy of a tree with updated values
   * at the leaves. The structure of the tree, and the tests, are unchanged.
   */
  decisiontree *updateTree(const dataset &ts, const decisiontree *dt) {
    if (dt->isleaf()) {
      return new decisiontree(leafValue(ts));
    }
    else {
      dataset ls, rs;
      split(ts, dt->test, ls, rs);
      return new decisiontree(dt->test, updateTree(ls, dt->left), updateTree(rs, dt->right));
    }
  }

  /*
   * Build up a decision tree using the extremely randomized (extra) tree
   * algorithm.
   */
  decisiontree *buildTree(const dataset &ts, int depth) {
    if (isConstant(ts) || ts.size() < p.nmin || depth > p.MAXDEPTH) {
      if (depth > p.MAXDEPTH) {
        cerr << "Maximum depth exceeded!." << endl;
      }
      return new decisiontree(leafValue(ts));
    }
    else {
      dttest test = findTest(ts);
      dataset ls, rs;

      split(ts, test, ls, rs);
      return new decisiontree(test, buildTree(ls, depth + 1), buildTree(rs, depth + 1));
    }
  }

  /* Split a dataset into two. This is static because, like other functions
   * parked in this class, it is a part of the overall algorithm, but does
   * not actually depend on any property of the tree itself.
   */
  static void split(const dataset &ts, const dttest &test, dataset &ls, dataset &rs) {
    ls.data.reserve(ts.size());
    rs.data.reserve(ts.size());
    for (int i = 0; i < ts.size(); i++) {
      if (ts.data[i].attributes[test.index] < test.value) {
        ls.data.push_back(ts.data[i]);
      }
      else {
        rs.data.push_back(ts.data[i]);
      }
    }
  }

  /*
   * Calculate the value of splitting a dataset by a particular test,
   * while avoiding the overhead of actually partitioning the dataset.
   */
  static void trySplit(const dataset &ts, const dttest &test, int &lsize, double &lvar, int &rsize, double &rvar) {
    lsize = 0;
    rsize = 0;
    
    double lmean = 0.0;
    double lm2 = 0.0;
    double rmean = 0.0;
    double rm2 = 0.0;

    for (int i = 0; i < ts.size(); i++) {
      double x = ts.data[i].output;
      if (ts.data[i].attributes[test.index] < test.value) {
        lsize++;

        // knuth's online variance...
        double delta = x - lmean;
        lmean += delta / lsize;
        lm2 += delta * (x - lmean);
      }
      else {
        rsize++;
        double delta = x - rmean;
        rmean += delta / rsize;
        rm2 += delta * (x - rmean);
      }
    }
    if (lsize > 1) {
      lvar = lm2 / (lsize - 1);
    }
    if (rsize > 1) {
      rvar = rm2 / (rsize - 1);
    }
  }

  /* This could be static if it didn't need to be virtual...
   */
  virtual double scoreTest(const dataset &ts, const dttest &test, double variance) const {
    int lsize, rsize;
    double lvar, rvar;

    trySplit(ts, test, lsize, lvar, rsize, rvar);
    if (lsize == 0 || rsize == 0) {
      return -1.0;
    }
    else {
      double lratio = (double) lsize / ts.size();
      double rratio = (double) rsize / ts.size();

      //cout << lratio << " " << lvar << " " << rratio << " " << rvar << endl;

      return (variance - lratio * lvar - rratio * rvar) / (variance + 0.0001);
    }
  }

  /** Implement the Extra tree test generation and selection algorithm.
   */
  dttest findTest(const dataset &ts) const {
    double variance = ts.outputVariance();

    /* We need to start off by finding the minimum and maximum values
     * for each of the attributes in the current training set.
     */
    double minv[nd];
    double maxv[nd];
    ts.getRanges(maxv, minv);

    /* Create initially empty array of tests.
     */
    vector<dttest> tests;
    tests.reserve(p.K);

    /* Now generate the list of usable features (indicies) and shuffle
     * it to randomize the choice of features.
     */
    vector<int> validIndices;
    validIndices.reserve(nd);
    for (int i = 0; i < nd; i++) {
      if (maxv[i] != minv[i]) {
        validIndices.push_back(i);
      }
    }

    vector<int> shuffledIndices = validIndices;
    random_shuffle(shuffledIndices.begin(), shuffledIndices.end());

    if (shuffledIndices.size() > p.K) {
      shuffledIndices.resize(p.K);
    }

    /* Sample at most K initial tests.
     */
    for (vector<int>::const_iterator it = shuffledIndices.begin();
         it != shuffledIndices.end();
         it++) {
      int index = *it;
      tests.push_back(dttest(index, rndInterval(minv[index], maxv[index])));
    }

    /* Sample additional tests with replacement, if needed.
     */
    while (tests.size() < p.K) {
      int index = rand() % nd;
      if (minv[index] != maxv[index]) {
	tests.push_back(dttest(index, rndInterval(minv[index], maxv[index])));
      }
    }

    /* We now have exactly K tests in the list. Return the
     * one which gives the largest score.
     */
    double max_score = -DBL_MAX;
    int max_index = -1;
    for (int i = 0; i < p.K; i++) {
      double tmp = scoreTest(ts, tests[i], variance);
      if (tmp > max_score) {
        max_score = tmp;
        max_index = i;
      }
    }
    if (max_index == -1) {
      cerr << "OOPS: " << max_score << " " << tests.size() << endl;
    }
    return tests[max_index];
  }

public:

  ExtraTree(int K, int M = 50, int nmin = 2): p(K, M, nmin) {
    forest.resize(M);
    for (int i = 0; i < M; i++) {
      forest[i] = new decisiontree(0.0);
    }
  }

  // copy constructor
  ExtraTree(const ExtraTree &et): p(et.p.K, et.p.M, et.p.nmin) {
    nd = et.nd;
    forest.resize(et.forest.size());
    for (int i = 0; i < et.forest.size(); i++) {
      forest[i] = new decisiontree(*et.forest[i]);
    }
  }

  void train(dataset ts, bool doUpdate = false) {
    nd = ts.nd();

    for (int i = 0; i < p.M; i++) {
      decisiontree *prev_dt = forest[i];
      forest[i] = (doUpdate) ? updateTree(ts, prev_dt) : buildTree(ts, 0);
      delete prev_dt;
    }
  }

  virtual double output(const vector<double> &data) const {
    double s = 0.0;
    for (int i = 0; i < p.M; i++) {
      s += forest[i]->output(data);
    }
    return s / p.M;
  }

};

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

int main(int argc, char **argv) {
  int domain = 1;

  Domain *pd;
  Regressor *pr;
  FQI *fqi;
  dataset ts1;
  dataset ts2;

  switch (domain) {
  default:
  case 0:
    cout << "Testing classification with parkinsons.data." << endl;
    readparkinsons(ts1);
    testClassification(10, ts1);

    cout << "Testing regression with yacht_hydrodynamics.data." << endl;
    readhydro(ts2);
    testRegression(10, ts2);
    return 0;

  case 1:
    pd = new MC();
    //pr = new ExtraTreeRegressor(pd->numActions, pd->numDimensions);
    pr = new SingleETRegressor(pd->numActions, pd->numDimensions);
    fqi = new FQI(pd, pr, 0.98, 400, 10, 10);
    break;
  case 2:
    pd = new Bicycle();
    pr = new SingleETRegressor(pd->numActions, pd->numDimensions);
    fqi = new FQI(pd, pr, 0.98, 400, 10, 10);
    break;
  case 3:
    pd = new HIV();
    //pr = new ExtraTreeRegressor(pd->numActions, pd->numDimensions);
    pr = new SingleETRegressor(pd->numActions, pd->numDimensions);
    fqi = new FQI(pd, pr, 0.98, 400, 10, 30);
    break;
  }
  fqi->run();
  cout << "done!" << endl;
}
