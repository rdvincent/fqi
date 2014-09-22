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

/* Use extremely randomized (extra) trees for classification instead of
 * regression.
 * The principal difference is the calculation of test scores and the
 * output values, otherwise the algorithm is largely unchanged.
 */
class ExtraTreeClassification: public ExtraTree {
private:
  /**
   * Simple un-normalized calculation of Shannon entropy.
   * @param p An array of real numbers representing a discrete
   * probability distribution
   * @return The entropy (0 ... log2(p.length))
   */
  double entropy(double p[], int n) const {
    double s = 0;
    for (int i = 0; i < n; i++) {
      if (p[i] > 0.0) {
        s += p[i] * log2(p[i]);
      }
    }
    return -s;
  }

  /**
   * Calculates the entropy of the classification of the
   * DataSet. See Geurts et al 2006, Quinlan 1986.
   * @param ts A [[DataSet]] for which we'd like to calculate the entropy.
   */
  double H_C(const dataset &ts) const {
    int c = ts.size();
    int n = 0;
    for (int i = 0; i < c; i++) {
      if (ts.data[i].output > 0) {
        n++;
      }
    }
    double p[2] = { (double)n / c, (double)(c - n) / c };
    return entropy(p, 2);
  }

  /**
   * Calculates the entropy of this particular split of a
   * DataSet. See Geurts et al 2006, Quinlan 1986.
   * @param ls Left-hand DataSet.
   * @param rs Right-hand DataSet.
   */
  double H_S(int lsize, int rsize) const {
    double c = lsize + rsize;
    double p[2] = { lsize / c, rsize / c };
    return entropy(p, 2);
  }

  /**
   * Calculates the average conditional entropy of the labels of
   * this split of a DataSet. This is used to calculate the 
   * information gain of the split outcome and classification.
   * @param ls Left-hand DataSet.
   * @param rs Right-hand DataSet.
   */
  double H_CS(int lsize, double lentropy, int rsize, double rentropy) const {
    double c = (lsize + rsize);
    return (lsize / c) * lentropy + (rsize / c) * rentropy;
  }

  /*
   * Calculate the value of splitting a dataset by a particular test,
   * while avoiding the overhead of actually partitioning the dataset.
   */
  void trySplit(const dataset &ts, const dttest &test, int &lsize, double &lentropy, int &rsize, double &rentropy) const {
    lsize = 0;
    rsize = 0;
    int lpos = 0;
    int rpos = 0;

    for (int i = 0; i < ts.size(); i++) {
      double x = ts.data[i].output;
      if (ts.data[i].attributes[test.index] < test.value) {
        lsize++;
        if (x > 0.0) {
          lpos++;
        }
      }
      else {
        rsize++;
        if (x > 0.0) {
          rpos++;
        }
      }
    }

    if (lsize > 0) {
      double p[2] = { (double) lpos / lsize, (double)(lsize - lpos) / lsize };
      lentropy = entropy(p, 2);
    }
    else {
      lentropy = 0.0;
    }

    if (rsize > 0) {
      double p[2] = { (double) rpos / rsize, (double)(rsize - rpos) / rsize };
      rentropy = entropy(p, 2);
    }
    else {
      rentropy = 0.0;
    }
  }

protected:
  /**
   * Calculate the value the tree should return at a leaf.
   */
  double leafValue(const dataset &ts) const {
    return ts.outputMode();
  }

  double scoreTest(const dataset &ts, const dttest & test, double variance) const {
    int lsize, rsize;
    double lentropy, rentropy;

    trySplit(ts, test, lsize, lentropy, rsize, rentropy);

    /* Calculate the information gain measure. */
    double thc = H_C(ts);
    return 2.0 * (thc - H_CS(lsize, lentropy, rsize, rentropy)) / (H_S(lsize, rsize) + thc);
  }

public:
  ExtraTreeClassification(int K, int M = 50, int nmin = 2)
    : ExtraTree(K, M, nmin) {
  }

  /**
   * Calculate the output (classification prediction) of the
   * forest. This is the right version for the 2-class case.
   * @param instance The output values of each tree in the forest.
   */
  virtual double output(const vector<double> & data) const {
    /* Classify according to the majority.
     */
    int x = 0;
    for (int i = 0; i < p.M; i++) {
      if (forest[i]->output(data) > 0) {
        x++;
      }
    }
    return (x > p.M / 2.0) ? 1.0 : -1.0;
  }
};

