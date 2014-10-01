/**
 * \file
 * \brief The extremely randomized (Extra) trees algorithm of Geurts et al. (2006).
 *
 * Copyright (c) 2008-2014 Robert D. Vincent.
 */
/**
 * Our trivial decision tree test class. This implements a simple
 * decision stump. We choose the index of one of the features in the
 * state vector, and a cutoff value.
 */
class dttest {
public:
  int index;            /**< Feature (or dimension) index to test.  */
  double value;         /**< Cutoff value. */

  /** Create a decision tree test */
  dttest(int i = -1, double v = 0.0) {
    index = i;
    value = v;
  }
};

/**
 * Very basic binary decision tree class.
 * In a weak attempt to use space efficiently, we represent a leaf node
 * by setting the index of the dttest to -1 and store the leaf value in the
 * dttest's \c value member.
 */
class decisiontree {
public:
  class dttest test;            /**< The test to perform at this node. */
  decisiontree *left;           /**< Left subtree. */
  decisiontree *right;          /**< Right subtree. */

  /** Construct a leaf node.
   * \param value The value to return at this leaf.
   */
  decisiontree(double value = 0.0): test(-1, value) {
    left = NULL;
    right = NULL;
  }

  /** Construct a compound node.
   * \param test The test to perform at this node.
   * \param left The left subtree.
   * \param right The right subtree.
   */
  decisiontree(const dttest &test, decisiontree *left, decisiontree *right) {
    this->test = test;
    this->left = left;
    this->right = right;
  }

  /**
   * Copy constructor for a decisiontree.
   */
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

  /**
   * Destructor for the decision tree.
   */
  ~decisiontree() {
    if (!isleaf()) {
      delete left;
      delete right;
    }
  }

  /**
   * Verify whether this node is a leaf.
   * \return True if this is a leaf node.
   */
  bool isleaf() const {
    return left == NULL;
  }

  /**
   * Calculate the output value for a given input vector, by descending
   * the tree and returning the value of the leaf node.
   * \param data The input data point to evaluate.
   * \return The value associated with this input data point.
   */
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

/**
 * Simple class used to group the parameters used by the Extremely
 * Randomized (Extra) Tree algorithm.
 */
class ExtraTreeParameters {
public:
  size_t K;                /**< Number of candidate tests per node. */
  size_t nmin;             /**< Minimum splittable node size. */
  size_t M;                /**< Number of trees per forest. */
  size_t MAXDEPTH;         /**< Maximum allowable tree depth. */

  /**
   * Constructor.
   * \param _K Initial value of \c K.
   * \param _M Initial value of \c M.
   * \param _nmin Initial value of \c nmin.
   */
  ExtraTreeParameters(size_t _K = 10, size_t _M = 50, size_t _nmin = 2) {
    K = _K;
    M = _M;
    nmin = _nmin;
    MAXDEPTH = 120;
  }
};

/**
 * Implements the Extremely Randomized (Extra) Trees algorithm of Geurts et al. (2006).
 */
class ExtraTree {
protected:
  int nd;                       /**< Number of dimensions. */
  vector<decisiontree *> forest; /**< The forest, a vector of trees */
  ExtraTreeParameters p;         /**< Our parameters. */

  /**
   * Calculate the value the tree should return at a leaf.
   * This needs to be overridden in derived classes, such as for
   * classification.
   * \param ts The dataset to evaluate.
   * \return The value to store in the leaf node.
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
   *
   * \param ts The training dataset to evaluate.
   * \return True if the dataset is effectively constant.
   */
  static bool isConstant(const dataset &ts) {
    if (ts.size() == 0)
      return true;

    bool result = true;
    double ref_output = ts.data[0].output;
    for (size_t i = 1; i < ts.size(); i++) {
      if (ref_output != ts.data[i].output) {
        result = false;
      }
    }

    if (result)
      return true;

    result = true;
    vector<double> ref_attr = ts.data[0].attributes;
    for (size_t i = 1; i < ts.size(); i++) {
      if (ref_attr != ts.data[i].attributes) {
        result = false;
      }
    }
    return (result);
  }

  /**
   * Update a tree by creating a new copy of a tree with updated values
   * at the leaves. The structure of the tree, and the tests, are unchanged.
   * \param ts The training dataset to use.
   * \param dt The original decision tree.
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

  /**
   * Build a decision tree using the extremely randomized (Extra) tree
   * algorithm.
   * \param ts The training dataset to use.
   * \param depth The current tree depth.
   * \return A decision tree.
   */
  decisiontree *buildTree(const dataset &ts, size_t depth) {
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

  /**
   * Split a dataset into two. This is static because, like other functions
   * parked in this class, it is a part of the overall algorithm, but does
   * not actually depend on any property of the tree itself.
   * \param ts The training dataset to split.
   * \param test The test on which to split the dataset.
   * \param ls The resulting dataset for the left subtree.
   * \param rs The resulting dataset for the right subtree.
   */
  static void split(const dataset &ts, const dttest &test, dataset &ls, dataset &rs) {
    ls.data.reserve(ts.size());
    rs.data.reserve(ts.size());
    for (size_t i = 0; i < ts.size(); i++) {
      if (ts.data[i].attributes[test.index] < test.value) {
        ls.data.push_back(ts.data[i]);
      }
      else {
        rs.data.push_back(ts.data[i]);
      }
    }
  }

  /**
   * Calculate the value of splitting a dataset by a particular test,
   * while avoiding the overhead of actually partitioning the dataset.
   * This is a key optimization for this algorithm.
   * \param ts The training dataset.
   * \param test The test on which to evaluate the split.
   * \param lsize The resulting size of the left-hand dataset.
   * \param lvar The resulting output variance of the left-hand dataset.
   * \param rsize The resulting size of the right-hand dataset.
   * \param rvar The resulting output variance of the right-hand dataset.
   */
  static void trySplit(const dataset &ts, const dttest &test, int &lsize, double &lvar, int &rsize, double &rvar) {
    lsize = 0;
    rsize = 0;

    double lmean = 0.0;
    double lm2 = 0.0;
    double rmean = 0.0;
    double rm2 = 0.0;

    for (size_t i = 0; i < ts.size(); i++) {
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

  /**
   * Calculate the score for a given test.
   * This could be static if it didn't need to be virtual...
   *
   * \param ts The training dataset.
   * \param test The test to evaluate.
   * \param variance The output variance of the entire dataset.
   * \return A score for this test applied to this dataset.
   */
  virtual double scoreTest(const dataset &ts, const dttest &test, double variance) const {
    int lsize, rsize;
    double lvar = 0.0, rvar = 0.0;

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

  /**
   * Implement the Extra tree test generation and selection algorithm.
   * \param ts The training dataset.
   * \return The best test found using the current parameters.
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
    for (size_t i = 0; i < p.K; i++) {
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
  /**
   * Construct a regression tree ensemble that will use the Extra
   * trees algorithm.
   *
   * \param K Number of candidate tests per node.
   * \param M Number of trees per forest.
   * \param nmin Minimum splittable node size.
   */
  ExtraTree(size_t K, size_t M = 50, size_t nmin = 2): p(K, M, nmin) {
    forest.resize(M);
    for (size_t i = 0; i < M; i++) {
      forest[i] = new decisiontree(0.0);
    }
  }

  /**
   * Copy constructor.
   * \param et The ExtraTree to copy.
   */
  ExtraTree(const ExtraTree &et): p(et.p.K, et.p.M, et.p.nmin) {
    nd = et.nd;
    forest.resize(et.forest.size());
    for (size_t i = 0; i < et.forest.size(); i++) {
      forest[i] = new decisiontree(*et.forest[i]);
    }
  }

  /**
   * Destructor.
   */
  virtual ~ExtraTree() {
    for (size_t i = 0; i < forest.size(); i++) {
      delete forest[i];
    }
  }
    

  /**
   * Train an Extremely Randomized tree algorithm for regression.
   * \param ts The training dataset.
   * \param doUpdate True if we should update only (i.e. preserve tree
   * structure).
   */
  void train(dataset ts, bool doUpdate = false) {
    nd = ts.nd();

    for (size_t i = 0; i < p.M; i++) {
      decisiontree *prev_dt = forest[i];
      forest[i] = (doUpdate) ? updateTree(ts, prev_dt) : buildTree(ts, 0);
      delete prev_dt;
    }
  }

  /**
   * Calculate the output value for this tree ensemble.
   * \param data The data point to evaluate.
   * \return The calculated output value.
   */
  virtual double output(const vector<double> &data) const {
    double s = 0.0;
    for (size_t i = 0; i < p.M; i++) {
      s += forest[i]->output(data);
    }
    return s / p.M;
  }

};

/**
 * Use extremely randomized (extra) trees for classification instead of
 * regression.
 * The principal difference is the calculation of test scores and the
 * output values, otherwise the algorithm is largely unchanged.
 */
class ExtraTreeClassification: public ExtraTree {
private:
  /**
   * Simple un-normalized calculation of Shannon entropy.
   * \param p An array of real numbers representing a discrete
   * probability distribution
   * \return The entropy (0 ... log2(p.length))
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
   * dataset. See Geurts et al 2006, Quinlan 1986.
   * \param ts A dataset for which we'd like to calculate the entropy.
   * \return The entropy.
   */
  double H_C(const dataset &ts) const {
    size_t c = ts.size();
    int n = 0;
    for (size_t i = 0; i < c; i++) {
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
   * \param lsize Size of left-hand DataSet.
   * \param rsize Size of right-hand DataSet.
   */
  double H_S(int lsize, int rsize) const {
    double c = lsize + rsize;
    double p[2] = { lsize / c, rsize / c };
    return entropy(p, 2);
  }

  /**
   * Calculates the average conditional entropy of the labels of
   * this split of a dataset. This is used to calculate the
   * information gain of the split outcome and classification.
   *
   * \param lsize Size of left-hand dataset.
   * \param lentropy Entropy of the left-hand dataset.
   * \param rsize Size of right-hand dataset.
   * \param rentropy Entropy of the right-hand dataset.
   */
  double H_CS(int lsize, double lentropy, int rsize, double rentropy) const {
    double c = (lsize + rsize);
    return (lsize / c) * lentropy + (rsize / c) * rentropy;
  }

  /**
   * Calculate the value of splitting a dataset by a particular test,
   * while avoiding the overhead of actually partitioning the dataset.
   * \param ts The dataset to partition.
   * \param test The dttest to apply.
   * \param lsize The size of the left subtree.
   * \param lentropy The classification entropy of the left subtree.
   * \param rsize The size of the right subtree.
   * \param rentropy The classification entropy of the right subtree.
   */
  void trySplit(const dataset &ts, const dttest &test, int &lsize, double &lentropy, int &rsize, double &rentropy) const {
    lsize = 0;
    rsize = 0;
    int lpos = 0;
    int rpos = 0;

    for (size_t i = 0; i < ts.size(); i++) {
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
   * For classification trees, the leaf value should be the majority class
   * of the dataset at the leaf.
   * \param ts The dataset to evaluate.
   * \return The value to store in the leaf node.
   */
  double leafValue(const dataset &ts) const {
    return ts.outputMode();
  }

  /**
   * For classification, the test score is based on a measure of the
   * information gain of the classification.
   *
   * \param ts The training dataset.
   * \param test The test to evaluate.
   * \param variance The output variance of the entire dataset.
   * \return A score for this test applied to this dataset.
   */
  double scoreTest(const dataset &ts, const dttest & test, double variance) const {
    int lsize, rsize;
    double lentropy, rentropy;

    trySplit(ts, test, lsize, lentropy, rsize, rentropy);

    /* Calculate the information gain measure. */
    double thc = H_C(ts);
    return 2.0 * (thc - H_CS(lsize, lentropy, rsize, rentropy)) / (H_S(lsize, rsize) + thc);
  }

public:
  /**
   * The constructor for the class.
   */
  ExtraTreeClassification(size_t K, size_t M = 50, size_t nmin = 2)
    : ExtraTree(K, M, nmin) {
  }

  /**
   * Calculate the output (classification prediction) of the
   * forest. This is the right version for the 2-class case.
   * \param data The output values of each tree in the forest.
   */
  virtual double output(const vector<double> & data) const {
    /* Classify according to the majority.
     */
    int x = 0;
    for (size_t i = 0; i < p.M; i++) {
      if (forest[i]->output(data) > 0) {
        x++;
      }
    }
    return (x > p.M / 2.0) ? 1.0 : -1.0;
  }
};
