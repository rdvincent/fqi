/**
 * \file
 * \brief Defines the Regressor interface for representing Q-functions
 * in FQI.
 *
 * Copyright (c) 2008-2014 Robert D. Vincent.
 */
/**
 * Utility class for dealing with random tie-breaking in the policy.
 */
class TieBreaker {
  int nties;                    /*!< Number of ties we've broken. */

public:
  TieBreaker() { nties = 1; }

  /**
   * Returns true if we should randomly break the tie at this point.
   */
  bool tie() {
    nties += 1;
    return rnd1() < (1.0 / nties);
  }
};

/**
 * Abstract class for a Q-function regressor.
 */
class Regressor {
public:
  const int numActions;         /*!< Number of actions */

  /**
   * Create an instance of a regressor that will represent the Q-function
   * in the RL environment.
   * \param na The number of actions in the RL environment.
   */
  Regressor(int na): numActions(na) {
  }

  /**
   * Virtual destructor.
   */
  virtual ~Regressor() {}

  /**
   * Select the best action for the given state.
   * \param s The state we want to evaluate.
   */
  virtual int bestAction(const vector<double> &s) const = 0;

  /**
   * Return the best Q-value for the given state.
   * \param s The state we want to evaluate.
   */
  virtual double bestQvalue(const vector<double> &s) const = 0;

  /**
   * Train the regressor.
   * \param ts The training dataset.
   * \param updateOnly If true, freeze the tree structure and tests,
   * updating the leaf values only.
   */
  virtual void train(const vector<dataset> &ts, bool updateOnly) = 0;

  /**
   * Return the mean Q-value for the entire training set. Used for
   * general bookkeeping and early termination calculations.
   * \param ts The training dataset.
   */
  virtual double meanQvalue(const vector<dataset> &ts) const = 0;

  /**
   * Return true if the regressor is "compound", that is, if it uses
   * a separate substructure to represent each discrete action.
   */
  virtual bool compound() const = 0;
};

/**
 * Q-function regressor implemented with extremely randomized trees.
 * This is the compound version that uses a separate forest to represent
 * the Q-values for each of the possible actions.
 */
class ExtraTreeRegressor: public Regressor {
  vector<ExtraTree> *regressor; /*!< A vector of extremely randomized trees. */

public:
  /**
   * Construct an ExtraTreeRegressor.
   * \param na Number of actions.
   * \param nd Number of dimensions.
   */
  ExtraTreeRegressor(int na, int nd): Regressor(na) {
    ExtraTree et(nd);
    regressor = new vector<ExtraTree>(na, et);
  }

  /**
   * Destroy an ExtraTreeRegressor
   */
  virtual ~ExtraTreeRegressor() {
    delete regressor;
  }

  /**
   * \return true
   */
  bool compound() const { return true; }

  int bestAction(const vector<double> &s) const {
    double max_q = -DBL_MAX;
    int max_n = 0;
    TieBreaker tb;

    for (int n = 0; n < numActions; n++) {
      double q = (*regressor)[n].output(s);
      if (q > max_q || (q == max_q && tb.tie())) {
        max_q = q;
        max_n = n;
      }
    }
    return max_n;
  }

  double bestQvalue(const vector<double> &s) const {
    double max_q = -DBL_MAX;
    for (int a = 0; a < numActions; a++) {
      double q = (*regressor)[a].output(s);
      if (q > max_q) {
        max_q = q;
      }
    }
    return max_q;
  }

  double meanQvalue(const vector<dataset> &ts) const {
    double q = 0.0;
    int n = 0;
    for (int a = 0; a < numActions; a++) {
      for (size_t j = 0; j < ts[a].size(); j++) {
        q += (*regressor)[a].output(ts[a].data[j].attributes);
        n += 1;
      }
    }
    return (q / n);
  }

  void train(const vector<dataset> &ts, bool updateOnly) {
    for (int a = 0; a < numActions; a++) {
      (*regressor)[a].train(ts[a], updateOnly);
    }
  }
};

/**
 * Q-function regressor implemented with extremely randomized trees.
 * This version uses a single tree to represent the joint state-action
 * space.
 */
class SingleETRegressor: public Regressor {
  ExtraTree *regressor;         /*!< The single forest. */
  int nd;                       /*!< The number of dimensions. */

public:
  /** Create a SingleETRegressor.
   * \param na Number of actions.
   * \param nd Number of dimensions.
   */
  SingleETRegressor(int na, int nd): Regressor(na) {
    regressor = new ExtraTree(nd+1);
    this->nd = nd;
  }

  virtual ~SingleETRegressor() {
    delete regressor;
  }

  bool compound() const { return false; }

  /**
   * \param s The state to evaluate.
   */
  int bestAction(const vector<double> &s) const {
    double max_q = -DBL_MAX;
    int max_a = 0;
    TieBreaker tb;

    vector<double> v = s;
    v.push_back(0);
    for (int a = 0; a < numActions; a++) {
      v[nd] = a;
      double q = regressor->output(v);
      if (q > max_q || (q == max_q && tb.tie())) {
        max_q = q;
        max_a = a;
      }
    }
    return max_a;
  }

  /**
   * \param s The state to evaluate.
   */
  double bestQvalue(const vector<double> &s) const {
    double max_q = -DBL_MAX;
    vector<double> v = s;
    v.push_back(0);
    for (int a = 0; a < numActions; a++) {
      v[nd] = a;
      double q = regressor->output(v);
      if (q > max_q) {
        max_q = q;
      }
    }
    return max_q;
  }

  /**
   * \param ts The training dataset.
   */
  double meanQvalue(const vector<dataset> &ts) const {
    double q = 0.0;
    int n = 0;
    for (size_t j = 0; j < ts[0].size(); j++) {
      double t = regressor->output(ts[0].data[j].attributes);
      //if (t < -1.0) {
      //cout << t << " " << ts[0].data[j].attributes << " " << ts[0].data[j].output << endl;
      //}
      q += t;
      n += 1;
    }
    return (q / n);
  }

  void train(const vector<dataset> &ts, bool updateOnly) {
    regressor->train(ts[0], updateOnly);
  }

};
