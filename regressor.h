/*
 * Generic class for a Q-function regressor.
 */
class TieBreaker {
  int nties;

public:
  TieBreaker() { nties = 1; }
    
  bool tie() {
    nties += 1;
    return rnd1() < (1.0 / nties);
  }
};

class Regressor {
public:
  const int numActions;
  Regressor(int na): numActions(na) {
  }
  virtual int bestAction(const vector<double> &s) const = 0;
  virtual double bestQvalue(const vector<double> &s) const = 0;
  virtual void train(const vector<dataset> &ts, bool updateOnly) = 0;
  virtual double meanQvalue(const vector<dataset> &ts) const = 0;
};

/*
 * Q-function regressor implemented with extremely randomized trees.
 */
class ExtraTreeRegressor: public Regressor {
  vector<ExtraTree> *regressor;

public:
  ExtraTreeRegressor(int na, int nd): Regressor(na) {
    ExtraTree et(nd);
    regressor = new vector<ExtraTree>(na, et);
  }

  ~ExtraTreeRegressor() {
    delete regressor;
  }

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
      for (int j = 0; j < ts[a].size(); j++) {
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

/*
 * Q-function regressor implemented with extremely randomized trees.
 * This version uses a single tree to represent the joint state-action
 * space.
 */
class SingleETRegressor: public Regressor {
  ExtraTree *regressor;
  int nd;

public:
  SingleETRegressor(int na, int nd): Regressor(na) {
    regressor = new ExtraTree(nd+1);
    this->nd = nd;
  }

  ~SingleETRegressor() {
    delete regressor;
  }

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

  double meanQvalue(const vector<dataset> &ts) const {
    double q = 0.0;
    int n = 0;
    for (int j = 0; j < ts[0].size(); j++) {
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
