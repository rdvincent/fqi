class Policy {
public:
  virtual int getAction(vector<double> s) const = 0;
};

class EGreedy: public Policy {
  double epsilon;
  Regressor *regressor;
public:
  EGreedy(Regressor *regressor, double epsilon = 0.15) {
    this->regressor = regressor;
    this->epsilon = epsilon;
  }

  int getAction(vector<double> s) const {
    if (rnd1() < epsilon) {
      return rand() % regressor->numActions;
    }
    else {
      return regressor->bestAction(s);
    }
  }
};
