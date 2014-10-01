/**
 * \file
 * \brief Defines an RL policy, a simple mapping from state to action.
 *
 * Copyright (c) 2008-2014 Robert D. Vincent.
 */
/**
 * Abstract class that represents a policy, which is simply a
 * mapping from a state \c s to an action.
 */
class Policy {
public:
  virtual ~Policy() {}

  /**
   * Select the appropriate action for this state.
   * \param s The state.
   */
  virtual int getAction(vector<double> s) const = 0;
};

/**
 * The class that represents the a common kind of stochastic policy
 * with discrete actions. With probability 1.0-epsilon, an e-greedy
 * policy selects the action that maximizes the current estimated
 * Q-function. Otherwise, the policy selects any one of the actions
 * with uniform probability.
 */
class EGreedy: public Policy {
  double epsilon;               /**< Probability of random action. */
  Regressor *regressor;         /**< The regressor in use. */

public:
  /**
   * Construct an e-greedy policy.
   * \param regressor The regression function that represents the Q-function.
   * \param epsilon The probability of a random action.
   */
  EGreedy(Regressor *regressor, double epsilon = 0.15) {
    this->regressor = regressor;
    this->epsilon = epsilon;
  }

  /**
   * Get an action for the e-greedy policy.
   * \param s The state for which an action is desired.
   */
  int getAction(vector<double> s) const {
    if (rnd1() < epsilon) {
      return rand() % regressor->numActions;
    }
    else {
      return regressor->bestAction(s);
    }
  }
};
