/** \file
 * \brief Defines the interface to a reinforcement learning algorithm.
 *
 * Copyright (c) 2008-2014 Robert D. Vincent.
 */
/** The structure that represents the result of taking one time step
 * in the domain.
 */
class OneStepResult {
 public:
  const vector<double> state;         /*!< The successor state. */
  const double reward;                /*!< The reward observed. */

  /** The constructor for the class.
   */
 OneStepResult(const vector<double> &s, double r): state(s), reward(r)
  {
  }
};

/** Abstract base class for a reinforcement learning environment.
 */
class Domain {
public:
  int numSteps;                 /*!< Maximum number of time steps allowed. */
  int numActions;               /*!< Number of possible actions. */
  int numDimensions;            /*!< Dimensionality of the state space. */

  /**
   * Given an action and a state, perform the action and transition to
   * the successor state.
   * \param s The current state.
   * \param a The chosen action.
   * \return A pair consisting of the new state and the reward.
   */
  virtual OneStepResult performAction(vector<double> s, int a) = 0;

  /**
   * Return the reward for this state and action.
   * \param s The state vector.
   * \param a The action.
   * \return The reward received.
   */
  virtual double getReward(vector<double> s, int a) const = 0;

  /**
   * Get the initial state of the domain. This could be constant, or it
   * could be a random state.
   * \return An initial state.
   */
  virtual vector<double> initialState() = 0;

  /**
   * Return a boolean that indicates whether the state is "terminal".
   * \param s The state to evaluate.
   * \return True if the state ends a trajectory.
   */
  virtual bool isTerminal(vector<double> s) const { return false; }

  /**
   * Returns a boolean that indicates whether the state transition function
   * of the domain is stochastic.
   */
  virtual bool isStochastic() { return false; }
};

extern Domain *CreateDomain(const char *dname, const char *propfile);
