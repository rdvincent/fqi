class Domain {
public:
  int numSteps;
  int numActions;
  int numDimensions;

  /* Given an action and a state, perform the action and transition to
   * the successor state.
   *
   * Returns a pair consisting of the new state and the reward.
   */
  virtual pair<vector<double>, double> performAction(vector<double> s, int a) = 0;

  /* Returns the initial state of the domain. This could be constant, or it 
   * could be a random state.
   */
  virtual vector<double> initialState() = 0;
  /* Returns a boolean that indicates whether the state is "terminal".
   */
  virtual bool isTerminal(vector<double> s) { return false; }

  /* Returns a boolean that indicates whether the state transition function
   * of the domain is stochastic.
   */
  virtual bool isStochastic() { return false; }
};

extern Domain *CreateDomain(const char *);

