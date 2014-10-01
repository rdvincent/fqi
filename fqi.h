/**
 * \file
 * \brief The fitted Q iteration algorithm from Ernst et al. (2005).
 *
 * Copyright (c) 2008-2014 Robert D. Vincent.
 */
/** Fitted Q iteration, following the algorithm described in Ernst et al.
 * (2005) "Tree-Based Batch Mode Reinforcement Learning".
 */
class FQI {
public:
  Domain *domain;               /*!< The RL environment. */
  Policy *explorePolicy;        /*!< An exploration policy. */
  Policy *exploitPolicy;        /*!< An control (exploitation) policy. */
  Regressor *regressor;         /*!< Q-function approximator. */
  double gamma;                 /*!< Discount factor. */
  int maxIterations;            /*!< Maximum number of FQI iterations.  */
  int maxRounds;                /*!< Maximum number of FQI rounds. */
  int nSubjects;                /*!< Number of simulated trajectories. */
  vector<tuple> *tuples;        /*!< Raw training data. */
  int numActions;               /*!< Number of action in the environment. */
  bool dynamicStopping;         /*!< Whether to stop each round dynamically. */
  bool forestPerAction;         /*!< Whether to use one forest per action. */
  bool debug;

  /**
   * Constructor for the fitted Q iteration class.
   * \param domain The reinforcement learning environment.
   * \param regressor The regression algorithm to use.
   * \param gamma The RL discount factor.
   * \param maxIterations The maximum number of FQI iterations per round.
   * \param maxRounds The maximum number of rounds (the trajectories are recomputed between each round).
   * \param nSubjects The number of trajectories ("subjects") added in each round.
   */
  FQI(Domain * domain,
      Regressor *regressor,
      double gamma = 0.98,
      int maxIterations = 60,
      int maxRounds = 10,
      int nSubjects = 30) {
    this->domain = domain;
    tuples = new vector<tuple>();
    explorePolicy = new EGreedy(regressor, 0.15);
    exploitPolicy = new EGreedy(regressor, 0.0);
    numActions = domain->numActions;
    dynamicStopping = true;
    this->regressor = regressor;
    this->gamma = gamma;
    this->maxIterations = maxIterations;
    this->maxRounds = maxRounds;
    this->nSubjects = nSubjects;
    forestPerAction = regressor->compound();
    debug = false;
  }

  /**
   * Class destructor. Just frees some memory.
   */
  ~FQI() {
    delete tuples;
    delete explorePolicy;
    delete exploitPolicy;
  }

  /**
   * Generate a new set of tuples for the FQI process, using the
   * policy derived from the current Q-function estimate. Also returns
   * the average discounted return of the generated episodes.
   */
  pair<vector<tuple> *,double> generate(int P, int N, Policy *policy) const {
    vector<tuple> *trajectory = new vector<tuple>;
    trajectory->reserve(P * N);
    double rtotal = 0.0;
    for (int p = 0; p < P; p++) {
      double d = 1.0;
      vector<double> s = domain->initialState();
      int n = 0;
      double repisode = 0.0;
      while (n < N && !domain->isTerminal(s)) {
        int a = policy->getAction(s);
        OneStepResult pp = domain->performAction(s, a);
        tuple t(s, a, pp.reward, pp.state);
        if (debug) {
          if (domain->isTerminal(pp.state)) {
            cout << "t" << n << endl;
          }
        }
        trajectory->push_back(t);
        repisode += d * pp.reward;
        d *= gamma;
        s = pp.state;
        n += 1;
      }
      if (debug) {
        cout << repisode << endl;
      }
      rtotal += repisode;
    }
    pair<vector<tuple> *,double> p(trajectory, rtotal / P);
    return p;
  }

  /**
   * Prints a tuple vector in a somewhat readable format.
   */
  void printTuples(const vector<tuple> &t1) const {
    for (size_t m = 0; m < t1.size(); m++) {
      cout << m << ": " << t1[m].s << " " << t1[m].a << " " << t1[m].r << endl;
    }
  }

  /**
   * Update the Q-function estimates for the training data tuples, based
   * on the current regressor.
   * This updates the tuples, but does not affect the state of the FQI
   * itself.
   */
  void update(vector<dataset> &ts) {
    /* version for per-action forests
     */
    if (forestPerAction) {
      for (int a = 0; a < numActions; a++) {
        ts[a].data.reserve(tuples->size());
      }
      for (size_t i = 0; i < tuples->size(); i++) {
        tuple tuple = (*tuples)[i];
        double output = tuple.r + gamma * regressor->bestQvalue(tuple.sp);
        datum d(output, tuple.s);
        ts[tuple.a].data.push_back(d);
      }
    }
    /* version for single-forest incorporating the action.
     */
    else {
      ts[0].data.reserve(tuples->size());
      for (size_t i = 0; i < tuples->size(); i++) {
        tuple tuple = (*tuples)[i];
        double output = tuple.r + gamma * regressor->bestQvalue(tuple.sp);
        vector<double> attr = tuple.s;
        attr.push_back(tuple.a);
        datum d(output, attr);
        ts[0].data.push_back(d);
      }
    }
  }

  /**
   * Runs a series of rounds of fitted Q iteration. Before each
   * of \c maxRounds rounds, a total of \c nSubjects trajectories are
   * computed using the current best policy estimate. These samples are
   * added to the entire training set, and then we perform \c maxIterations
   * iterations of the FQI algorithm, in which we re-estimate the best
   * Q-function values for the the training set and re-run the regression on
   * the updated training set.
   */
  void run() {
    double q1 = 0.0;
    double q2 = 0.0;

    for (int r = 0; r < maxRounds; r++) {
      pair<vector<tuple> *, double> p = generate(nSubjects, domain->numSteps, explorePolicy);
      vector<tuple> *t0 = p.first;

      tuples->insert(tuples->begin(), t0->begin(), t0->end());
      delete t0;

      cout << "round " << r << ": " << tuples->size() << " " << p.second << endl;

      for (int n = 0; n < maxIterations; n++) {
        cout << "step " << n << "... " << flush;

        /* Update the training set. */

        vector<dataset> ts(numActions);
        update(ts);

        /* Use the updated training set to train a new regressor. */

        bool updateOnly = (n > maxIterations * 3 / 4);
        if (updateOnly)
          cout << "updating trees... " << flush;
        else
          cout << "building trees... " << flush;

        regressor->train(ts, updateOnly);

        /* If we are doing dynamic stopping, then see if the stopping
         * criterion has been met.
         */
        if (dynamicStopping && !updateOnly) {
          q1 = q2;
          q2 = regressor->meanQvalue(ts);
          double delta = (q1 != 0.0) ? fabs(q1-q2)/fabs(q1) : 1.0;
          cout << q1 << " " << q2 << " " << delta << " ";
          if (delta < 0.005) {
            n = maxIterations - 10;
          }
        }

        cout << "done!" << endl;
      }

      if (domain->isStochastic()) {
        /* If the domain is stochastic, generate 100 episodes with the
         * current best policy.
         */
        p = generate(100, domain->numSteps, exploitPolicy);
      }
      else {
        /* If the domain is deterministic, generate a single episode with
         * the current best policy. We also print the trajectory.
         * \todo Make the trajectory printing optional.
         */
        p = generate(1, domain->numSteps, exploitPolicy);
        /* Print out the entire trajectory.
         */
        printTuples(*p.first);
      }
      delete p.first;
      cout << "Mean return " << p.second << endl;
    }
  }
};
