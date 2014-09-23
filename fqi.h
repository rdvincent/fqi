/*! Fitted Q iteration */
class FQI {
public:
  Domain *domain;
  Policy *explorePolicy;
  Policy *exploitPolicy;
  Regressor *regressor;
  double gamma;
  int maxIterations;
  int maxRounds;
  int nSubjects;
  vector<tuple> *tuples;
  int numActions;
  bool dynamicStopping;
  bool forestPerAction;

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
  }

  ~FQI() {
    delete tuples;
    delete explorePolicy;
    delete exploitPolicy;
  }

  /* Generate a new set of tuples for the FQI process, using the
   * policy derived from the current Q-function estimate. Also returns
   * the average discounted return of the generated episodes.
   */
  pair<vector<tuple> *,double> generate(int P, int N, Policy *policy) const {
    vector<tuple> *result = new vector<tuple>;
    double rtotal = 0.0;
    result->reserve(P * N);
    for (int p = 0; p < P; p++) {
      double d = 1.0;
      vector<double> s = domain->initialState();
      int n = 0;
      double repisode = 0.0;
      while (n < N && !domain->isTerminal(s)) {
        int a = policy->getAction(s);
        pair<vector<double>, double> pp = domain->performAction(s, a);
        tuple t(s, a, pp.second, pp.first);
        if (domain->isTerminal(pp.first)) {
          cout << "t" << n << endl;
        }
        result->push_back(t);
        repisode += d * pp.second;
        d *= gamma;
        s = pp.first;
        n += 1;
      }
      cout << repisode << endl;
      rtotal += repisode;
    }
    pair<vector<tuple> *,double> p(result, rtotal / P);
    return p;
  }

  void printTuples(const vector<tuple> &t1) {
    for (int m = 0; m < t1.size(); m++) {
      cout << m << ": " << t1[m].s << " " << t1[m].a << " " << t1[m].r << endl;
    }
  }

  void update(vector<dataset> &ts) const {
    // version for per-action forests
    if (forestPerAction) {
      for (int a = 0; a < numActions; a++) {
        ts[a].data.reserve(tuples->size());
      }
      for (int i = 0; i < tuples->size(); i++) {
        tuple tuple = (*tuples)[i];
        double output = tuple.r + gamma * regressor->bestQvalue(tuple.sp);
        datum d(output, tuple.s);
        ts[tuple.a].data.push_back(d);
      }
    }
    else {
      ts[0].data.reserve(tuples->size());
      for (int i = 0; i < tuples->size(); i++) {
        tuple tuple = (*tuples)[i];
        double output = tuple.r + gamma * regressor->bestQvalue(tuple.sp);
        vector<double> attr = tuple.s;
        attr.push_back(tuple.a);
        datum d(output, attr);
        ts[0].data.push_back(d);
      }
    }
  }

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

        // Build our training set.

        vector<dataset> ts(numActions);
        update(ts);

        // Use the updated training set to build new decision trees.

        bool updateOnly = (n > maxIterations * 3 / 4);
        if (updateOnly)
          cout << "updating trees... " << flush;
        else
          cout << "building trees... " << flush;

        regressor->train(ts, updateOnly);

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

      /* Generate a single episode with the current best exploit policy.
       */
      if (domain->isStochastic()) {
        p = generate(100, domain->numSteps, exploitPolicy);
      }
      else {
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

