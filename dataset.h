class datum {
public:
  double output;
  vector<double> attributes;

  datum() {
  }

  datum(double o, vector<double> a) {
    output = o;
    attributes = a;
  }
};

class dataset {
public:
  vector<datum> data;

  int size() const { return data.size(); }

  int nd() const { return data[0].attributes.size(); }

  void randomFold(int ntrain, dataset &train, dataset &test) const {
    train.data.insert(train.data.begin(), data.begin(), data.end());
    random_shuffle(train.data.begin(), train.data.end());
    test.data.insert(test.data.begin(), train.data.begin() + ntrain, train.data.end());
    train.data.resize(ntrain);
  }

  double outputMean() const {
    int n = data.size();
    if (n == 0) {
      return 0.0;
    }
    else {
      int i = 0;
      double s = 0.0;
      while (i < n) {
        s += data[i].output;
        i += 1;
      }
      return s / n;
    }
  }
  
  /**
   * Calculate the mode of the outputs of the training set.
   */
  double outputMode() const {
    /* For classification, choose the majority output rather
     * than the mean. TODO: Fix for multiclass?
     */
    int n = data.size();
    int p = 0;
    for (int i = 0; i < n; i++) {
      if (data[i].output > 0.0) {
        p++;
      }
    }
    int q = n - p;
    return (p > q) ? 1.0 : -1.0;
  }

  /**
   * Calculate the variance of the outputs of a set of training instances.
   * @return The variance in the output labels of the dataset.
   */
  double outputVariance() const {
    double mean = outputMean();
    int n = data.size();
    int i = 0;
    double s = 0.0;
    while (i < n) {
      double t = data[i].output - mean;
      s += t * t;
      i += 1;
    }
    return s / n;
  }

  /** Return two vectors containing the minimum and maximum values
   * along each dimension.
   * @param minv The array of minimum values (to be filled in).
   * @param maxv The array of maximum values (to be filled in).
   */
  void getRanges(double maxv[], double minv[]) const {
    /* Find the minimum and maximum values
     * for each of the attributes in the current training set.
     */
    int M = nd();
    int N = data.size();
    for (int i = 0; i < M; i++) {
      double mn = DBL_MAX;
      double mx = -DBL_MAX;
      for (int j = 0; j < N; j++) {
        const datum &d = data[j];
        double tmp = d.attributes[i];
        if (tmp > mx)
          mx = tmp;
        if (tmp < mn)
          mn = tmp;
      }

      minv[i] = mn;
      maxv[i] = mx;
    }
  }
};
