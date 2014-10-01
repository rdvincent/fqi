/** \file
 * \brief Contains definitions of the datum and dataset classes used
 * internally by the FQI algorithm.
 *
 * Copyright (c) 2008-2014 Robert D. Vincent.
 */
/** The class that represents a single data point in our larger \c dataset.
 */
class datum {
public:
  double output;                /*!< Q-function value for this point. */
  vector<double> attributes;    /*!< State or state/action variables. */

  datum() {
  }

  /** Construct a \c datum */
  datum(double o, vector<double> a) {
    output = o;
    attributes = a;
  }
};

/** The class that represents our training data in the FQI algorithm.
 */
class dataset {
public:
  vector<datum> data;           /*!< The set of data points. */

  /** Returns the size of the dataset */
  size_t size() const { return data.size(); }

  /** Returns the number of dimensions in the data set. */
  size_t nd() const { return data[0].attributes.size(); }

  /** For a given number of training examples, returns a randomly-selected
   * training and test dataset derived from this dataset.
   * \param ntrain Number of training samples desired.
   * \param train The resulting training dataset.
   * \param test The resulting testing dataset.
   */
  void randomFold(int ntrain, dataset &train, dataset &test) const {
    train.data.insert(train.data.begin(), data.begin(), data.end());
    random_shuffle(train.data.begin(), train.data.end());
    test.data.insert(test.data.begin(), train.data.begin() + ntrain, train.data.end());
    train.data.resize(ntrain);
  }

  /**
   * Calculate the mean of the output values for the dataset.
   * \return The mean (average) value of the output labels of the dataset.
   */
  double outputMean() const {
    size_t n = data.size();
    if (n == 0) {
      return 0.0;
    }
    else {
      size_t i = 0;
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
   * \return The modal value of the output labels of the dataset.
   */
  double outputMode() const {
    /* For classification, choose the majority output rather
     * than the mean. TODO: Fix for multiclass?
     */
    size_t n = data.size();
    size_t p = 0;
    for (size_t i = 0; i < n; i++) {
      if (data[i].output > 0.0) {
        p++;
      }
    }
    size_t q = n - p;
    return (p > q) ? 1.0 : -1.0;
  }

  /**
   * Calculate the variance of the outputs of a set of training instances.
   * \return The variance in the output labels of the dataset.
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

  /**
   * Return two arrays containing the minimum and maximum values
   * along each dimension. The arrays \b must be allocated to contain
   * at least as many members as there are dimensions in the dataset.
   *
   * \param minv The resulting array of minimum values.
   * \param maxv The resulting array of maximum values.
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
