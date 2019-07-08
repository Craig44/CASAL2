/**
 * @file LeastSquares.h
 * @author  C.Marsh
 * @date 5/7/2019
 * @section LICENSE
 *
 *
 * @section DESCRIPTION
 *
 * << Add Description >>
 */
#ifndef AGE_OBSERVATIONS_LEASTSQUARES_H_
#define AGE_OBSERVATIONS_LEASTSQUARES_H_

// headers
#include "Observations/Observation.h"


// namespaces
namespace niwa {
namespace observations {
namespace age {

/**
 * class definition
 */
class LeastSquares : public niwa::Observation {
public:
  // methods
  LeastSquares(Model* model);
  virtual                     ~LeastSquares() = default;
  void                        DoValidate() override final;
  virtual void                DoBuild() override;
  void                        DoReset() override final;
  void                        PreExecute() override final;
  void                        Execute() override final;
  void                        CalculateScore() override final;
  bool                        HasYear(unsigned year) const override final { return std::find(years_.begin(), years_.end(), year) != years_.end(); }

protected:
  // members
  vector<unsigned>                years_;
  vector<double>                  covariates_;
  vector<double>                  observed_;
  Double                          a_;
  Double                          b_;
  double                          sigma_;
  vector<Double>                  y_hat_;
  string                          time_step_label_ = "";


};

} /* namespace age */
} /* namespace niwa */
} /* namespace observations */

#endif /* AGE_OBSERVATIONS_LEASTSQUARES_H_ */
