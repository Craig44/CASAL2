/**
 * @file StanBFGS.h
 * @author C.Marsh
 * @version 1.0
 * @date 15/04/2019
 * @section LICENSE
 *
 * Copyright NIWA Science ©2019 BigBlueData.co.nz
 *
 * @section DESCRIPTION
 *
 * This minimiser is borrowed from the Stan C++ library
 */
#ifndef MINIMISERS_STAN_BFGS_H_
#define MINIMISERS_STAN_BFGS_H_

// headers
#include "Minimisers/Minimiser.h"

// namespaces
namespace niwa {
namespace minimisers {

//**********************************************************************
//
//
//**********************************************************************
class StanBFGS :  public niwa::Minimiser  {
public:
  // Methods
  StanBFGS(Model* model);
  virtual                     ~StanBFGS() = default;
  void                        DoValidate() override final { };
  void                        DoBuild() override final { };
  void                        DoReset() override final { };
  void                        Execute() override final;

private:
  // Members
  int                         max_iterations_;
  int                         max_evaluations_;
  double                      gradient_tolerance_;
  double                      step_size_;
};

} /* namespace minimisers */
} /* namespace niwa */

#endif /* MINIMISERS_STAN_BFGS_H_ */
