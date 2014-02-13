/**
 * @file Normal.cpp
 * @author  Scott Rasmussen (scott.rasmussen@zaita.com)
 * @version 1.0
 * @date 8/03/2013
 * @section LICENSE
 *
 * Copyright NIWA Science �2013 - www.niwa.co.nz
 *
 * $Date: 2008-03-04 16:33:32 +1300 (Tue, 04 Mar 2008) $
 */

// Headers
#include "Normal.h"

// Namespaces
namespace isam {
namespace estimates {

/**
 * Default constructor
 */
Normal::Normal() {
  parameters_.Bind<double>(PARAM_MU, &mu_, "Mu");
  parameters_.Bind<double>(PARAM_CV, &cv_, "Cv");
}

/**
 * Validate the parameters passed in from the configuration file
 */
void Normal::DoValidate() {
  if (cv_ <= 0.0)
    LOG_ERROR(parameters_.location(PARAM_CV) << ": cv (" << AS_DOUBLE(cv_) << ") cannot be less than or equal to 0.0");
}

/**
 * Calculate and return the score for this prior
 */
Double Normal::GetScore() {
  Double score_ = 0.5 * ((value() - mu_) / (cv_ * mu_)) * ((value() - mu_) / (cv_ * mu_));
  return score_;
}

} /* namespace estimates */
} /* namespace isam */
