/**
 * @file UnConstrained.cpp
 * @author C.Marsh
 * @github https://github.com/Craig44
 * @date 26/6/2019
 * @section LICENSE
 *
 * Copyright NIWA Science ©2019 - www.niwa.co.nz
 *
 */

// headers
#include "UnConstrained.h"

#include "Model/Model.h"
#include "Model/Objects.h"
#include "Model/Managers.h"
#include "Estimates/Manager.h"
#include "Estimates/Estimate.h"

// namespaces
namespace niwa {
namespace estimatetransformations {

/**
 * Default constructor
 */
UnConstrained::UnConstrained(Model* model) : EstimateTransformation(model) {
  parameters_.Bind<string>(PARAM_ESTIMATE_LABEL, &estimate_label_, "Label of estimate block to apply transformation. Defined as $\theta_1$ in the documentation", "");
}

/**
 */
void UnConstrained::DoValidate() {

}

/**
 *
 */
void UnConstrained::DoBuild() {
  LOG_FINEST() << "transformation on @estimate " << estimate_label_;
  estimate_ = model_->managers().estimate()->GetEstimateByLabel(estimate_label_);
  if (estimate_ == nullptr) {
    LOG_ERROR_P(PARAM_ESTIMATE) << "Estimate " << estimate_label_ << " could not be found. Have you defined it?";
    return;
  }

  // Initialise for -r runs
  current_untransformed_value_ = estimate_->value();
  original_lower_bound_ = estimate_->lower_bound();
  original_upper_bound_ = estimate_->upper_bound();
  // Can't remember what this stuff does

/*
  LOG_FINE() << "transform with objective = " << transform_with_jacobian_ << " estimeate transform " << estimate_->transform_for_objective() << " together = " << !transform_with_jacobian_ && !estimate_->transform_for_objective();
  if (!transform_with_jacobian_ && !estimate_->transform_for_objective()) {
    LOG_ERROR_P(PARAM_TRANSFORM_WITH_JACOBIAN) << "You have specified a transformation that does not contribute a jacobian, and the prior parameters do not refer to the transformed estimate, in the @estimate" << estimate_label_ << ". This is not advised, and may cause bias estimation. Please address the user manual if you need help";
  }
  if (estimate_->transform_with_jacobian_is_defined()) {
    if (transform_with_jacobian_ != estimate_->transform_with_jacobian()) {
      LOG_ERROR_P(PARAM_TRANSFORM_WITH_JACOBIAN) << "This parameter is not consistent with the equivalent parameter in the @estimate block " << estimate_label_ << ". please make sure these are both true or both false.";
    }
  }

  if (!estimate_->transform_for_objective()) {
    estimate_->set_lower_bound(log(estimate_->lower_bound()));
    estimate_->set_upper_bound(log(estimate_->upper_bound()));
  }
*/

  LOG_FINEST() << "Finish DoBuild()";

}

/**
 * Transform a parameter denoted by x from lb < x < ub -> -Inf < y < Inf
 */
void UnConstrained::DoTransform() {
  LOG_MEDIUM() << "parameter before transform = " << estimate_->value() << " lower bound " << original_lower_bound_ << " upper bound " << original_upper_bound_;
  //(x - lb)/(ub - lb))
  Double x = estimate_->value();
  Double val = (x - original_lower_bound_) / (original_upper_bound_ - original_lower_bound_);
  Double transformed_val = log(val / (1.0 - val));
  LOG_MEDIUM() << "x = " << transformed_val;
  estimate_->set_value(transformed_val);
  LOG_MEDIUM() << "parameter after transform = " << estimate_->value() << " and should be on -Inf -> Inf bounds ";
}

/**
 * Return a parameter value to the coordinates supplied by the user in the @estimate block
 */
void UnConstrained::DoRestore() {
  LOG_MEDIUM() << "parameter before restore = " << estimate_->value() <<  " and should be on -Inf -> Inf bounds ";
  Double x = estimate_->value();
  Double inv_logit_x;
  if (x > 0) {
    Double exp_minus_x = exp(-x);
    inv_logit_x = 1.0 / (1.0 + exp_minus_x);
    // Prevent x from reaching one unless it really really should.
    if ((x < std::numeric_limits<double>::infinity()) && (inv_logit_x == 1))
      inv_logit_x = 1 - 1e-15;
  }  else {
    Double exp_x = exp(x);
    inv_logit_x = 1.0 - 1.0 / (1.0 + exp_x);
    // Prevent x from reaching zero unless it really really should.
    if ((x > -std::numeric_limits<double>::infinity()) && (inv_logit_x == 0))
      inv_logit_x = 1e-15;
  }
  x = original_lower_bound_ + (original_upper_bound_ - original_lower_bound_) * inv_logit_x;
  LOG_MEDIUM() << "x = " << x;
  estimate_->set_value(x);
  LOG_MEDIUM() << "parameter after restore = " << estimate_->value() << " lower bound " << original_lower_bound_ << " upper bound " << original_upper_bound_;
}

/**
 * This method will check if the estimate needs to be transformed for the objective function. If it does then
 * it'll do the transformation.
 */
void UnConstrained::TransformForObjectiveFunction() {
  if (estimate_->transform_for_objective())
    DoTransform();
}

/**
 * This method will check if the estimate needs to be Restored from the objective function. If it does then
 * it'll do the undo the transformation.
 */
void UnConstrained::RestoreFromObjectiveFunction() {
  if (estimate_->transform_for_objective())
    DoRestore();
}


/**
 *
 */
Double UnConstrained::GetScore() {
  if(transform_with_jacobian_) {
    jacobian_ = 1.0 / current_untransformed_value_;
    LOG_MEDIUM() << "jacobian: " << jacobian_;
    return jacobian_;
  } else
    return 0.0;
}

/**
 * Get the target addressables so we can ensure each
 * object is not referencing multiple ones as this would
 * cause chain issues
 *
 * @return Set of addressable labels
 */
std::set<string> UnConstrained::GetTargetEstimates() {
  set<string> result;
  result.insert(estimate_label_);
  return result;
}
} /* namespace estimatetransformations */
} /* namespace niwa */
