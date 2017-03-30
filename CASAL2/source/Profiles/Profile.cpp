/**
 * @file Profile.cpp
 * @author  Scott Rasmussen (scott.rasmussen@zaita.com)
 * @date 28/03/2014
 * @section LICENSE
 *
 * Copyright NIWA Science �2014 - www.niwa.co.nz
 *
 */

// headers
#include "Profile.h"

#include "Estimates/Children/Uniform.h"
#include "Model/Model.h"
#include "Model/Objects.h"
#include "Utilities/To.h"

// namespaces
namespace niwa {

namespace util = niwa::utilities;

/**
 * default constructor
 */
Profile::Profile(Model* model) : model_(model) {
  parameters_.Bind<string>(PARAM_LABEL, &label_, "Label", "", "");
  parameters_.Bind<unsigned>(PARAM_STEPS, &steps_, "The number of steps to take between the lower and upper bound", "");
  parameters_.Bind<Double>(PARAM_LOWER_BOUND, &lower_bound_, "The lower bounds", "");
  parameters_.Bind<Double>(PARAM_UPPER_BOUND, &upper_bound_, "The upper bounds", "");
  parameters_.Bind<string>(PARAM_PARAMETER, &parameter_, "The system parameter to profile", "");
  parameters_.Bind<string>(PARAM_SAME, &same_parameter_, "A Parameter that are constrained to have the same value as the parameter being profiled", "", "");
}

/**
 *
 */
void Profile::Validate() {
  parameters_.Populate();
}

/**
 *
 */
void Profile::Build() {
  string type       = "";
  string label      = "";
  string parameter  = "";
  string index      = "";

  /**
   * Explode the parameter string so we can get the estimable
   * name (parameter) and the index
   */

  model_->objects().ExplodeString(parameter_, type, label, parameter, index);
  if (type == "" || label == "" || parameter == "") {
    LOG_ERROR_P(PARAM_PARAMETER) << ": parameter " << parameter_
        << " is not in the correct format. Correct format is object_type[label].estimable(array index)";
  }
  model_->objects().ImplodeString(type, label, parameter, index, parameter_);

  string error = "";
  base::Object* target = model_->objects().FindObject(parameter_, error);
  if (!target) {
    LOG_ERROR_P(PARAM_PARAMETER) << ": parameter " << parameter_ << " is not a valid estimable in the system";
    return;
  }

  // If Estimable is a map need to give index
  target_ = target->GetEstimable(parameter,index);

  LOG_FINEST() << "Running profile on parameter: " << parameter << ", that has type: " << type << " and label: " << label;
  if (target_ == 0)
    LOG_ERROR_P(PARAM_PARAMETER) << ": parameter " << parameter_ << " is not a valid estimable in the system";
  original_value_ = *target_;

  step_size_ = (upper_bound_ - lower_bound_) / (steps_ + 1);

  /**
   * Deal with the same parameter
   */
  if (parameters_.Get(PARAM_SAME)->has_been_defined()) {
    string same_type       = "";
    string same_label      = "";
    string same_parameter  = "";
    string same_index      = "";

    /**
     * Explode the parameter string so we can get the estimable
     * name (parameter) and the index
     */
    model_->objects().ExplodeString(same_parameter_, same_type, same_label, same_parameter, same_index);
    if (same_type == "" || same_label == "" || same_parameter == "") {
      LOG_ERROR_P(PARAM_SAME) << ": same parameter " << same_parameter_
          << " is not in the correct format. Correct format is object_type[label].estimable(array index)";
    }
    model_->objects().ImplodeString(same_type, same_label, same_parameter, same_index, same_parameter_);

    string error = "";
    base::Object* same_target = model_->objects().FindObject(same_parameter_, error);
    if (!same_target) {
      LOG_ERROR_P(PARAM_SAME) << ": same parameter " << same_parameter_ << " is not a valid estimable in the system";
      return;
    }

    // If Estimable is a map need to give index
    same_target_ = same_target->GetEstimable(same_parameter,same_index);
    if (same_target_ == 0)
      LOG_ERROR_P(PARAM_SAME) << ": same parameter " << same_parameter_ << " is not a valid estimable in the system";
    same_original_value_ = *same_target_;
    LOG_MEDIUM() << "start_value for same param = " << same_original_value_;
  }
}

/**
 *
 */
void Profile::FirstStep() {
  *target_ = lower_bound_;
  if (parameters_.Get(PARAM_SAME)->has_been_defined()) {
    *same_target_ = lower_bound_;
    LOG_MEDIUM() << "esitmating with profile parameter = " <<  *target_  << " and same parameter = " << *same_target_;
  }
}
/**
 *
 */
void Profile::NextStep() {
  *target_ += step_size_;
  if (parameters_.Get(PARAM_SAME)->has_been_defined()) {
    *same_target_ += step_size_;
    LOG_MEDIUM() << "esitmating with profile parameter = " <<  *target_  << " and same parameter = " << *same_target_;
  }
}

void Profile::RestoreOriginalValue() {
  *target_ = original_value_;
  if (parameters_.Get(PARAM_SAME)->has_been_defined()) {
    *same_target_ = original_value_;
    LOG_MEDIUM() << "esitmating with profile parameter = " <<  *target_  << " and same parameter = " << *same_target_;
  }
}

} /* namespace niwa */
