/**
 * @file TimeStepProportionsByCategory.cpp
 * @author Scott Rasmussen (scott.rasmussen@zaita.com)
 * @github https://github.com/Zaita
 * @date 10/03/2015
 * @section LICENSE
 *
 * Copyright NIWA Science �2015 - www.niwa.co.nz
 *
 */

// headers
#include "TimeStep.h"

#include "TimeSteps/Manager.h"
#include "Utilities/DoubleCompare.h"

// namespaces
namespace niwa {
namespace observations {
namespace age {

/**
 *
 */
TimeStepProportionsByCategory::TimeStepProportionsByCategory(Model* model)
   : observations::age::ProportionsByCategory(model) {
  parameters_.Bind<Double>(PARAM_TIME_STEP_PROPORTION, &time_step_proportion_, "Proportion through the time step to analyse the partition from", "", Double(0.5));

  mean_proportion_method_ = true;
}

/**
 *
 */
void TimeStepProportionsByCategory::DoBuild() {
  ProportionsByCategory::DoBuild();

  if (time_step_proportion_ < 0.0 || time_step_proportion_ > 1.0)
    LOG_ERROR_P(PARAM_TIME_STEP_PROPORTION) << ": time_step_proportion (" << AS_DOUBLE(time_step_proportion_) << ") must be between 0.0 and 1.0";
  proportion_of_time_ = time_step_proportion_;

  auto time_step = model_->managers().time_step()->GetTimeStep(time_step_label_);
  if (!time_step) {
    LOG_ERROR_P(PARAM_TIME_STEP) << " " << time_step_label_ << " could not be found. Have you defined it?";
  } else {
      for (unsigned year : years_)
        time_step->SubscribeToBlock(this, year);
  }
}

} /* namespace age */
} /* namespace observations */
} /* namespace niwa */
