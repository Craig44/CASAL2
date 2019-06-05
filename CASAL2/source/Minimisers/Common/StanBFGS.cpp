/**
 * @file StanBFGS.h
 * @author C.Marsh
 * @version 1.0
 * @date 15/04/2019
 * @section LICENSE
 *
 *
 */
// Local Headers
#include <Minimisers/Common/StanBFGS.h>
#include <Minimisers/Common/StanBFGS/Callback.h>

// Other Casal headers
#include "Model/Model.h"
#include "Estimates/Manager.h"
#include "ObjectiveFunction/ObjectiveFunction.h"
#include "EstimateTransformations/Manager.h"
#include "GlobalConfiguration/GlobalConfiguration.h"

// Stan headers
#include <stan/model/model_header.hpp>
#include <stan/model/finite_diff_grad.hpp>
#include <stan/optimization/bfgs.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/math.hpp>

//#include <test/unit/services/instrumented_callbacks.hpp>
#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>//


// namespaces
namespace niwa {
namespace minimisers {
//using namespace niwa::minimisers::stanbfgs;

/**
 * Default constructor
 */
StanBFGS::StanBFGS(Model* model) : Minimiser(model) {
  parameters_.Bind<int>(PARAM_MAX_ITERATIONS, &max_iterations_, "Maximum number of iterations", "", 1000);
  parameters_.Bind<int>(PARAM_MAX_EVALUATIONS, &max_evaluations_, "Maximum number of evaluations", "", 4000);
  parameters_.Bind<Double>(PARAM_TOLERANCE, &gradient_tolerance_, "Tolerance of the gradient for convergence", "", 0.02);
  parameters_.Bind<Double>(PARAM_STEP_SIZE, &step_size_, "Minimum Step-size before minimisation fails", "", 1e-7);
}

/**
 * Execute the minimiser to solve the model
 */
void StanBFGS::Execute() {
  LOG_MEDIUM();
  // Variables
  // stan::io::empty_var_context context;
  unsigned random_seed = model_->global_configuration().random_seed();
  estimates::Manager* estimate_manager = model_->managers().estimate();
  stanbfgs::CallBack  cas_call_back(model_);
  LOG_FINE() << "build Stan Call back";

  LOG_FINE() << "log normal(1 | 2, 3)=" << stan::math::normal_log(1, 2, 3);
  // This confirms we have access to Stan functionality


  LOG_FINE() << "callback model_name = " << cas_call_back.model_name();

  std::vector<double>  start_values;
  std::vector<int> params_i;
  std::vector<double> gradient;
  std::ostream* msgs = 0;
  // Transform to unconstrained space
  model_->managers().estimate_transformation()->TransformEstimates();
  vector<Estimate*> estimates = estimate_manager->GetIsEstimated();
  for (Estimate* estimate : estimates) {
    if (!estimate->estimated())
      continue;
    start_values.push_back((double)estimate->value());
  }

  double log_p = cas_call_back.log_prob<true, false, double>(start_values, params_i, msgs);
  LOG_FINE() << "log_p = " << log_p;
  // Calculate gradient
  //double log_p_grad = stan::model::log_prob_grad<false, true, CallBack>(call_back, start_values, params_i,gradient, msgs);


  LOG_FINE() << "Finished DoExecute";











/*
  gammadiff::CallBack  call_back(model_);
  estimates::Manager* estimate_manager = model_->managers().estimate();
  LOG_FINE() << "estimate_manager: " << estimate_manager;

  vector<double>  lower_bounds;
  vector<double>  upper_bounds;
  vector<double>  start_values;

  model_->managers().estimate_transformation()->TransformEstimates();
  vector<Estimate*> estimates = estimate_manager->GetIsEstimated();
  LOG_FINE() << "estimates.size(): " << estimates.size();
  for (Estimate* estimate : estimates) {
    if (!estimate->estimated())
      continue;

    LOG_FINE() << "Estimate: " << estimate;
    LOG_FINE() << "transformed value: " << estimate->value();
    LOG_FINE() << "Parameter: " << estimate->parameter();

    lower_bounds.push_back((double)estimate->lower_bound());
    upper_bounds.push_back((double)estimate->upper_bound());
    start_values.push_back((double)estimate->value());

    if (estimate->value() < estimate->lower_bound()) {
      LOG_FATAL() << "When starting the GammDiff numerical_differences minimiser the starting value (" << estimate->value() << ") for estimate "
          << estimate->parameter() << " was less than the lower bound (" << estimate->lower_bound() << ")";
    } else if (estimate->value() > estimate->upper_bound()) {
      LOG_FATAL() << "When starting the GammDiff numerical_differences minimiser the starting value (" << estimate->value() << ") for estimate "
          << estimate->parameter() << " was greater than the upper bound (" << estimate->upper_bound() << ")";
    }
  }

  LOG_FINE() << "Launching minimiser";
  int status = 0;
  gammadiff::Engine clGammaDiff;
  clGammaDiff.optimise_finite_differences(call_back,
      start_values, lower_bounds, upper_bounds,
      status, max_iterations_, max_evaluations_, gradient_tolerance_,
      hessian_,1,step_size_);

  model_->managers().estimate_transformation()->RestoreEstimates();

  switch(status) {
    case -1:
      result_ = MinimiserResult::kError;
      break;
    case 0:
      result_ = MinimiserResult::kTooManyIterations;
      break;
    case 1:
      result_ = MinimiserResult::kSuccess;
      break;
    default:
      break;
  }
  */
}

} /* namespace minimisers */
} /* namespace niwa */
