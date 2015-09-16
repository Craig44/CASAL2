/*
 * BetaDiff.cpp
 *
 *  Created on: 20/05/2013
 *      Author: Admin
 */
#ifdef USE_AUTODIFF
#ifdef USE_BETADIFF
#include "BetaDiff.h"

#include <betadiff.h>

#include "Estimates/Manager.h"
#include "Model/Model.h"
#include "ObjectiveFunction/ObjectiveFunction.h"

namespace niwa {
namespace minimisers {

/**
 * Objective Function
 */
class MyModel {};
class MyObjective {
public:
  Double operator()(const MyModel& model, const dvv& x_unbounded) {
    vector<EstimatePtr> estimates = estimates::Manager::Instance().GetEnabled();

    for (int i = 0; i < x_unbounded.size(); ++i) {
      dvariable estimate = x_unbounded[i + 1];
      estimates[i]->SetTransformedValue(estimate.x);
    }

    ObjectiveFunction& objective = ObjectiveFunction::Instance();
    Model::Instance()->FullIteration();

    objective.CalculateScore();
    Double score = objective.score();

    return score;
  }
};


/**
 * Default constructor
 */
BetaDiff::BetaDiff() {
  parameters_.Bind<int>(PARAM_MAX_ITERATIONS, &max_iterations_, "Maximum number of iterations", "", 1000);
  parameters_.Bind<int>(PARAM_MAX_EVALUATIONS, &max_evaluations_, "Maximum number of evaluations", "", 4000);
  parameters_.Bind<double>(PARAM_TOLERANCE, &gradient_tolerance_, "Tolerance of the gradient for convergence", "", 2e-3);
}

/**
 *
 */
void BetaDiff::Execute() {
  estimates::Manager& estimate_manager = estimates::Manager::Instance();

  vector<EstimatePtr> estimates = estimate_manager.GetEnabled();


  dvector lower_bounds((int)estimates.size());
  dvector upper_bounds((int)estimates.size());
  dvector start_values((int)estimates.size());

  int i = 0;
  for (EstimatePtr estimate : estimates) {
    ++i;

    if (!estimate->enabled())
      continue;

    lower_bounds[i] = AS_DOUBLE(estimate->lower_bound());
    upper_bounds[i] = AS_DOUBLE(estimate->upper_bound());
    start_values[i] = AS_DOUBLE(estimate->GetTransformedValue());

//    if (estimate->value() < estimate->lower_bound()) {
//      LOG_ERROR_P("When starting the DESolver minimiser the starting value (" << estimate->value() << ") for estimate "
//          << estimate->parameter() << " was less than the lower bound (" << estimate->lower_bound() << ")");
//    } else if (estimate->value() > estimate->upper_bound()) {
//      LOG_ERROR_P("When starting the DESolver minimiser the starting value (" << estimate->value() << ") for estimate "
//          << estimate->parameter() << " was greater than the upper bound (" << estimate->upper_bound() << ")");
//    }
  }

  MyModel my_model;
  MyObjective my_objective;

  dmatrix optimise_hessian(estimates.size(), estimates.size());
  int convergence = 0;
  double score = optimise<MyModel, MyObjective>(my_model, my_objective, start_values, lower_bounds, upper_bounds, convergence, 0,
      max_iterations_, max_evaluations_, gradient_tolerance_, 0, &optimise_hessian);

  for (int row = 0; row < (int)estimates.size(); ++row) {
    for (int col = 0; col < (int)estimates.size(); ++col) {
      hessian_[row][col] = optimise_hessian[row+1][col+1];
    }
  }
}

} /* namespace reports */
} /* namespace niwa */
#endif /* USE_BETADIFF */
#endif /* USE_AUTODIFF */
