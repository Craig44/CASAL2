/**
 * @file StanBFGS.h
 * @author C.Marsh
 * @version 1.0
 * @date 15/04/2019
 * @section LICENSE
 *
 *
 */
#ifdef USE_AUTODIFF
#ifdef USE_STAN
// Local Headers
#include <Minimisers/Common/StanBFGS.h>
#include <Minimisers/Common/StanBFGS/Callback.hpp>

// Other Casal headers
#include "Model/Model.h"
#include "Estimates/Manager.h"
#include "ObjectiveFunction/ObjectiveFunction.h"
#include "EstimateTransformations/Manager.h"
#include "GlobalConfiguration/GlobalConfiguration.h"

// Stan headers
//#include <stan/model/model_header.hpp>
#include <stan/optimization/bfgs.hpp>

//#include <stan/math.hpp>


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

  // Transform to unconstrained space
  estimates::Manager* estimate_manager = model_->managers().estimate();

  model_->managers().estimate_transformation()->TransformEstimates();
  vector<Estimate*> estimates = estimate_manager->GetIsEstimated();
  // Variables
  // stan::io::empty_var_context context;
  unsigned random_seed = model_->global_configuration().random_seed();
  LOG_MEDIUM() << "random seed = " << random_seed;
  stanbfgs::CallBack  cas_call_back(model_, estimate_manager->GetIsEstimatedCount());
  LOG_MEDIUM() << "build Stan Call back, num params = " << estimate_manager->GetIsEstimatedCount();

  LOG_MEDIUM() << "log normal(1 | 2, 3)=" << stan::math::normal_log(1, 2, 3);
  // This confirms we have access to Stan functionality


  LOG_MEDIUM() << "callback model_name = " << cas_call_back.model_name();

  std::vector<Double>  start_values;
  std::vector<int> params_i;
  std::vector<Double> gradient;
  std::ostream* msgs = 0;

/*

  vector<Double> actual_vals = {2.07018, 2.33322 };
  unsigned iter = 0;
  for (Estimate* estimate : estimates) {
    if (!estimate->estimated())
      continue;
    estimate->set_value(actual_vals[iter]);
    ++iter;
  }
  model_->FullIteration();
  ObjectiveFunction& objective = model_->objective_function();
  objective.CalculateScore();
  Double ll = objective.score(); // endpoint in the stack = dependent variable (function value)
  ll.grad();
  stan::math::print_stack(std::cerr);

  LOG_MEDIUM() << "ll = " << ll.val();
  for (Estimate* estimate : estimates) {
    if (!estimate->estimated())
      continue;
    LOG_MEDIUM() << estimate->value().val();
    LOG_MEDIUM() << estimate->value().adj();
  }
*/

  for (Estimate* estimate : estimates) {
    if (!estimate->estimated())
      continue;
    start_values.push_back(estimate->value());
    LOG_MEDIUM() << "value = " << estimate->value();
  }

  Double log_p = cas_call_back.log_prob<false, true, Double>(start_values, params_i, msgs);
  LOG_MEDIUM() << "log_p = " << log_p;

  vector<double> test_grad(start_values.size(),0.0);
  log_p.grad(start_values, test_grad);


  for (unsigned i = 0; i < start_values.size(); ++i) {
    LOG_MEDIUM() << "val = " << start_values[i].val();
    LOG_MEDIUM() << "adj = " << start_values[i].adj();
    LOG_MEDIUM() << "grad = " << test_grad[i];
  }

/*
  LOG_MEDIUM() << "clear memory";
  stan::math::recover_memory(); // This is getting called somewhere which is fucking everything up.
  LOG_MEDIUM() << "try again";
  log_p = cas_call_back.log_prob<false, true, Double>(start_values, params_i, msgs);
  LOG_MEDIUM() << "complete";*/

/*
  //stan::math::print_stack(std::cerr);
  // Calculate gradient
  LOG_MEDIUM() << "before calling log_prob_grad check starting values";
  for(unsigned i = 0; i < start_values.size(); ++i)
    LOG_MEDIUM() << start_values[i];

  // This is the log_prob_grad function
  vector<Double> ad_params_r(start_values.size());
  for (size_t i = 0; i < cas_call_back.num_params_r(); ++i) {
    Double var_i(start_values[i]);
    ad_params_r[i] = var_i;
  }


  // test gradient
  model_->managers().estimate_transformation()->TransformEstimates();
  vector<double> starting_vals_for_grad;
  for (Estimate* estimate : estimates) {
    if (!estimate->estimated())
      continue;
    double this_val = AS_DOUBLE(estimate->value());
    starting_vals_for_grad.push_back(this_val);
  }
  vector<double> gradients_grad(starting_vals_for_grad.size(),0.0);

  LOG_MEDIUM() << "About to calc gradient";
  double log_p_grad = stan::model::log_prob_grad<true, false, stanbfgs::CallBack>(cas_call_back, starting_vals_for_grad, params_i, gradients_grad, msgs);


  for (unsigned i = 0; i < gradients_grad.size(); ++i) {
    LOG_MEDIUM() << "val = " << starting_vals_for_grad[i] << " grad = " << gradients_grad[i];
  }

  bool gradient_ok = boost::math::isfinite(stan::math::sum(gradients_grad));
  if (!gradient_ok)
    LOG_MEDIUM() << "gradient was not calculated properly = " << stan::math::sum(gradients_grad);

  LOG_MEDIUM() << "about to enter estimation, fingers crossed";
  */
  //-------------- Try an optimisation
  /*Eigen::Matrix<double, Eigen:: Dynamic, 1> vals_for_adaptor_;
  vals_for_adaptor_.resize(starting_vals_for_grad.size());*/
  vector<double> starting_vals_for_optim;
  unsigned i = 0;
  for (Estimate* estimate : estimates) {
    if (!estimate->estimated())
      continue;
    double this_val = AS_DOUBLE(estimate->value());
    //vals_for_adaptor_(i) = this_val;
    starting_vals_for_optim.push_back(this_val);
    ++i;
  }
  std::stringstream out;

/*  LOG_MEDIUM() << "about to build model adaptor \n";
  stan::optimization::ModelAdaptor<stanbfgs::CallBack> _adaptor(cas_call_back, params_i, &out);
  LOG_MEDIUM() << "Build model adaptor \n";
  Eigen::Matrix<double, Eigen:: Dynamic, 1> gradient_for_adaptor_;
  double f;

  _adaptor(vals_for_adaptor_,f);

  LOG_MEDIUM() << "f = " << f;*/

  //int result = _adaptor();
  //LOG_MEDIUM() << result <<  "test () operator \n";


  stan::optimization::BFGSLineSearch<stanbfgs::CallBack,stan::optimization::LBFGSUpdate<> > lbfgs(cas_call_back, starting_vals_for_optim, params_i, &out);
  LOG_MEDIUM() << "Successfully built lfbgs \n\n";

  lbfgs._conv_opts.tolRelGrad =  1e+7;

  int ret = 0;
  while (ret == 0) {
    ret = lbfgs.step();
  }
  LOG_MEDIUM() << lbfgs.get_code_string(ret) << "\n";
  // Print message
  LOG_MEDIUM() << "ret = " << ret << endl;

  LOG_MEDIUM() << "grad evals = " << lbfgs.grad_evals() << endl;
  LOG_MEDIUM() << "logp = " << lbfgs.logp() << endl;
  LOG_MEDIUM() << "grad norm = " << lbfgs.grad_norm() << endl;

  LOG_MEDIUM() << "result = \n\n";
  LOG_MEDIUM() << "size of current x = " << lbfgs.curr_x().size() << "\n";
  auto current_f =  lbfgs.curr_x();
  for (unsigned i = 0; i < current_f.size(); ++i) {
    LOG_MEDIUM()<< current_f[i] << " ";
  }



  LOG_MEDIUM() << "Finished DoExecute";


}

} /* namespace minimisers */
} /* namespace niwa */
#endif /* USE_STAN */
#endif /* USE_AUTODIFF */
