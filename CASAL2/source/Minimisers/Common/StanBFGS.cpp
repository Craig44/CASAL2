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
#include <Minimisers/Common/StanBFGS/Callback.hpp>

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
  LOG_MEDIUM() << "random seed = " << random_seed;
  estimates::Manager* estimate_manager = model_->managers().estimate();
  CallBack  cas_call_back(model_, estimate_manager->GetIsEstimatedCount());
  LOG_MEDIUM() << "build Stan Call back";

  LOG_MEDIUM() << "log normal(1 | 2, 3)=" << stan::math::normal_log(1, 2, 3);
  // This confirms we have access to Stan functionality


  LOG_MEDIUM() << "callback model_name = " << cas_call_back.model_name();

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
    LOG_MEDIUM() << "value = " << estimate->value();
  }

  double log_p = cas_call_back.log_prob<false, true, double>(start_values, params_i, msgs);
  LOG_MEDIUM() << "log_p = " << log_p;

  /*
  double test_val = 324.2;
  double log_p = cas_call_back.test(test_val);

  LOG_FINE() << "log_p = " << log_p;
   */

  // Calculate gradient
  LOG_MEDIUM() << "before calling log_prob_grad check starting values";
  for(unsigned i = 0; i < start_values.size(); ++i)
    LOG_MEDIUM() << start_values[i];

  // This is the log_prob_grad function
  vector<stan::math::var> ad_params_r(start_values.size());
  for (size_t i = 0; i < cas_call_back.num_params_r(); ++i) {
    stan::math::var var_i(start_values[i]);
    ad_params_r[i] = var_i;
  }
  stan::math::var adLogProb = cas_call_back.template log_prob<true, false>(ad_params_r, params_i, msgs);

  double lp = adLogProb.val();
  adLogProb.grad(ad_params_r, gradient);
  LOG_MEDIUM() << "lp = " << lp;

  for (unsigned i = 0; i < gradient.size(); ++i) {
    LOG_MEDIUM() << "val = " << start_values[i] << " grad = " << gradient[i];
  }



  double log_p_grad = stan::model::log_prob_grad<true, false, CallBack>(cas_call_back, start_values, params_i,gradient, msgs);
  LOG_MEDIUM() << "log_p_grad = " << log_p_grad;

  for (unsigned i = 0; i < gradient.size(); ++i) {
    LOG_MEDIUM() << "val = " << start_values[i] << " grad = " << gradient[i];
  }

  bool gradient_ok = boost::math::isfinite(stan::math::sum(gradient));
  if (!gradient_ok)
    LOG_MEDIUM() << "gradient was not calculated properly = " << stan::math::sum(gradient);

  //-------------- Try an optimisation

  std::stringstream out;
  stan::optimization::BFGSLineSearch<CallBack,stan::optimization::LBFGSUpdate<> > lbfgs(cas_call_back, start_values, params_i, &out);
  lbfgs._conv_opts.tolRelGrad =  1e+7;

  LOG_MEDIUM() << "about to check step()\n\n";
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
