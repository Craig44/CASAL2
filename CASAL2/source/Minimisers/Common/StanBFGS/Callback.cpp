
/**
 * @file Callback.cpp
 * @author  Scott Rasmussen (scott.rasmussen@zaita.com)
 * @version 1.0
 * @date 17/04/2013
 * @section LICENSE
 *
 * Copyright NIWA Science ©2013 - www.niwa.co.nz
 *
 * $Date: 2008-03-04 16:33:32 +1300 (Tue, 04 Mar 2008) $
 */

// headers
#include <Minimisers/Common/StanBFGS/Callback.h>

#include "Model/Model.h"
#include "Estimates/Manager.h"
#include "ObjectiveFunction/ObjectiveFunction.h"
#include "EstimateTransformations/Manager.h"

// namespaces
namespace niwa {
namespace minimisers {
namespace stanbfgs {

/**
 * Default Constructor
 */
CallBack::CallBack(Model* model) : model_(model) {

}

/* @return Return Log_prob, the most important function in this call back
 * @param propto = True if calculation is up to proportion (double-only terms dropped). Ignored but required to overload Stan functions
 * @param jacobian_ = add jacobian to log-prob, also ignored because we deal with this in Casal2
 * @param params_r__ unconstrained parameters, this is the most important input in this function.
 * @param params_i__ integer parameters ignored as we don't deal with these.
 * @param pstream__ a writing object, also ignored but requried to mimick the function.
 *
*/


template<bool propto__, bool jacobian__, typename T__>
T__ CallBack::log_prob(vector<T__>& params_r__, vector<int>& params_i__, std::ostream* pstream__) const {
  // Update our Components with the New Parameters

  stan::math::accumulator<T__> lp_accum__; //

  vector<Estimate*> estimates = model_->managers().estimate()->GetIsEstimated();

  if (params_r__.size() != estimates.size()) {
    LOG_CODE_ERROR() << "The number of enabled estimates does not match the number of test solution values";
  }

  for (unsigned i = 0; i < params_r__.size(); ++i)
    estimates[i]->set_value(params_r__[i]);

  model_->managers().estimate_transformation()->RestoreEstimates();
  model_->FullIteration();

  ObjectiveFunction& objective = model_->objective_function();
  objective.CalculateScore();

  model_->managers().estimate_transformation()->TransformEstimates();
  lp_accum__.add(-objective.score());


  return lp_accum__.sum();// Casal2 works with Negative log-likelihood Stan works with log-likelihood space
}

/*
// Overload the above function for eigen style parameters.
template <bool propto, bool jacobian, typename T_>
T_ CallBack::log_prob(Eigen::Matrix<T_,Eigen::Dynamic,1>& params_r, std::ostream* pstream) const {
  std::vector<T_> vec_params_r;
  vec_params_r.reserve(params_r.size());
  for (int i = 0; i < params_r.size(); ++i)
    vec_params_r.push_back(params_r(i));
  std::vector<int> vec_params_i;
  return log_prob<propto,jacobian,T_>(vec_params_r, vec_params_i, pstream);
}
*/


/*
 * This method transforms parameters from constrained -> unconstrained space taking the values by reference & params_r__
 * And saving the unconstrained values as params_r__ using the writer, So we can also copy this functionality pretty easy
 * For a pre-subscribed problem. generalising might be difficult.
*/

void CallBack::transform_inits(const stan::io::var_context& context__,
               std::vector<int>& params_i__,
               std::vector<double>& params_r__,
               std::ostream* pstream__) const {

}


void CallBack::transform_inits(const stan::io::var_context& context,
           Eigen::Matrix<double,Eigen::Dynamic,1>& params_r,
           std::ostream* pstream__) const {
  std::vector<double> params_r_vec;
  std::vector<int> params_i_vec;
  transform_inits(context, params_i_vec, params_r_vec, pstream__);
  params_r.resize(params_r_vec.size());
  for (int i = 0; i < params_r.size(); ++i)
    params_r(i) = params_r_vec[i];
}

void CallBack::get_param_names(std::vector<std::string>& names__) const {
  vector<Estimate*> estimates = model_->managers().estimate()->GetIsEstimated();
  names__.resize(estimates.size());
  unsigned i = 0;
  for (auto& param : estimates) {
    names__[i] = param->parameter();
    ++i;
  }
}



template <typename RNG>
void CallBack::write_array(RNG& base_rng__,
         std::vector<double>& params_r__,
         std::vector<int>& params_i__,
         std::vector<double>& vars__,
         bool include_tparams__,
         bool include_gqs__,
         std::ostream* pstream__ )  {

  vector<Estimate*> estimates = model_->managers().estimate()->GetIsEstimated();
  model_->managers().estimate_transformation()->RestoreEstimates();

}

template <typename RNG>
void CallBack::write_array(RNG& base_rng,
         Eigen::Matrix<double,Eigen::Dynamic,1>& params_r,
         Eigen::Matrix<double,Eigen::Dynamic,1>& vars,
         bool include_tparams,
         bool include_gqs,
         std::ostream* pstream )  {

  std::vector<double> params_r_vec(params_r.size());
  for (int i = 0; i < params_r.size(); ++i)
  params_r_vec[i] = params_r(i);
  std::vector<double> vars_vec;
  std::vector<int> params_i_vec;
  write_array(base_rng,params_r_vec,params_i_vec,vars_vec,include_tparams,include_gqs,pstream);
  vars.resize(vars_vec.size());
  for (int i = 0; i < vars.size(); ++i)
    vars(i) = vars_vec[i];
}


void CallBack::constrained_param_names(std::vector<std::string>& param_names__,  bool include_tparams__, bool include_gqs__ ) const {
  return;
}


void CallBack::unconstrained_param_names(std::vector<std::string>& param_names__,  bool include_tparams__, bool include_gqs__ ) const {
  return;
}
} /* namespace stanbfgs */
} /* namespace minimiser */
} /* namespace niwa */
