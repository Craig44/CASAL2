/**
 * @file Callback.hpp
 * @author C.Marsh
 * @version 1.0
 * @date 15/04/2019
 * @section LICENSE
 *
 * Copyright NIWA Science ©2019 BigBlueData.co.nz
 *
 * @section I went to .hpp because we are implementing non-specialized template
 * functions which need to be visible to translation units that use it see https://stackoverflow.com/questions/10632251/undefined-reference-to-template-function
 *
 * Implement a call back that Stan minimiser can call back with.
 */
#ifndef MINIMISERS_STAN_BFGS_CALLBACK_H_
#define MINIMISERS_STAN_BFGS_CALLBACK_H_
// Headers

#include "Model/Model.h"
#include "Estimates/Manager.h"
#include "ObjectiveFunction/ObjectiveFunction.h"
#include "EstimateTransformations/Manager.h"

// namespaces
namespace niwa {
namespace minimisers {
namespace stanbfgs {

using std::vector;
using stan::io::dump;
using stan::model::prob_grad;
using namespace stan::math;
/**
 * Class definition
 */
class CallBack {
public:
  CallBack(Model* model, size_t number_of_pars) : num_params_r__(number_of_pars), model_(model) {

  }
  virtual                     ~CallBack() = default;

  template<bool propto__, bool jacobian__, typename T__>
  T__                       log_prob(vector<T__>& params_r__, vector<int>& params_i__, std::ostream* pstream__ = 0) const {
    accumulator<T__> lp_accum__; //
    LOG_MEDIUM() << "entering log_prob";
    vector<Estimate*> estimates = model_->managers().estimate()->GetIsEstimated();

    if (params_r__.size() != estimates.size()) {
      LOG_CODE_ERROR() << "The number of enabled estimates does not match the number of test solution values";
    }
    double val = 0.0;
    for (unsigned i = 0; i < params_r__.size(); ++i) {
      val = stan::math::value_of(params_r__[i]);
      LOG_MEDIUM() << "setting value to = " << val;
      estimates[i]->set_value(val); // the grad function has params_r__ as a vector of stan::math::var objects.
    }

    model_->managers().estimate_transformation()->RestoreEstimates();
    model_->FullIteration();



    ObjectiveFunction& objective = model_->objective_function();
    objective.CalculateScore();

    //model_->managers().estimate_transformation()->TransformEstimates();
    lp_accum__.add(-objective.score());// Casal2 works with Negative log-likelihood Stan works with log-likelihood space


    return lp_accum__.sum();


  }

  template <bool propto, bool jacobian, typename T_>
  T_                        log_prob(Eigen::Matrix<T_,Eigen::Dynamic,1>& params_r, std::ostream* pstream = 0) const {
    std::vector<T_> vec_params_r;
    vec_params_r.reserve(params_r.size());
    for (int i = 0; i < params_r.size(); ++i)
      vec_params_r.push_back(params_r(i));
    std::vector<int> vec_params_i;
    return log_prob<propto,jacobian,T_>(vec_params_r, vec_params_i, pstream);
  }
  //template <bool propto, bool jacobian, typename T_>
  //T_                        log_prob(Eigen::Matrix<T_,Eigen::Dynamic,1>& params_r, std::ostream* pstream) const;
  void                      transform_inits(const stan::io::var_context& context__, std::vector<int>& params_i__, std::vector<double>& params_r__, std::ostream* pstream__) const {
    return;
  }
  void                      transform_inits(const stan::io::var_context& context, Eigen::Matrix<double,Eigen::Dynamic,1>& params_r, std::ostream* pstream__) const {
    std::vector<double> params_r_vec;
    std::vector<int> params_i_vec;
    transform_inits(context, params_i_vec, params_r_vec, pstream__);
    params_r.resize(params_r_vec.size());
    for (int i = 0; i < params_r.size(); ++i)
      params_r(i) = params_r_vec[i];
  }

  inline size_t             num_params_r() const { return num_params_r__;}
  inline size_t             num_params_i() const {return param_ranges_i__.size(); }
  inline std::pair<int, int> param_range_i(size_t idx) const { return param_ranges_i__[idx];}
  // thes must be for reporting I think can ignore for now, just make sure it
  // modifies the names__ vector to be of same length as params.
  void  get_param_names(std::vector<std::string>& names__) const {
    vector<Estimate*> estimates = model_->managers().estimate()->GetIsEstimated();
    names__.resize(estimates.size());
    unsigned i = 0;
    for (auto& param : estimates) {
      names__[i] = param->parameter();
      ++i;
    }
  }
  static std::string        model_name() { return "Casal2";}

  template <typename RNG>
  void write_array(RNG& base_rng__, std::vector<double>& params_r__, std::vector<int>& params_i__, std::vector<double>& vars__, bool include_tparams__ = true, bool include_gqs__ = true,
           std::ostream* pstream__ = 0) {
    return;
  }
  template <typename RNG>
  void                      write_array(RNG& base_rng, Eigen::Matrix<double,Eigen::Dynamic,1>& params_r, Eigen::Matrix<double,Eigen::Dynamic,1>& vars, bool include_tparams = true, bool include_gqs = true, std::ostream* pstream = 0) {
    return;
  }
  void                      constrained_param_names(std::vector<std::string>& param_names__, bool include_tparams__, bool include_gqs__ ) const {
    return;
  }
  void                      unconstrained_param_names(std::vector<std::string>& param_names__, bool include_tparams__, bool include_gqs__ ) const {
    return;
  }

protected:
  size_t num_params_r__;
  std::vector<std::pair<int, int> > param_ranges_i__;
private:
  Model*                    model_;
};

} /* namespace stanbfgs */
} /* namespace minimiser */
}

#endif /* MINIMISERS_STAN_BFGS_CALLBACK_H_ */
