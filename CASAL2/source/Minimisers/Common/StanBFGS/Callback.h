/**
 * @file Callback.h
 * @author  Scott Rasmussen (scott.rasmussen@zaita.com)
 * @version 1.0
 * @date 17/04/2013
 * @section LICENSE
 *
 * Copyright NIWA Science ©2013 - www.niwa.co.nz
 *
 * @section DESCRIPTION
 *
 * The time class represents a moment of time.
 *
 * $Date: 2008-03-04 16:33:32 +1300 (Tue, 04 Mar 2008) $
 */
#ifndef MINIMISERS_STANBFGS_CALLBACK_H_
#define MINIMISERS_STANBFGS_CALLBACK_H_

// Headers

#include "Model/Model.h"

#include <stan/model/model_header.hpp>
#include <stan/io/empty_var_context.hpp>

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
  CallBack(Model* model);
  virtual                     ~CallBack() = default;

  template<bool propto__, bool jacobian__, typename T__>
  T__                       log_prob(vector<T__>& params_r__, vector<int>& params_i__, std::ostream* pstream__) const ;
  template <bool propto, bool jacobian, typename T_>
  T_                        log_prob(Eigen::Matrix<T_,Eigen::Dynamic,1>& params_r, std::ostream* pstream) const;
  void                      transform_inits(const stan::io::var_context& context__, std::vector<int>& params_i__, std::vector<double>& params_r__, std::ostream* pstream__) const ;
  void                      transform_inits(const stan::io::var_context& context, Eigen::Matrix<double,Eigen::Dynamic,1>& params_r, std::ostream* pstream__) const;

  inline size_t             num_params_r() const { return num_params_r__;}
  inline size_t             num_params_i() const {return param_ranges_i__.size(); }
  inline std::pair<int, int> param_range_i(size_t idx) const { return param_ranges_i__[idx];}
  // thes must be for reporting I think can ignore for now, just make sure it
  // modifies the names__ vector to be of same length as params.
  void                      get_param_names(std::vector<std::string>& names__) const;
  static std::string        model_name() { return "Casal2";}

  template <typename RNG> void write_array(RNG& base_rng__, std::vector<double>& params_r__, std::vector<int>& params_i__, std::vector<double>& vars__, bool include_tparams__ = true, bool include_gqs__ = true,
           std::ostream* pstream__ = 0) ;
  template <typename RNG>
  void                      write_array(RNG& base_rng, Eigen::Matrix<double,Eigen::Dynamic,1>& params_r, Eigen::Matrix<double,Eigen::Dynamic,1>& vars, bool include_tparams = true, bool include_gqs = true, std::ostream* pstream = 0) ;
  void                      constrained_param_names(std::vector<std::string>& param_names__, bool include_tparams__, bool include_gqs__ ) const;
  void                      unconstrained_param_names(std::vector<std::string>& param_names__, bool include_tparams__, bool include_gqs__ ) const;

protected:
  size_t num_params_r__;
  std::vector<std::pair<int, int> > param_ranges_i__;
private:
  Model*                    model_;
};

} /* namespace stanbfgs */
} /* namespace minimiser */
} /* namespace niwa */

#endif /* MINIMISERS_STANBFGS_CALLBACK_H_ */
