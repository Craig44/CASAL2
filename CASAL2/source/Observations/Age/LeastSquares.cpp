/**
 * @file LeastSquares.cpp
 * @author  C.Marsh
 * @date 5/7/2019
 * @section LICENSE
 *
 *
 */

// headers
#include "LeastSquares.h"

#include "TimeSteps/Manager.h"
// namespaces
namespace niwa {
namespace observations {
namespace age {

namespace utils = niwa::utilities;

/**
 * Default constructor
 */
LeastSquares::LeastSquares(Model* model) : Observation(model) {
  parameters_.Bind<Double>(PARAM_A, &a_, "y-intercept", "");
  parameters_.Bind<Double>(PARAM_B, &b_, "slope parameter", "");
  parameters_.Bind<double>(PARAM_X, &covariates_, "Covariate", "");
  parameters_.Bind<double>(PARAM_Y, &observed_, "Observed vals", "");
  parameters_.Bind<double>(PARAM_SIGMA, &sigma_, "obs std", "");
  parameters_.Bind<string>(PARAM_TIME_STEP, &time_step_label_, "The label of time-step that the observation occurs in", "");

  RegisterAsAddressable(PARAM_A, &a_);
  RegisterAsAddressable(PARAM_B, &b_);

  allowed_likelihood_types_.push_back(PARAM_NORMAL);
  allowed_likelihood_types_.push_back(PARAM_LOGNORMAL);
}

/**
 *
 */
void LeastSquares::DoValidate() {
  LOG_MEDIUM() << "DoValidate";
  LOG_TRACE();
  if (covariates_.size() != observed_.size())
    LOG_ERROR_P(PARAM_X) << "x and y need to be the same length";
  y_hat_.resize(observed_.size(),0.0);

  unsigned year = 1975;
  years_.push_back(year);
}

/**
 *
 */
void LeastSquares::DoBuild() {
  LOG_MEDIUM() << "DoBuild";


  auto time_step = model_->managers().time_step()->GetTimeStep(time_step_label_);
  if (!time_step) {
    LOG_ERROR_P(PARAM_TIME_STEP) << time_step_label_ << " could not be found. Have you defined it?";
  } else {
    for (unsigned year : years_)
      time_step->SubscribeToBlock(this, year);
  }
  for (auto year : years_) {
    if((year < model_->start_year()) || (year > model_->final_year()))
      LOG_ERROR_P(PARAM_YEARS) << "Years can't be less than start_year (" << model_->start_year() << "), or greater than final_year (" << model_->final_year() << "). Please fix this.";
  }
}

/**
 *
 */
void LeastSquares::PreExecute() {
/*  LOG_MEDIUM() << "PreExecute n obs = " << observed_.size() << " n covariates = " << covariates_.size();
  for (unsigned i = 0; i < observed_.size(); ++i)
    LOG_MEDIUM() <<  "observed = " << observed_[i] << " x = " << covariates_[i];*/
}


/**
 *
 */
void LeastSquares::DoReset() {
/*  LOG_MEDIUM() << "DoReset n obs = " << observed_.size() << " n covariates = " << covariates_.size();
  for (unsigned i = 0; i < observed_.size(); ++i)
    LOG_MEDIUM() <<  "observed = " << observed_[i] << " x = " << covariates_[i];*/
}
/**
 *
 */
void LeastSquares::Execute() {

  //LOG_MEDIUM() << "Entering observation " << label_;
  Double zero_val = 0.0;
  string category = "dummy";
  LOG_MEDIUM() << "a = " << a_ << " b = " << b_ << " year = " << model_->current_year();
  for (unsigned i = 0; i < observed_.size(); ++i) {
    y_hat_[i] = a_ + b_ * covariates_[i];
    LOG_MEDIUM() <<  y_hat_[i] << " observed = " << observed_[i] << " x = " << covariates_[i];
    SaveComparison(category, y_hat_[i], observed_[i], zero_val, sigma_ /  y_hat_[i], zero_val, delta_, zero_val);
  }
  //LOG_MEDIUM() << "Calculating score for observation = " << label_;


}

/**
 *
 */
void LeastSquares::CalculateScore() {
  LOG_MEDIUM() << "CalculateScore";
/*  for (unsigned i = 0; i < observed_.size(); ++i)
    LOG_MEDIUM() <<  "observed = " << observed_[i] << " x = " << covariates_[i];*/
  //LOG_MEDIUM() << "Entering CalculateScore " << label_;
  likelihood_->GetScores(comparisons_);
  //LOG_MEDIUM() << "save scores";


  for (unsigned year : years_) {
    scores_[year] = 0.0;
    for (obs::Comparison comparison : comparisons_[year]) {
      scores_[year] += comparison.score_;
    }
  }

/*  for (unsigned year : years_) {
    scores_[year] = likelihood_->GetInitialScore(comparisons_, year);
    for (obs::Comparison comparison : comparisons_[year]) {
      scores_[year] += comparison.score_;
    }
  }*/
}


} /* namespace age */
} /* namespace observations */
} /* namespace niwa */

