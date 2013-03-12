/**
 * @file ObjectiveFunction.cpp
 * @author  Scott Rasmussen (scott.rasmussen@zaita.com)
 * @version 1.0
 * @date 21/02/2013
 * @section LICENSE
 *
 * Copyright NIWA Science �2013 - www.niwa.co.nz
 *
 * $Date: 2008-03-04 16:33:32 +1300 (Tue, 04 Mar 2008) $
 */

// Headers
#include "ObjectiveFunction.h"

#include "Estimates/Manager.h"
#include "Penalties/Manager.h"

// Namespaces
namespace isam {

/**
 * Default constructor
 */
ObjectiveFunction::ObjectiveFunction() {
}

/**
 * Singleton instance method
 */
ObjectiveFunction& ObjectiveFunction::Instance() {
  static ObjectiveFunction singleton;
  return singleton;
}

/**
 * Calculate our score for the current run
 */
void ObjectiveFunction::CalculateScore() {
  score_list_.clear();

  /**
   * Get the scores from each of the observations/likelihoods
   */


  /**
   * Get the scores from each of the penalties
   */
  const vector<penalties::Info>& penalties = penalties::Manager::Instance().flagged_penalties();
  for (penalties::Info penalty : penalties) {
    objective::Score new_score;

    new_score.label_ = PARAM_PENALTY + string("->") + penalty.label_;
    new_score.score_ = penalty.score_;

    score_list_.push_back(new_score);
    score_ += new_score.score_;
  }

  /**
   * Get the scores from each of the priors
   */
  vector<EstimatePtr> estimates = estimates::Manager::Instance().GetObjects();
  for (EstimatePtr estimate : estimates) {
    if (!estimate->prior())
      continue;

    objective::Score new_score;
    new_score.label_ = PARAM_PRIOR + string("->") + estimate->parameter();
    new_score.score_ = estimate->GetPriorScore();

    score_list_.push_back(new_score);
    score_ += new_score.score_;
  }
}

} /* namespace isam */
