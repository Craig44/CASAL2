/**
 * @file UnConstrained.h
 * @author C.Marsh
 * @github
 * @date 26/6/2019
 * @section LICENSE
 *
 * Copyright NIWA Science ©2016 - www.niwa.co.nz
 *
 * @section DESCRIPTION
 *
 * This takes a single parameter with lower and upper bounds and transforms to -Inf -> Inf space.
 */
#ifndef SOURCE_ESTIMATETRANSFORMATIONS_CHILDREN_UNCONSTAINED_H_
#define SOURCE_ESTIMATETRANSFORMATIONS_CHILDREN_UNCONSTAINED_H_

// headers
#include "EstimateTransformations/EstimateTransformation.h"

// namespaces
namespace niwa {
class Estimate;
namespace estimatetransformations {

/**
 *
 */
class UnConstrained : public EstimateTransformation {
public:
  UnConstrained() = delete;
  explicit UnConstrained(Model* model);
  virtual ~UnConstrained() = default;
  void                        TransformForObjectiveFunction() override final;
  void                        RestoreFromObjectiveFunction() override final;
  std::set<string>            GetTargetEstimates() override final;
  Double                      GetScore() override final;

protected:
  // methods
  void                        DoValidate() override final;
  void                        DoBuild() override final;
  void                        DoTransform() override final;
  void                        DoRestore() override final;
};

} /* namespace estimatetransformations */
} /* namespace niwa */

#endif /* SOURCE_ESTIMATETRANSFORMATIONS_CHILDREN_UNCONSTAINED_H_ */
