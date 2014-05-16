/**
 * @file CasalComplex1.Test.cpp
 * @author  Scott Rasmussen (scott.rasmussen@zaita.com)
 * @date 14/03/2014
 * @section LICENSE
 *
 * Copyright NIWA Science �2014 - www.niwa.co.nz
 *
 */
#ifdef TESTMODE

#include "CasalComplex1.h"

#include "Model/Model.h"
#include "ObjectiveFunction/ObjectiveFunction.h"
#include "Observations/Manager.h"
#include "Observations/Observation.h"
#include "TestResources/TestFixtures/InternalEmptyModel.h"

// Namespaces
namespace isam {
namespace testcases {

using std::cout;
using std::endl;
using isam::testfixtures::InternalEmptyModel;

/**
 *
 */
TEST_F(InternalEmptyModel, Model_CasalComplex1_BasicRun) {
  AddConfigurationLine(test_cases_casal_complex_1, "CasalComplex1.h", 31);
  LoadConfiguration();

  ModelPtr model = Model::Instance();
  model->Start(RunMode::kBasic);

  ObjectiveFunction& obj_function = ObjectiveFunction::Instance();
  EXPECT_DOUBLE_EQ(16420.52830668812, obj_function.score());
}

/**
 *
 */
TEST_F(InternalEmptyModel, Model_CasalComplex1_Estimation) {
  AddConfigurationLine(test_cases_casal_complex_1, "CasalComplex1.h", 31);
  LoadConfiguration();

  ModelPtr model = Model::Instance();
  model->Start(RunMode::kEstimation);

  ObjectiveFunction& obj_function = ObjectiveFunction::Instance();
  EXPECT_DOUBLE_EQ(463.46264220283479, obj_function.score());
}

/**
 *
 */
TEST_F(InternalEmptyModel, Model_CasalComplex1_Simulation) {
  AddConfigurationLine(test_cases_casal_complex_1, "CasalComplex1.h", 31);
  LoadConfiguration();

  ModelPtr model = Model::Instance();
  model->Start(RunMode::kSimulation);

  ObjectiveFunction& obj_function = ObjectiveFunction::Instance();
  EXPECT_DOUBLE_EQ(463.46264220283479, obj_function.score());

  ObservationPtr observation = observations::Manager::Instance().GetObservation("chatTANage1992");
  if (!observation && observation->label() != "chatTANage1992")
    LOG_ERROR("Observation chatTANage1992 could not be loaded for testing");

  map<unsigned, vector<obs::Comparison> >& comparisons = observation->comparisons();
  ASSERT_EQ(1u, comparisons.size());
  ASSERT_EQ(26u, comparisons[1992].size());

  EXPECT_DOUBLE_EQ(0.0033325927047311295, comparisons[1992][0].observed_);
  EXPECT_DOUBLE_EQ(0.010337720945175074,  comparisons[1992][1].observed_);
  EXPECT_DOUBLE_EQ(0.022200684697852937,  comparisons[1992][2].observed_);
  EXPECT_DOUBLE_EQ(0.03210354836461965,   comparisons[1992][3].observed_);
  EXPECT_DOUBLE_EQ(0.031475490187957762,  comparisons[1992][4].observed_);
  EXPECT_DOUBLE_EQ(0.065309640265429694,  comparisons[1992][5].observed_);
  EXPECT_DOUBLE_EQ(0.080539623046253006,  comparisons[1992][6].observed_);
  EXPECT_DOUBLE_EQ(0.16234263666897,      comparisons[1992][7].observed_);
  EXPECT_DOUBLE_EQ(0.097915145226750461,  comparisons[1992][8].observed_);
  EXPECT_DOUBLE_EQ(0.068854611441572358,  comparisons[1992][9].observed_);
}

} /* namespace testcases */
} /* namespace isam */


#endif
