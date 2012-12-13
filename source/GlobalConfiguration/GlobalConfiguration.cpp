/**
 * @file GlobalConfiguration.cpp
 * @author  Scott Rasmussen (scott.rasmussen@zaita.com)
 * @version 1.0
 * @date 18/09/2012
 * @section LICENSE
 *
 * Copyright NIWA Science �2012 - www.niwa.co.nz
 *
 * $Date: 2008-03-04 16:33:32 +1300 (Tue, 04 Mar 2008) $
 */

// Headers
#include "GlobalConfiguration.h"

#include "Utilities/Logging/Logging.h"

namespace isam {

/**
 * Default Constructor
 */
GlobalConfiguration::GlobalConfiguration() {

  parameters_[PARAM_DEBUG]            = "f";
  parameters_[PARAM_SKIP_CONFIG_FILE] = "f";
  parameters_[PARAM_CONFIG_FILE]      = "isam.txt";
}

/**
 * Destructor
 */
GlobalConfiguration::~GlobalConfiguration() {
}

/**
 * Standard singletone instance method
 *
 * @return static shared_ptr<> to the instance
 */
shared_ptr<GlobalConfiguration> GlobalConfiguration::Instance() {
  static GlobalConfigurationPtr instance = GlobalConfigurationPtr(new GlobalConfiguration());
  return instance;
}

/**
 * This accessor will return the debug flag.
 *
 * @return If we should run in debug mode of not
 */
bool GlobalConfiguration::debug_mode() {
  bool result = false;

  bool success = util::To<bool>(parameters_[PARAM_DEBUG], result);
  if (!success)
    LOG_CODE_ERROR("Could not convert the debug_mode value stored in global configuration to a boolean. The value was " << parameters_[PARAM_DEBUG]);

  return result;
}

/**
 * This accessor will return the flag to indicate
 * if we should skip loading a configuration file.
 *
 * This is primarily used when running unit tests and
 * we want to load the configuration from memory.
 *
 * @return flag indicating if we skip configuration file load or not
 */
bool GlobalConfiguration::skip_config_file() {
  bool result = false;

  bool success = util::To<bool>(parameters_[PARAM_SKIP_CONFIG_FILE], result);
  if (!success)
    LOG_CODE_ERROR("Could not convert skip_config_file value stored in global configuration to a boolean. The value was " << parameters_[PARAM_SKIP_CONFIG_FILE]);

  return result;
}

/**
 * This method will clear any previously set parameters within our global configuration.
 * All member objects need to be cleared during this method.
 */
void GlobalConfiguration::Clear() {
  parameters_.clear();
  command_line_parameters_.clear();
}

} /* namespace isam */
