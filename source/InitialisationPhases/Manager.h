/*
 * Manager.h
 *
 *  Created on: 13/12/2012
 *      Author: Admin
 */

#ifndef INITIALISATIONPHASES_MANAGER_H_
#define INITIALISATIONPHASES_MANAGER_H_

// Headers
#include "BaseClasses/Manager.h"
#include "InitialisationPhases/InitialisationPhase.h"

// Namespaces
namespace isam {
namespace initialisationphases {

/**
 * Class Definition
 */
class Manager : public base::Manager<isam::initialisationphases::Manager, isam::InitialisationPhase> {
public:
  Manager();
  virtual ~Manager() noexcept(true);
};

} /* namespace initialisationphases */
} /* namespace isam */
#endif /* INITIALISATIONPHASES_MANAGER_H_ */
