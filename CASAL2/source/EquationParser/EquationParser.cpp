/**
 * @file fEquationParser.cpp
 * @author Scott Rasmussen (scott.rasmussen@zaita.com)
 * @github https://github.com/Zaita
 * @date 21/11/2014
 * @section LICENSE
 *
 * Copyright NIWA Science ©2017 - www.niwa.co.nz
 *
 */

// headers
#include "EquationParser.h"

#include "Model/Model.h"
#include "Model/Objects.h"

// namespaces
namespace niwa {

/**
 * Default constructor
 */
EquationParser::EquationParser(Model* model) : model_(model) {
#ifdef USE_ADOLC
  parser_ = new Parser<adouble, adub, const badouble&>(this);

  parser_->AddOneArgumentFunction("abs", fabs);
  parser_->AddOneArgumentFunction("acos", acos);
  parser_->AddOneArgumentFunction("asin", asin);
  parser_->AddOneArgumentFunction("atan", atan);
  parser_->AddOneArgumentFunction("ceil", ceil);
  parser_->AddOneArgumentFunction("cos", cos);
  parser_->AddOneArgumentFunction("cosh", cosh);
  parser_->AddOneArgumentFunction("exp", exp);
  parser_->AddOneArgumentFunction("floor", floor);
  parser_->AddOneArgumentFunction("log", log);
  parser_->AddOneArgumentFunction("log10", log10);
  parser_->AddOneArgumentFunction("sin", sin);
  parser_->AddOneArgumentFunction("sinh", sinh);
  parser_->AddOneArgumentFunction("sqrt", sqrt);
  parser_->AddOneArgumentFunction("tan", tan);
  parser_->AddOneArgumentFunction("tanh", tanh);

#elif USE_BETADIFF
  parser_ = new Parser<adouble, adub, const badouble&>(this);

//  parser_->AddOneArgumentFunction("abs", fabs);
//  parser_->AddOneArgumentFunction("acos", acos);
//  parser_->AddOneArgumentFunction("asin", asin);
//  parser_->AddOneArgumentFunction("atan", atan);
//  parser_->AddOneArgumentFunction("ceil", ceil);
//  parser_->AddOneArgumentFunction("cos", cos);
//  parser_->AddOneArgumentFunction("cosh", cosh);
//  parser_->AddOneArgumentFunction("exp", exp);
//  parser_->AddOneArgumentFunction("floor", floor);
//  parser_->AddOneArgumentFunction("log", log);
//  parser_->AddOneArgumentFunction("log10", log10);
//  parser_->AddOneArgumentFunction("sin", sin);
//  parser_->AddOneArgumentFunction("sinh", sinh);
//  parser_->AddOneArgumentFunction("sqrt", sqrt);
//  parser_->AddOneArgumentFunction("tan", tan);
//  parser_->AddOneArgumentFunction("tanh", tanh);

#elif USE_CPPAD
  parser_ = new Parser<CppAD::AD<double>, CppAD::AD<double>, const CppAD::AD<double>&>(this);



#elif USE_STAN
  parser_ = new Parser<stan::math::var, stan::math::var, const stan::math::var&>(this);

//  parser_->AddOneArgumentFunction("abs", fabs);
//  parser_->AddOneArgumentFunction("acos", acos);
//  parser_->AddOneArgumentFunction("asin", asin);
//  parser_->AddOneArgumentFunction("atan", atan);
//  parser_->AddOneArgumentFunction("ceil", ceil);
//  parser_->AddOneArgumentFunction("cos", cos);
//  parser_->AddOneArgumentFunction("cosh", cosh);
//  parser_->AddOneArgumentFunction("exp", exp);
//  parser_->AddOneArgumentFunction("floor", floor);
//  parser_->AddOneArgumentFunction("log", log);
//  parser_->AddOneArgumentFunction("log10", log10);
//  parser_->AddOneArgumentFunction("sin", sin);
//  parser_->AddOneArgumentFunction("sinh", sinh);
//  parser_->AddOneArgumentFunction("sqrt", sqrt);
//  parser_->AddOneArgumentFunction("tan", tan);
//  parser_->AddOneArgumentFunction("tanh", tanh);

#else
  parser_ = new Parser<Double, Double, Double>(this);

  parser_->AddOneArgumentFunction("abs", fabs);
  parser_->AddOneArgumentFunction("acos", acos);
  parser_->AddOneArgumentFunction("asin", asin);
  parser_->AddOneArgumentFunction("atan", atan);
  parser_->AddOneArgumentFunction("ceil", ceil);
  parser_->AddOneArgumentFunction("cos", cos);
  parser_->AddOneArgumentFunction("cosh", cosh);
  parser_->AddOneArgumentFunction("exp", exp);
  parser_->AddOneArgumentFunction("floor", floor);
  parser_->AddOneArgumentFunction("log", log);
  parser_->AddOneArgumentFunction("log10", log10);
  parser_->AddOneArgumentFunction("sin", sin);
  parser_->AddOneArgumentFunction("sinh", sinh);
  parser_->AddOneArgumentFunction("sqrt", sqrt);
  parser_->AddOneArgumentFunction("tan", tan);
  parser_->AddOneArgumentFunction("tanh", tanh);
#endif
}

/**
 *
 */
EquationParser::~EquationParser() {
  if (parser_ != nullptr)
    delete parser_;
}

/**
 * This method will try to find the registered
 * addressable/object that we're looking for
 *
 * @param name The name (e.g process[recruitment].r0
 * @return Reference to the double value (not const)
 */
Double& EquationParser::LookupValue(const std::string& name) {
  LOG_FINEST() << "Equation lookup on value: " << name;

  string error = "";
  if (!model_->objects().VerfiyAddressableForUse(name, addressable::kLookup, error)) {
    LOG_FATAL() << "addressable "<< name << " could not be verified for use in equation_parser. Error was " << error;
  }

  Double* value = model_->objects().GetAddressable(name);
  LOG_FINEST() << name << ".value: " << AS_DOUBLE((*value));
  return *value;
}


/**
 *
 */
Double EquationParser::Parse(string equation) {
  return parser_->Evaluate(equation);
}


} /* namespace niwa */
