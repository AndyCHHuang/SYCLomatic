//===--- Rules.cpp -------------------------------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//
#include "ASTTraversal.h"
#include "CallExprRewriter.h"
#include "Rules.h"
#include "Utility.h"
#include "Error.h"
#include "MapNames.h"
#include "llvm/Support/YAMLTraits.h"
std::vector<std::string> MetaRuleObject::RuleFiles;
std::vector<MetaRuleObject> MetaRules;

void registerMacroRule(MetaRuleObject &R) {
  auto It = MapNames::MacroRuleMap.find(R.In);
  if (It != MapNames::MacroRuleMap.end()) {
    if (It->second.Priority > R.Priority) {
      It->second.Id = R.RuleId;
      It->second.Priority = R.Priority;
      It->second.In = R.In;
      It->second.Out = R.Out;
      It->second.HelperFeature =
        clang::dpct::HelperFeatureEnum::no_feature_helper;
      It->second.Includes = R.Includes;
    }
  } else {
    MapNames::MacroRuleMap.emplace(
        R.In,
        MacroMigrationRule(R.RuleId, R.Priority, R.In, R.Out,
                           clang::dpct::HelperFeatureEnum::no_feature_helper,
                           R.Includes));
  }
}

void registerAPIRule(MetaRuleObject &R) {
  // register rule
  clang::dpct::ASTTraversalMetaInfo::registerRule((char *)&R, R.RuleId, [=] {
    return new clang::dpct::UserDefinedAPIRule(R.In);
  });
  // create and register rewriter
  // RewriterMap contains entries like {"FunctionName", RewriterFactory}
  // The priority of all the default rules are RulePriority::Fallback and
  // recorded in the RewriterFactory.
  // Search for existing rules of the same function name
  // if there is existing rule,
  //   compare the priority and decide whether to add the rule into the
  //   RewriterMap
  // if there is no existing rule,
  //   add the new rule to the RewriterMap
  auto It =
      clang::dpct::CallExprRewriterFactoryBase::RewriterMap->find(R.In);
  if (It == clang::dpct::CallExprRewriterFactoryBase::RewriterMap->end()) {
    clang::dpct::CallExprRewriterFactoryBase::RewriterMap->emplace(
        R.In,
        clang::dpct::createUserDefinedRewriterFactory(R.In, R.Out, R.Priority));
  } else if (It->second->Priority > R.Priority) {
    (*clang::dpct::CallExprRewriterFactoryBase::RewriterMap)[R.In] =
        clang::dpct::createUserDefinedRewriterFactory(R.In, R.Out, R.Priority);
  }
}

void importRules(llvm::cl::list<std::string> &RuleFiles) {
  for (auto &RuleFile : RuleFiles) {
    makeCanonical(RuleFile);
    // open the yaml file
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Buffer =
        llvm::MemoryBuffer::getFile(RuleFile);
    if (!Buffer) {
      llvm::errs() << "error: failed to read " << RuleFile << ": "
                   << Buffer.getError().message() << "\n";
      clang::dpct::ShowStatus(MigrationErrorInvalidRuleFilePath);
      dpctExit(MigrationErrorInvalidRuleFilePath);
    }

    // load rules
    llvm::yaml::Input YAMLIn(Buffer.get()->getBuffer());
    YAMLIn >> MetaRules;

    if (YAMLIn.error()) {
      // yaml parsing fail
      clang::dpct::ShowStatus(MigrationErrorCannotParseRuleFile);
      dpctExit(MigrationErrorCannotParseRuleFile);
    }

    MetaRuleObject::setRuleFiles(RuleFile);

    //Register Rules
    for (MetaRuleObject &r : MetaRules) {
      switch (r.Kind) {
      case (RuleKind::Macro):
        registerMacroRule(r);
        break;
      case (RuleKind::API):
        registerAPIRule(r);
        break;
      default:
        break;
      }
    }
  }
}

// /RuleOutputString is the string specified in rule's "Out" session
void OutputBuilder::parse(std::string &RuleOutputString) {
  int StrStartIdx = 0;
  size_t i;
  for (i = 0; i < RuleOutputString.length(); i++) {
    switch (RuleOutputString[i]) {
    case '\\': {
      // save previous string
      auto StringBuilder = std::make_shared<OutputBuilder>();
      StringBuilder->Kind = Kind::String;
      StringBuilder->Str =
          RuleOutputString.substr(StrStartIdx, i - StrStartIdx);
      SubBuilders.push_back(StringBuilder);
      // skip "/" and set the begin of the next string
      i++;
      StrStartIdx = i;
      break;
    }
    case '$':
    {
      // save previous string
      auto StringBuilder = std::make_shared<OutputBuilder>();
      StringBuilder->Kind = Kind::String;
      StringBuilder->Str =
          RuleOutputString.substr(StrStartIdx, i - StrStartIdx);
      SubBuilders.push_back(StringBuilder);
      SubBuilders.push_back(consumeKeyword(RuleOutputString, i));
      StrStartIdx = i;
    }
    break;
    default:
      break;
    }
  }
  if (i > StrStartIdx) {
    auto StringBuilder = std::make_shared<OutputBuilder>();
    StringBuilder->Kind = Kind::String;
    StringBuilder->Str = RuleOutputString.substr(StrStartIdx, i - StrStartIdx);
    SubBuilders.push_back(StringBuilder);
  }
}

// /OutStr is the string specified in rule's "Out" session
void OutputBuilder::ignoreWhitespaces(std::string &OutStr, size_t &Idx) {
  for (; Idx < OutStr.size(); Idx++) {
    if (OutStr[Idx] != ' ' && OutStr[Idx] != '\r' && OutStr[Idx] != '\n' &&
      OutStr[Idx] != '\t') {
      return;
    }
  }
}

// /OutStr is the string specified in rule's "Out" session
void OutputBuilder::consumeRParen(std::string &OutStr, size_t &Idx) {
  ignoreWhitespaces(OutStr, Idx);
  if (OutStr[Idx] != ')') {
    llvm::errs() << "error: expecting an RParen around "
      << OutStr.substr(Idx, 10) << "\n";
    clang::dpct::ShowStatus(MigrationErrorCannotParseRuleFile);
    dpctExit(MigrationErrorCannotParseRuleFile);
  } else {
    Idx++;
  }
}



// /OutStr is the string specified in rule's "Out" session
void OutputBuilder::consumeLParen(std::string &OutStr, size_t &Idx) {
  ignoreWhitespaces(OutStr, Idx);
  if (OutStr[Idx] != '(') {
    llvm::errs() << "error: expecting an LParen around "
      << OutStr.substr(Idx, 10) << "\n";
    clang::dpct::ShowStatus(MigrationErrorCannotParseRuleFile);
    dpctExit(MigrationErrorCannotParseRuleFile);
  }
  else {
    Idx++;
  }
}

// /OutStr is the string specified in rule's "Out" session
int OutputBuilder::consumeArgIndex(std::string &OutStr, size_t &Idx) {
  ignoreWhitespaces(OutStr, Idx);
  if (OutStr[Idx] != '$') {
    llvm::errs()
        << "error: expecting \'$\' followed by a positive integer around "
        << OutStr.substr(Idx, 10) << "\n";
    clang::dpct::ShowStatus(MigrationErrorCannotParseRuleFile);
    dpctExit(MigrationErrorCannotParseRuleFile);
  }
  // consume $
  Idx++;
  auto DollarSignIdx = Idx;
  ignoreWhitespaces(OutStr, Idx);
  int ArgIndex = 0;
  for (int i = Idx; i < OutStr.size(); i++) {
    if (!std::isdigit(OutStr[i])) {
      if (i == 0) {
        // report unknown KW
        llvm::errs() << "error: expecting a positive integer around "
          << OutStr.substr(i, 10) << "\n";
        clang::dpct::ShowStatus(MigrationErrorCannotParseRuleFile);
        dpctExit(MigrationErrorCannotParseRuleFile);
      } else {
        // process arg number
        std::string ArgNumStr = OutStr.substr(Idx, i - Idx);
        Idx = i;
        ArgIndex = std::stoi(ArgNumStr);
        break;
      }
    }
  }
  if (ArgIndex <= 0) {
    // report invalid ArgIndex
    llvm::errs() << "error: expecting a positive integer around "
      << OutStr.substr(DollarSignIdx, 10) << "\n";
    clang::dpct::ShowStatus(MigrationErrorCannotParseRuleFile);
    dpctExit(MigrationErrorCannotParseRuleFile);
  }
  // Adjust the index because the arg index in rules starts from $1,
  // and the arg index starts from 0 in CallExpr.
  return ArgIndex - 1;
}

// /OutStr is the string specified in rule's "Out" session
std::shared_ptr<OutputBuilder>
OutputBuilder::consumeKeyword(std::string &OutStr, size_t &Idx) {
  auto ResultBuilder = std::make_shared<OutputBuilder>();
  if (OutStr.substr(Idx, 6) == "$queue") {
    Idx += 6;
    ResultBuilder->Kind = Kind::Queue;
  } else if (OutStr.substr(Idx, 8) == "$context") {
    Idx += 8;
    ResultBuilder->Kind = Kind::Context;
  } else if (OutStr.substr(Idx, 7) == "$device") {
    Idx += 7;
    ResultBuilder->Kind = Kind::Device;
  } else if (OutStr.substr(Idx, 13) == "$type_name_of") {
    Idx += 13;
    consumeLParen(OutStr, Idx);
    ResultBuilder->Kind = Kind::TypeName;
    ResultBuilder->ArgIndex = consumeArgIndex(OutStr, Idx);
    consumeRParen(OutStr, Idx);
  } else if (OutStr.substr(Idx, 8) == "$addr_of") {
    Idx += 8;
    consumeLParen(OutStr, Idx);
    ResultBuilder->Kind = Kind::AddrOf;
    ResultBuilder->ArgIndex = consumeArgIndex(OutStr, Idx);
    consumeRParen(OutStr, Idx);
  } else if (OutStr.substr(Idx, 11) == "$deref_type") {
    Idx += 11;
    consumeLParen(OutStr, Idx);
    ResultBuilder->Kind = Kind::DerefedTypeName;
    ResultBuilder->ArgIndex = consumeArgIndex(OutStr, Idx);
    consumeRParen(OutStr, Idx);
  } else if (OutStr.substr(Idx, 6) == "$deref") {
    Idx += 6;
    consumeLParen(OutStr, Idx);
    ResultBuilder->Kind = Kind::Deref;
    ResultBuilder->ArgIndex = consumeArgIndex(OutStr, Idx);
    consumeRParen(OutStr, Idx);
  } else {
    ResultBuilder->Kind = Kind::Arg;
    ResultBuilder->ArgIndex = consumeArgIndex(OutStr, Idx);
  }
  return ResultBuilder;
}

using namespace clang::ast_matchers;

void clang::dpct::UserDefinedAPIRule::registerMatcher(
    clang::ast_matchers::MatchFinder &MF) {
  MF.addMatcher(callExpr(callee(functionDecl(hasName(APIName)))).bind("call"),
                this);
}

void clang::dpct::UserDefinedAPIRule::runRule(
    const clang::ast_matchers::MatchFinder::MatchResult &Result) {
  if (const CallExpr *CE =
    getAssistNodeAsType<CallExpr>(
      Result, "call")) {
    auto &SM = DpctGlobalInfo::getSourceManager();
    dpct::ExprAnalysis EA;
    EA.analyze(CE);
    auto Range = getDefinitionRange(CE->getBeginLoc(), CE->getEndLoc());
    auto Len = Lexer::MeasureTokenLength(
      Range.getEnd(), SM, DpctGlobalInfo::getContext().getLangOpts());
    Len += SM.getDecomposedLoc(Range.getEnd()).second -
      SM.getDecomposedLoc(Range.getBegin()).second;
    auto ReplStr = EA.getReplacedString();
    emplaceTransformation(
      new ReplaceText(Range.getBegin(), Len, std::move(ReplStr)));
  }
}