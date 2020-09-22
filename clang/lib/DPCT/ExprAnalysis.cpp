//===--- ExprAnalysis.cpp -----------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2020 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "ExprAnalysis.h"

#include "ASTTraversal.h"
#include "AnalysisInfo.h"
#include "CallExprRewriter.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprConcepts.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/ExprOpenMP.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtGraphTraits.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/StmtOpenMP.h"

namespace clang {
namespace dpct {

#define ANALYZE_EXPR(EXPR)                                                     \
  case Stmt::EXPR##Class:                                                      \
    return analyzeExpr(static_cast<const EXPR *>(Expression));

std::map<const Expr *, std::string> ArgumentAnalysis::DefaultArgMap;

void TemplateDependentReplacement::replace(
    const std::vector<TemplateArgumentInfo> &TemplateList) {
  SourceStr.replace(Offset, Length,
                    getTargetArgument(TemplateList).getString());
}

TemplateDependentStringInfo::TemplateDependentStringInfo(
    const std::string &SrcStr,
    const std::map<size_t, std::shared_ptr<TemplateDependentReplacement>>
        &InTDRs)
    : SourceStr(SrcStr) {
  for (auto TDR : InTDRs)
    TDRs.emplace_back(TDR.second->alterSource(SourceStr));
}

std::shared_ptr<TemplateDependentStringInfo>
TemplateDependentStringInfo::applyTemplateArguments(
    const std::vector<TemplateArgumentInfo> &TemplateList) {
  std::shared_ptr<TemplateDependentStringInfo> Result =
      std::make_shared<TemplateDependentStringInfo>();
  Result->SourceStr = SourceStr;
  int OffsetShift = 0;
  auto &Repls = Result->TDRs;
  auto &Str = Result->SourceStr;
  Result->IsDependOnWrittenArgument = true;
  for (auto &R : TDRs) {
    size_t ReplsSize = Repls.size();
    size_t ApplyOffset = R->getOffset() + OffsetShift;
    auto &TargetArg = R->getTargetArgument(TemplateList);
    auto &TargetList = TargetArg.getDependentStringInfo()->TDRs;
    Repls.resize(ReplsSize + TargetList.size());

    Str.replace(ApplyOffset, R->getLength(), TargetArg.getString());
    std::transform(TargetList.begin(), TargetList.end(),
                   Repls.begin() + ReplsSize,
                   [&](std::shared_ptr<TemplateDependentReplacement> OldRepl)
                       -> std::shared_ptr<TemplateDependentReplacement> {
                     auto Repl = OldRepl->alterSource(Str);
                     Repl->shift(ApplyOffset);
                     return Repl;
                   });
    OffsetShift += TargetArg.getString().length() - R->getLength();
    Result->IsDependOnWrittenArgument &= TargetArg.isWritten();
  }
  return Result;
}
const TemplateArgumentInfo &TemplateDependentReplacement::getTargetArgument(
    const std::vector<TemplateArgumentInfo> &TemplateList) {
  if (TemplateIndex < TemplateList.size())
    return TemplateList[TemplateIndex];
  static TemplateArgumentInfo TAI;
  return TAI;
}

SourceLocation ExprAnalysis::getExprLocation(SourceLocation Loc) {
  if (SM.isMacroArgExpansion(Loc))
    return getExprLocation(SM.getImmediateSpellingLoc(Loc));
  else
    return SM.getExpansionLoc(Loc);
}

std::pair<SourceLocation, size_t>
ExprAnalysis::getSpellingOffsetAndLength(SourceLocation Loc) {
  if (Loc.isInvalid())
    return std::pair<SourceLocation, size_t>(Loc, SrcLength);
  Loc = SM.getSpellingLoc(Loc);
  return std::pair<SourceLocation, size_t>(
      Loc,
      Lexer::MeasureTokenLength(Loc, SM, Context.getLangOpts()));
}

std::pair<SourceLocation, size_t>
ExprAnalysis::getSpellingOffsetAndLength(SourceLocation BeginLoc,
                                         SourceLocation EndLoc) {
  if (EndLoc.isValid()) {
    auto Begin = SM.getSpellingLoc(BeginLoc);
    auto End = getSpellingOffsetAndLength(EndLoc);
    return std::pair<SourceLocation, size_t>(
        Begin, SM.getCharacterData(End.first) - SM.getCharacterData(Begin) +
                   End.second);
  }
  return getSpellingOffsetAndLength(BeginLoc);
}

std::pair<SourceLocation, size_t> ExprAnalysis::getSpellingOffsetAndLength(const Expr *E) {
  SourceLocation SpellingBeginLoc, SpellingEndLoc;
  SpellingBeginLoc = SM.getSpellingLoc(E->getBeginLoc());
  SpellingEndLoc = SM.getSpellingLoc(E->getEndLoc());

  // Both Begin/End are not macro
  // or
  // SpellingBeginLoc and SpellingEndLoc are in the same macro define
  if (!isExprStraddle(E)) {
    auto LastTokenLength =
        Lexer::MeasureTokenLength(SpellingEndLoc, SM, Context.getLangOpts());
    return std::pair<SourceLocation, size_t>(
        SpellingBeginLoc, SM.getCharacterData(SpellingEndLoc) -
                              SM.getCharacterData(SpellingBeginLoc) +
                              LastTokenLength);
  }

  // If the expr is straddle, use the expansion range
  auto ExpansionBeginLoc = SM.getExpansionLoc(E->getBeginLoc());
  auto ExpansionEndLoc = SM.getExpansionLoc(E->getEndLoc());
  auto LastTokenLength =
    Lexer::MeasureTokenLength(ExpansionEndLoc, SM, Context.getLangOpts());
  return std::pair<SourceLocation, size_t>(
    ExpansionBeginLoc, SM.getCharacterData(ExpansionEndLoc) -
    SM.getCharacterData(ExpansionBeginLoc) +
    LastTokenLength);
}


std::pair<size_t, size_t> ExprAnalysis::getOffsetAndLength(SourceLocation Loc) {
  if (Loc.isInvalid())
    return std::pair<size_t, size_t>(0, SrcLength);
  Loc = getExprLocation(Loc);
  FileId = SM.getDecomposedLoc(Loc).first;
  return std::pair<size_t, size_t>(
      getOffset(Loc),
      Lexer::MeasureTokenLength(Loc, SM, Context.getLangOpts()));
}

std::pair<size_t, size_t>
ExprAnalysis::getOffsetAndLength(SourceLocation BeginLoc,
                                 SourceLocation EndLoc) {
  if (EndLoc.isValid()) {
    auto Begin = getOffset(getExprLocation(BeginLoc));
    auto End = getOffsetAndLength(EndLoc);
    return std::pair<size_t, size_t>(Begin, End.first - Begin + End.second);
  }
  return getOffsetAndLength(BeginLoc);
}

std::pair<size_t, size_t>
ExprAnalysis::getOffsetAndLength(SourceLocation BeginLoc, SourceLocation EndLoc,
                                 const Expr *Parent) {
  const std::shared_ptr<ast_type_traits::DynTypedNode> P =
      std::make_shared<ast_type_traits::DynTypedNode>(
          ast_type_traits::DynTypedNode::create(*Parent));
  if (BeginLoc.isMacroID() && isInsideFunctionLikeMacro(BeginLoc, EndLoc, P)) {
    BeginLoc = SM.getExpansionLoc(SM.getImmediateSpellingLoc(BeginLoc));
    EndLoc = SM.getExpansionLoc(SM.getImmediateSpellingLoc(EndLoc));
  } else {
    if (EndLoc.isValid()) {
      BeginLoc = SM.getExpansionRange(BeginLoc).getBegin();
      EndLoc = SM.getExpansionRange(EndLoc).getEnd();
    }
  }
  // Calculate offset and length from SourceLocation
  auto End = getOffset(EndLoc);
  auto LastTokenLength =
      Lexer::MeasureTokenLength(EndLoc, SM, Context.getLangOpts());

  auto DecompLoc = SM.getDecomposedLoc(BeginLoc);
  FileId = DecompLoc.first;
  // The offset of Expr used in ExprAnalysis is related to SrcBegin not
  // FileBegin
  auto Begin = DecompLoc.second - SrcBegin;
  return std::pair<size_t, size_t>(Begin, End - Begin + LastTokenLength);
}

std::pair<size_t, size_t> ExprAnalysis::getOffsetAndLength(const Expr *E) {
  SourceLocation BeginLoc, EndLoc;
  // if the parent expr is inside macro and current expr is macro arg expansion,
  // use the expansion location of the macro arg in the macro definition.
  if (IsInMacroDefine) {
    if (SM.isMacroArgExpansion(E->getBeginLoc())) {
      BeginLoc = SM.getSpellingLoc(
          SM.getImmediateExpansionRange(E->getBeginLoc()).getBegin());
      EndLoc = SM.getSpellingLoc(
          SM.getImmediateExpansionRange(E->getEndLoc()).getEnd());
    } else {
      BeginLoc =
          SM.getExpansionLoc(SM.getImmediateSpellingLoc(E->getBeginLoc()));
      EndLoc = SM.getExpansionLoc(SM.getImmediateSpellingLoc(E->getEndLoc()));
      if (isExprStraddle(E)) {
        auto Range = getTheOneBeforeLastImmediateExapansion(E->getBeginLoc(), E->getEndLoc());
        BeginLoc = SM.getImmediateSpellingLoc(Range.first);
        EndLoc = SM.getImmediateSpellingLoc(Range.second);
      }
    }
  } else if (E->getBeginLoc().isMacroID() && !isOuterMostMacro(E)) {
    // If E is not OuterMostMacro, use the spelling location
    BeginLoc = SM.getExpansionLoc(SM.getSpellingLoc(E->getBeginLoc()));
    EndLoc = SM.getExpansionLoc(SM.getSpellingLoc(E->getEndLoc()));
    if (isExprStraddle(E)) {
      auto Range = getTheOneBeforeLastImmediateExapansion(E->getBeginLoc(), E->getEndLoc());
      BeginLoc = SM.getImmediateSpellingLoc(Range.first);
      EndLoc = SM.getImmediateSpellingLoc(Range.second);
    }
  } else {
    // If E is the OuterMostMacro, use the expansion location
    BeginLoc = SM.getExpansionRange(E->getBeginLoc()).getBegin();
    EndLoc = SM.getExpansionRange(E->getEndLoc()).getEnd();
  }
  // Calculate offset and length from SourceLocation
  auto End = getOffset(EndLoc);
  auto LastTokenLength =
      Lexer::MeasureTokenLength(EndLoc, SM, Context.getLangOpts());

  // Find the begin/end location include prefix and postfix
  // Set Prefix and Postfix strings
  auto BeginLocWithoutPrefix = BeginLoc;
  BeginLoc = getBeginLocOfPreviousEmptyMacro(BeginLoc);
  auto RewritePrefixLength = SM.getCharacterData(BeginLocWithoutPrefix) -
                             SM.getCharacterData(BeginLoc);

  auto EndLocWithoutPostfix = EndLoc;
  EndLoc = getEndLocOfFollowingEmptyMacro(EndLoc);
  auto RewritePostfixLength =
      SM.getCharacterData(EndLoc) - SM.getCharacterData(EndLocWithoutPostfix);

  ExprBeginLoc = BeginLoc;
  ExprEndLoc = EndLoc;
  RewritePrefix =
    std::string(SM.getCharacterData(BeginLoc), RewritePrefixLength);

  // Get the token end of EndLocWithoutPostfix and EndLoc for correct
  // RewritePostfix
  EndLocWithoutPostfix =
    EndLocWithoutPostfix.getLocWithOffset(Lexer::MeasureTokenLength(
      EndLocWithoutPostfix, SM,
      dpct::DpctGlobalInfo::getContext().getLangOpts()));
  EndLoc = EndLoc.getLocWithOffset(Lexer::MeasureTokenLength(
      EndLoc, SM, dpct::DpctGlobalInfo::getContext().getLangOpts()));

  RewritePostfix = std::string(SM.getCharacterData(EndLocWithoutPostfix),
    RewritePostfixLength);

  auto DecompLoc = SM.getDecomposedLoc(BeginLoc);
  FileId = DecompLoc.first;
  // The offset of Expr used in ExprAnalysis is related to SrcBegin not
  // FileBegin
  auto Begin = DecompLoc.second - SrcBegin;
  return std::pair<size_t, size_t>(Begin, End - Begin + LastTokenLength);
}

void ExprAnalysis::initExpression(const Expr *Expression) {
  E = Expression;
  SrcBegin = 0;
  if (E && E->getBeginLoc().isValid()) {
    std::tie(SrcBegin, SrcLength) = getOffsetAndLength(E);
    ReplSet.init(
        std::string(SM.getBufferData(FileId).substr(SrcBegin, SrcLength)));
  } else {
    SrcLength = 0;
    ReplSet.init("");
  }
}

void ExprAnalysis::initSourceRange(const SourceRange &Range) {
  SrcBegin = 0;
  if (Range.getBegin().isValid()) {
    std::tie(SrcBegin, SrcLength) =
        getOffsetAndLength(Range.getBegin(), Range.getEnd());
    ReplSet.init(std::string(
        SM.getCharacterData(getExprLocation(Range.getBegin())), SrcLength));
  } else {
    SrcLength = 0;
    ReplSet.init("");
  }
}

void StringReplacements::replaceString() {
  SourceStr.reserve(SourceStr.length() + ShiftLength);
  auto Itr = ReplMap.rbegin();
  while (Itr != ReplMap.rend()) {
    Itr->second->replaceString();
    ++Itr;
  }
  ReplMap.clear();
}

ExprAnalysis::ExprAnalysis(const Expr *Expression)
    : Context(DpctGlobalInfo::getContext()),
      SM(DpctGlobalInfo::getSourceManager()) {
  analyze(Expression);
}

void ExprAnalysis::dispatch(const Stmt *Expression) {
  switch (Expression->getStmtClass()) {
#define STMT(CLASS, PARENT) ANALYZE_EXPR(CLASS)
#define STMT_RANGE(BASE, FIRST, LAST)
#define LAST_STMT_RANGE(BASE, FIRST, LAST)
#define ABSTRACT_STMT(STMT)
#include "clang/AST/StmtNodes.inc"
  default:
    return;
  }
}

void ExprAnalysis::analyzeExpr(const DeclRefExpr *DRE) {
  std::string CTSName;
  auto Qualifier = DRE->getQualifier();
  if (Qualifier) {
    // To handle class template specializations,
    // e.g: template<> class numeric_limits<int>.
    if (Qualifier->getKind() == clang::NestedNameSpecifier::TypeSpec) {
      auto CTSDecl = dyn_cast<ClassTemplateSpecializationDecl>(
          DRE->getDecl()->getDeclContext());
      if (CTSDecl) {
        CTSName =
            CTSDecl->getTypeForDecl()->getAsCXXRecordDecl()->getNameAsString();
        CTSName += "::" + DRE->getNameInfo().getAsString();
      }
    }
  }
  if (!CTSName.empty()) {
    RefString = CTSName;
  } else {
    RefString = DRE->getNameInfo().getAsString();
  }

  if (auto TemplateDecl = dyn_cast<NonTypeTemplateParmDecl>(DRE->getDecl()))
    addReplacement(DRE, TemplateDecl->getIndex());
  else if (auto ECD = dyn_cast<EnumConstantDecl>(DRE->getDecl())) {
    auto &ReplEnum = MapNames::findReplacedName(EnumConstantRule::EnumNamesMap,
                                                ECD->getName().str());
    if (!ReplEnum.empty())
      addReplacement(DRE, ReplEnum);
    else {
      auto &ReplBLASEnum = MapNames::findReplacedName(MapNames::BLASEnumsMap,
                                                      ECD->getName().str());
      if (!ReplBLASEnum.empty())
        addReplacement(DRE, ReplBLASEnum);
    }
  }
}

void ExprAnalysis::analyzeExpr(const CXXConstructExpr *Ctor) {
  if (Ctor->getConstructor()->getDeclName().getAsString() == "dim3") {
    std::string ArgsString;
    llvm::raw_string_ostream OS(ArgsString);
    DpctGlobalInfo::printCtadClass(OS, MapNames::getClNamespace() + "::range",
                                   3)
        << "(";
    ArgumentAnalysis A;
    std::string ArgStr = "";
    for (auto Arg : Ctor->arguments()) {
      A.analyze(Arg);
      ArgStr = ", " + A.getReplacedString() + ArgStr;
    }
    ArgStr.replace(0, 2, "");
    OS << ArgStr << ")";
    OS.flush();

    // Special handling for implicit ctor.
    // #define GET_BLOCKS(a) a
    // dim3 A = GET_BLOCKS(1);
    // Result if using SM.getExpansionRange:
    //   sycl::range<3> A = sycl::range<3>(1, 1, GET_BLOCKS(1));
    // Result if using addReplacement(E):
    //   #define GET_BLOCKS(a) sycl::range<3>(1, 1, a)
    //   sycl::range<3> A = GET_BLOCKS(1);
    if (Ctor->getParenOrBraceRange().isInvalid() && isOuterMostMacro(Ctor)) {
      return addReplacement(
          SM.getExpansionRange(Ctor->getBeginLoc()).getBegin(),
          SM.getExpansionRange(Ctor->getEndLoc()).getEnd(),
          ArgsString);
    }
    addReplacement(Ctor, ArgsString);
  }
}

void ExprAnalysis::analyzeExpr(const MemberExpr *ME) {
  static MapNames::MapTy NdItemMemberMap{{"__fetch_builtin_x", "2"},
                                         {"__fetch_builtin_y", "1"},
                                         {"__fetch_builtin_z", "0"}};
  static const MapNames::MapTy NdItemMap{
      {"__cuda_builtin_blockIdx_t", "get_group"},
      {"__cuda_builtin_gridDim_t", "get_group_range"},
      {"__cuda_builtin_blockDim_t", "get_local_range"},
      {"__cuda_builtin_threadIdx_t", "get_local_id"},
  };
  auto BaseType =
      DpctGlobalInfo::getUnqualifiedTypeName(ME->getBase()->getType());
  auto ItemItr = NdItemMap.find(BaseType);
  if (ItemItr != NdItemMap.end()) {
    std::string FieldName = ME->getMemberDecl()->getName().str();
    if (MapNames::replaceName(NdItemMemberMap, FieldName)) {
      addReplacement(ME, buildString(DpctGlobalInfo::getItemName(), ".",
                                     ItemItr->second, "(", FieldName, ")"));
    }
  } else if (BaseType == "dim3") {
    addReplacement(
        ME->getOperatorLoc(), ME->getMemberLoc(),
        MapNames::findReplacedName(MapNames::Dim3MemberNamesMap,
                                   ME->getMemberNameInfo().getAsString()));
  } else if (BaseType == "cudaDeviceProp") {
    std::string ReplacementStr = MapNames::findReplacedName(
        DevicePropVarRule::PropNamesMap, ME->getMemberNameInfo().getAsString());
    if (!ReplacementStr.empty()) {
      addReplacement(ME->getMemberLoc(), "get_" + ReplacementStr + "()");
    }
  } else if (BaseType == "textureReference") {
    std::string FieldName = ME->getMemberDecl()->getName().str();
    if (MapNames::replaceName(TextureRule::TextureMemberNames, FieldName)) {
      addReplacement(ME->getMemberLoc(), buildString("get_", FieldName, "()"));
    }
  } else if (MapNames::SupportedVectorTypes.find(BaseType) !=
             MapNames::SupportedVectorTypes.end()) {
    if (*BaseType.rbegin() == '1') {
      addReplacement(ME->getOperatorLoc(), ME->getEndLoc(), "");
    } else {
      std::string MemberName = ME->getMemberNameInfo().getAsString();
      if (MapNames::replaceName(MapNames::MemberNamesMap, MemberName)) {
        addReplacement(ME->getMemberLoc(), MemberName);
      }
    }
  }
  dispatch(ME->getBase());
}

void ExprAnalysis::analyzeExpr(const UnaryExprOrTypeTraitExpr *UETT) {
  if (UETT->getKind() == UnaryExprOrTypeTrait::UETT_SizeOf) {
    if (UETT->isArgumentType()) {
      analyzeType(UETT->getArgumentTypeInfo(), UETT);
    } else {
      analyzeExpr(UETT->getArgumentExpr());
    }
  }
}

void ExprAnalysis::analyzeExpr(const CStyleCastExpr *Cast) {
  if (Cast->getCastKind() == CastKind::CK_BitCast ||
      Cast->getCastKind() == CastKind::CK_IntegralCast) {
    analyzeType(Cast->getTypeInfoAsWritten(), Cast);
  }
  dispatch(Cast->getSubExpr());
}

// Precondition: CE != nullptr
void ExprAnalysis::analyzeExpr(const CallExpr *CE) {
  // To set the RefString
  dispatch(CE->getCallee());
  // If the callee requires rewrite, get the rewriter
  auto Itr = CallExprRewriterFactoryBase::RewriterMap.find(RefString);
  if (Itr != CallExprRewriterFactoryBase::RewriterMap.end()) {
    auto Result = Itr->second->create(CE)->rewrite();
    if (Result.hasValue()) {
      addReplacement(CE, Result.getValue());
    }
  } else {
    // If the callee does not need rewrite, analyze the args
    for (auto Arg : CE->arguments())
      analyzeArgument(Arg);
  }
}

void ExprAnalysis::analyzeExpr(const CXXNamedCastExpr *NCE) {
  analyzeType(NCE->getTypeInfoAsWritten(), NCE);
  dispatch(NCE->getSubExprAsWritten());
}

void ExprAnalysis::analyzeType(TypeLoc TL, const Expr *CSCE) {
  SourceRange SR = TL.getSourceRange();
  std::string TyName;

  auto ETL = TL.getAs<ElaboratedTypeLoc>();
  if (ETL) {
    auto QualifierLoc = ETL.getQualifierLoc();
    TL = ETL.getNamedTypeLoc();
    TyName = getNestedNameSpecifierString(QualifierLoc);
    if (ETL.getTypePtr()->getKeyword() == ETK_Typename) {
      if (QualifierLoc)
        SR.setBegin(QualifierLoc.getBeginLoc());
      else
        SR = TL.getSourceRange();
    }
  }

#define TYPELOC_CAST(Target) static_cast<const Target &>(TL)
  switch (TL.getTypeLocClass()) {
  case TypeLoc::Qualified:
    return analyzeType(TL.getUnqualifiedLoc(), CSCE);
  case TypeLoc::LValueReference:
  case TypeLoc::RValueReference:
    return analyzeType(TYPELOC_CAST(ReferenceTypeLoc).getPointeeLoc(), CSCE);
  case TypeLoc::Pointer:
    return analyzeType(TYPELOC_CAST(PointerTypeLoc).getPointeeLoc(), CSCE);
  case TypeLoc::Typedef:
    TyName +=
        TYPELOC_CAST(TypedefTypeLoc).getTypedefNameDecl()->getName().str();
    break;
  case TypeLoc::Builtin:
  case TypeLoc::Record:
    TyName += DpctGlobalInfo::getTypeName(TL.getType());
    break;
  case TypeLoc::TemplateTypeParm:
    if (auto D = TYPELOC_CAST(TemplateTypeParmTypeLoc).getDecl()) {
      return addReplacement(TL.getBeginLoc(), TL.getEndLoc(), CSCE,
                            D->getIndex());
    } else {
      return;
    }
  case TypeLoc::TemplateSpecialization: {
    llvm::raw_string_ostream OS(TyName);
    auto &TSTL = TYPELOC_CAST(TemplateSpecializationTypeLoc);
    TSTL.getTypePtr()->getTemplateName().print(OS, Context.getPrintingPolicy());
    SR.setEnd(TSTL.getTemplateNameLoc());
    analyzeTemplateSpecializationType(TSTL);
    break;
  }
  case TypeLoc::DependentTemplateSpecialization:
    analyzeTemplateSpecializationType(
        TYPELOC_CAST(DependentTemplateSpecializationTypeLoc));
    break;
  default:
    return;
  }
  if (MapNames::replaceName(MapNames::TypeNamesMap, TyName))
    addReplacement(SR.getBegin(), SR.getEnd(), CSCE, TyName);
}

void ExprAnalysis::analyzeTemplateArgument(const TemplateArgumentLoc &TAL) {
  switch (TAL.getArgument().getKind()) {
  case TemplateArgument::Type:
    return analyzeType(TAL.getTypeSourceInfo());
  case TemplateArgument::Expression:
    return dispatch(TAL.getSourceExpression());
  case TemplateArgument::Integral:
    return dispatch(TAL.getSourceIntegralExpression());
  case TemplateArgument::Declaration:
    return dispatch(TAL.getSourceDeclExpression());
  default:
    break;
  }
}

void ExprAnalysis::applyAllSubExprRepl() {
  for (std::shared_ptr<ExtReplacement> Repl : SubExprRepl) {
    DpctGlobalInfo::getInstance().addReplacement(Repl);
  }
}


const std::string &ArgumentAnalysis::getDefaultArgument(const Expr *E) {
  auto &Str = DefaultArgMap[E];
  if (Str.empty())
    Str = getStmtSpelling(E);
  return Str;
}

void KernelArgumentAnalysis::dispatch(const Stmt *Expression) {
  switch (Expression->getStmtClass()) {
    ANALYZE_EXPR(DeclRefExpr)
    ANALYZE_EXPR(MemberExpr)
    ANALYZE_EXPR(CXXMemberCallExpr)
    ANALYZE_EXPR(CallExpr)
    ANALYZE_EXPR(ArraySubscriptExpr)
    ANALYZE_EXPR(UnaryOperator)
  default:
    return ExprAnalysis::dispatch(Expression);
  }
}

int64_t
KernelConfigAnalysis::calculateWorkgroupSize(const CXXConstructExpr *Ctor) {
  int64_t Size = 1;
  auto Num = Ctor->getNumArgs();
  for (size_t i = 0; i < Num; ++i) {
    if (Ctor->getArg(i)->isDefaultArgument()) {
      return Size;
    }

    Expr::EvalResult ER;
    if (!Ctor->getArg(i)->isValueDependent() &&
        Ctor->getArg(i)->EvaluateAsInt(ER, DpctGlobalInfo::getContext())) {
      int64_t Value = ER.Val.getInt().getExtValue();
      Size = Size * Value;
    } else {
      // Not all args can be evaluated, so return a value larger than 256 to
      // emit warning.
      return 265 + 1;
    }
  }
  return Size;
}

void KernelArgumentAnalysis::analyzeExpr(const DeclRefExpr *DRE) {
  if (DRE->getType()->isReferenceType()) {
    IsRedeclareRequired = true;
  } else if (!DRE->getDecl()->isInLocalScope()) {
    IsRedeclareRequired = true;
  } else if (auto VD = dyn_cast<VarDecl>(DRE->getDecl())) {
    bool PreviousFlag = IsRedeclareRequired;
    if (VD->getStorageClass() == SC_Static) {
      IsRedeclareRequired = true;
      // exclude const variable with zero-init and const-init
      if (VD->getType().isConstQualified()) {
        if (VD->hasInit() && VD->checkInitIsICE()) {
          IsRedeclareRequired = PreviousFlag;
        }
      }
    }
  }

  // The VarDecl in MemVarInfo are matched in MemVarRule, which only matches
  // variables on device. They are migrated to objects, so need add get_ptr() by
  // setting IsDefinedOnDevice flag.
  if (auto VD = dyn_cast<VarDecl>(DRE->getDecl())) {
    if (auto Var = DpctGlobalInfo::getInstance().findMemVarInfo(VD)) {
      IsDefinedOnDevice = true;
      IsRedeclareRequired = true;
      if (!IsAddrOf && !VD->getType()->isArrayType()) {
        addReplacement(Lexer::getLocForEndOfToken(
                           DRE->getEndLoc(), 0, SM,
                           DpctGlobalInfo::getContext().getLangOpts()),
                       DRE->getEndLoc(), "[0]");
      } else {
        addReplacement(Lexer::getLocForEndOfToken(
                           DRE->getEndLoc(), 0, SM,
                           DpctGlobalInfo::getContext().getLangOpts()),
                       DRE->getEndLoc(), ".get_ptr()");
      }
    }
  }
  Base::analyzeExpr(DRE);
}

void KernelArgumentAnalysis::analyzeExpr(const MemberExpr *ME) {
  if (ME->getBase()->getType()->isDependentType()) {
    IsRedeclareRequired = true;
  }

  if (ME->getBase()->isImplicitCXXThis()) {
    IsRedeclareRequired = true;
  }

  if (ME->isArrow()) {
    IsRedeclareRequired = true;
  }

  if (auto RD = ME->getBase()->getType()->getAsCXXRecordDecl()) {
    if (!RD->isStandardLayout()) {
      IsRedeclareRequired = true;
    }
  }
  Base::analyzeExpr(ME);
}

void KernelArgumentAnalysis::analyzeExpr(const UnaryOperator *UO) {
  if (UO->getOpcode() == UO_Deref) {
    IsRedeclareRequired = true;
    return;
  }
  if (UO->getOpcode() == UO_AddrOf) {
    IsAddrOf = true;
  }
  dispatch(UO->getSubExpr());
  /// If subexpr is variable defined on device, remove operator '&'.
  if (IsAddrOf && IsDefinedOnDevice) {
    addReplacement(UO->getOperatorLoc(), "");
  }
  /// Clear flag 'IsDefinedOnDevice' and 'IsAddrOf'
  IsDefinedOnDevice = false;
  IsAddrOf = false;
}

void KernelArgumentAnalysis::analyze(const Expr *Expression) {
  IsPointer = Expression->getType()->isPointerType();
  TryGetBuffer = IsPointer && DpctGlobalInfo::getUsmLevel() == UsmLevel::none &&
                 !isNullPtr(Expression);
  IsRedeclareRequired = false;
  ArgumentAnalysis::analyze(Expression);
}

bool KernelArgumentAnalysis::isNullPtr(const Expr *E) {
  E = E->IgnoreCasts();
  if (isa<GNUNullExpr>(E))
    return true;
  if (isa<CXXNullPtrLiteralExpr>(E))
    return true;
  if (auto IL = dyn_cast<IntegerLiteral>(E)) {
    if (!IL->getValue().getZExtValue())
      return true;
  }
  return false;
}

void KernelConfigAnalysis::dispatch(const Stmt *Expression) {
  switch (Expression->getStmtClass()) {
    ANALYZE_EXPR(CXXConstructExpr)
    ANALYZE_EXPR(CXXTemporaryObjectExpr)
  default:
    return ArgumentAnalysis::dispatch(Expression);
  }
}

void KernelConfigAnalysis::analyzeExpr(const CXXConstructExpr *Ctor) {
  if (Ctor->getConstructor()->getDeclName().getAsString() == "dim3") {
    if (ArgIndex == 1) {
      if (calculateWorkgroupSize(Ctor) <= 256)
        NeedEmitWGSizeWarning = false;
    }

    std::string CtorString;
    llvm::raw_string_ostream OS(CtorString);
    DpctGlobalInfo::printCtadClass(OS, MapNames::getClNamespace() + "::range",
                                   3)
        << "(";
    auto Args = getCtorArgs(Ctor);
    if (DoReverse && Ctor->getNumArgs() == 3) {
      Reversed = true;
      int Index = Args.size();
      while (Index)
        OS << Args[--Index] << ", ";
    } else {
      for (auto &A : Args)
        OS << A << ", ";
    }
    OS.flush();
    // Special handling for implicit ctor.
    // #define GET_BLOCKS(a) a
    // foo_kernel<<<GET_BLOCKS(1), 2, 0>>>();
    // Result if using SM.getExpansionRange:
    //   sycl::range<3>(1, 1, GET_BLOCKS(1)) in kernel
    // Result if using addReplacement(E):
    //   #define GET_BLOCKS(a) sycl::range<3>(1, 1, a)
    //   GET_BLOCKS(1) in kernel
    if (Ctor->getParenOrBraceRange().isInvalid() && isOuterMostMacro(Ctor)) {
      return addReplacement(
          SM.getExpansionRange(Ctor->getBeginLoc()).getBegin(),
          SM.getExpansionRange(Ctor->getEndLoc()).getEnd(),
          CtorString.replace(CtorString.length() - 2, 2, ")"));
    }
    return addReplacement(Ctor,
                          CtorString.replace(CtorString.length() - 2, 2, ")"));
  }
  return ArgumentAnalysis::analyzeExpr(Ctor);
}

std::vector<std::string>
KernelConfigAnalysis::getCtorArgs(const CXXConstructExpr *Ctor) {
  std::vector<std::string> Args;
  ArgumentAnalysis A(IsInMacroDefine);
  for (auto Arg : Ctor->arguments())
    Args.emplace_back(getCtorArg(A, Arg));
  return Args;
}

void KernelConfigAnalysis::analyze(const Expr *E, unsigned int Idx,
                                   bool ReverseIfNeed) {
  ArgIndex = Idx;
  MustDim3 = ArgIndex < 2;

  if (IsInMacroDefine && SM.isMacroArgExpansion(E->getBeginLoc())) {
    Reversed = false;
    DirectRef = true;
    if (ArgIndex == 3 && isPredefinedStreamHandle(E)) {
      addReplacement("0");
      return;
    }
    initArgumentExpr(E);
    return;
  }

  DoReverse = ReverseIfNeed;
  if (ArgIndex == 3 && isPredefinedStreamHandle(E)) {
    addReplacement("0");
    return;
  }

  ArgumentAnalysis::analyze(E);

  if (getTargetExpr()->IgnoreImplicit()->getStmtClass() ==
          Stmt::DeclRefExprClass ||
      getTargetExpr()->IgnoreImpCasts()->getStmtClass() ==
          Stmt::MemberExprClass ||
      getTargetExpr()->IgnoreImpCasts()->getStmtClass() ==
          Stmt::IntegerLiteralClass) {
    if (MustDim3 && getTargetExpr()->getType()->isIntegralType(
                        DpctGlobalInfo::getContext())) {
      addReplacement(buildString(DpctGlobalInfo::getCtadClass(
                                     MapNames::getClNamespace() + "::range", 3),
                                 "(0, 0, ", getReplacedString(), ")"));
      Reversed = true;
      return;
    }

    DirectRef = true;
  }
}

std::string ArgumentAnalysis::getRewriteString() {
  // Find rewrite range
  auto RewriteRange = getLocInCallSpelling(getTargetExpr());
  auto RewriteRangeBegin = RewriteRange.first;
  auto RewriteRangeEnd = RewriteRange.second;
  size_t RewriteLength = SM.getCharacterData(RewriteRangeEnd) -
                         SM.getCharacterData(RewriteRangeBegin);
  // Get original string
  auto DL = SM.getDecomposedLoc(RewriteRangeBegin);
  std::string OriginalStr =
      std::string(SM.getBufferData(DL.first).substr(DL.second, RewriteLength));

  StringReplacements SRs;
  SRs.init(std::move(OriginalStr));

  for (std::shared_ptr<ExtReplacement> SubRepl : SubExprRepl) {
    if (isInRange(RewriteRangeBegin, RewriteRangeEnd, SubRepl->getFilePath(),
                  SubRepl->getOffset()) &&
        isInRange(RewriteRangeBegin, RewriteRangeEnd, SubRepl->getFilePath(),
                  SubRepl->getOffset() + SubRepl->getLength())) {
      SRs.addStringReplacement(
          SubRepl->getOffset() - SM.getDecomposedLoc(RewriteRangeBegin).second,
          SubRepl->getLength(), SubRepl->getReplacementText().str());
    }
  }

  return SRs.getReplacedString();
}

std::pair<SourceLocation, SourceLocation> ArgumentAnalysis::getLocInCallSpelling(const Expr* E) {
  auto BeginCandidate = SM.getExpansionRange(E->getSourceRange()).getBegin();
  auto EndCandidate = SM.getExpansionRange(E->getSourceRange()).getEnd();
  auto LastTokenLength =
    Lexer::MeasureTokenLength(EndCandidate, SM, Context.getLangOpts());
  EndCandidate = EndCandidate.getLocWithOffset(LastTokenLength);
  if (E->getBeginLoc().isMacroID() && !isInRange(CallSpellingBegin, CallSpellingEnd, BeginCandidate)) {
    // Try getImmediateSpellingLoc
    // e.g. M1(call(M2))
    BeginCandidate = SM.getImmediateSpellingLoc(E->getBeginLoc());
    if (BeginCandidate.isMacroID()) {
      BeginCandidate =
        SM.getExpansionLoc(BeginCandidate);
    }
    if (!isInRange(CallSpellingBegin, CallSpellingEnd, BeginCandidate)) {
      // Try getImmediateExpansionRange
      // e.g. #define M1(x) call(x)
      BeginCandidate =
        SM.getSpellingLoc(SM.getImmediateExpansionRange(E->getBeginLoc()).getBegin());
      if (!isInRange(CallSpellingBegin, CallSpellingEnd, BeginCandidate)) {
        // Multi-Level funclike special process
        // e.g.
        // #define M1(x) call1(x)
        // #define M2(y) call2(y)
        // M1(M2(3))
        BeginCandidate = getStmtSpellingSourceRange(E).getBegin();
        if (!isInRange(CallSpellingBegin, CallSpellingEnd, BeginCandidate)) {
          if (!isExprStraddle(E)) {
            // Default use SpellingLoc
            // e.g. M1(call(targetExpr))
            BeginCandidate = SM.getSpellingLoc(E->getBeginLoc());
          } else {
            BeginCandidate = SM.getExpansionRange(E->getSourceRange()).getBegin();
          }
        }
      }
    }
  }

  // Similar to get begin loc, but need to add last token length
  if (E->getEndLoc().isMacroID() && !isInRange(CallSpellingBegin, CallSpellingEnd, EndCandidate)) {
    // Try ImmediateSpelling
    EndCandidate = SM.getImmediateSpellingLoc(E->getEndLoc());
    if (EndCandidate.isMacroID()) {
      EndCandidate = SM.getImmediateExpansionRange(EndCandidate).getEnd();
    }
    auto LastTokenLength =
      Lexer::MeasureTokenLength(EndCandidate, SM, Context.getLangOpts());
    EndCandidate = EndCandidate.getLocWithOffset(LastTokenLength);
    if (!isInRange(CallSpellingBegin, CallSpellingEnd, EndCandidate)) {
      // Try ImmediateExpansion
      EndCandidate =
        SM.getSpellingLoc(SM.getImmediateExpansionRange(E->getEndLoc()).getEnd());
      auto LastTokenLength =
        Lexer::MeasureTokenLength(EndCandidate, SM, Context.getLangOpts());
      EndCandidate = EndCandidate.getLocWithOffset(LastTokenLength);
      if (!isInRange(CallSpellingBegin, CallSpellingEnd, EndCandidate)) {
        EndCandidate = getStmtSpellingSourceRange(E).getEnd();
        auto LastTokenLength =
          Lexer::MeasureTokenLength(EndCandidate, SM, Context.getLangOpts());
        EndCandidate = EndCandidate.getLocWithOffset(LastTokenLength);
        if (!isInRange(CallSpellingBegin, CallSpellingEnd, EndCandidate)) {
          if (!isExprStraddle(E)) {
            // Default use SpellingLoc
            EndCandidate = SM.getSpellingLoc(E->getEndLoc());
            auto LastTokenLength =
              Lexer::MeasureTokenLength(EndCandidate, SM, Context.getLangOpts());
            EndCandidate = EndCandidate.getLocWithOffset(LastTokenLength);
          } else {
            EndCandidate = SM.getExpansionRange(E->getSourceRange()).getEnd();
          }
        }
      }
    }
  }
  return std::pair<SourceLocation, SourceLocation>(BeginCandidate, EndCandidate);
}

void SideEffectsAnalysis::dispatch(const Stmt *Expression) {
  switch (Expression->getStmtClass()) {
    ANALYZE_EXPR(BinaryOperator)
  // The following types of expressions don't have side effects
  case Stmt::IntegerLiteralClass:
  case Stmt::FloatingLiteralClass:
  case Stmt::StringLiteralClass:
  case Stmt::DeclRefExprClass:
  case Stmt::ArraySubscriptExprClass:
  case Stmt::CStyleCastExprClass:
  case Stmt::CXXStaticCastExprClass:
  case Stmt::ImplicitCastExprClass:
  case Stmt::ParenExprClass:
  case Stmt::ConditionalOperatorClass:
  case Stmt::MemberExprClass:
      break;
  default:
    HasSideEffects = true;
  }
  return ExprAnalysis::dispatch(Expression);
}

} // namespace dpct
} // namespace clang
