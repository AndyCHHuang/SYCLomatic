//===--- ASTTraversal.cpp --------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "ASTTraversal.h"
#include "AnalysisInfo.h"
#include "Debug.h"
#include "SaveNewFiles.h"
#include "Utility.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Path.h"

#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::syclct;
using namespace clang::tooling;

extern std::string CudaPath;
extern std::string SyclctInstallPath; // Installation directory for this tool

auto parentStmt = anyOf(hasParent(compoundStmt()), hasParent(forStmt()),
                        hasParent(whileStmt()), hasParent(ifStmt()));

std::unordered_map<std::string, std::unordered_set</* Comment ID */ int>>
    TranslationRule::ReportedComment;

static std::set<SourceLocation> AttrExpansionFilter;

// Remember the location of the last inclusion directive for each file
static std::map<FileID, SourceLocation> IncludeLocations;

unsigned TranslationRule::PairID = 0;

void IncludesCallbacks::ReplaceCuMacro(const Token &MacroNameTok) {
  std::string InRoot = ATM.InRoot;
  std::string InFile = SM.getFilename(MacroNameTok.getLocation());
  bool IsInRoot = !llvm::sys::fs::is_directory(InFile) &&
                  (isChildPath(InRoot, InFile) || isSamePath(InRoot, InFile));

  if (!IsInRoot) {
    return;
  }
  if (!MacroNameTok.getIdentifierInfo()) {
    return;
  }
  std::string MacroName = MacroNameTok.getIdentifierInfo()->getName().str();
  if (MapNames::MacrosMap.find(MacroName) != MapNames::MacrosMap.end()) {
    std::string ReplacedMacroName = MapNames::MacrosMap.at(MacroName);
    TransformSet.emplace_back(new ReplaceToken(MacroNameTok.getLocation(),
                                               std::move(ReplacedMacroName)));
  }
}

void IncludesCallbacks::MacroDefined(const Token &MacroNameTok,
                                     const MacroDirective *MD) {
  std::string InRoot = ATM.InRoot;
  std::string InFile = SM.getFilename(MacroNameTok.getLocation());
  bool IsInRoot = !llvm::sys::fs::is_directory(InFile) &&
                  (isChildPath(InRoot, InFile) || isSamePath(InRoot, InFile));

  if (!IsInRoot) {
    return;
  }

  // Remove __global__, __host__ and __device__ if they act as replacement
  // tokens other macros.
  auto MI = MD->getMacroInfo();
  for (auto Iter = MI->tokens_begin(); Iter != MI->tokens_end(); ++Iter) {
    auto II = Iter->getIdentifierInfo();
    if (!II)
      continue;
    if (II->hasMacroDefinition() && (II->getName().str() == "__host__" ||
                                     II->getName().str() == "__device__" ||
                                     II->getName().str() == "__global__")) {
      TransformSet.emplace_back(new ReplaceToken(Iter->getLocation(), ""));
    }
  }
}
void IncludesCallbacks::MacroExpands(const Token &MacroNameTok,
                                     const MacroDefinition &MD,
                                     SourceRange Range, const MacroArgs *Args) {
  std::string InRoot = ATM.InRoot;
  std::string InFile = SM.getFilename(MacroNameTok.getLocation());
  bool IsInRoot = !llvm::sys::fs::is_directory(InFile) &&
                  (isChildPath(InRoot, InFile) || isSamePath(InRoot, InFile));

  if (!IsInRoot) {
    return;
  }

  // Record the expansion locations of the macros containing CUDA attributes.
  // FunctionAttrsRule should/will NOT work on these locations.
  auto MI = MD.getMacroInfo();
  for (auto Iter = MI->tokens_begin(); Iter != MI->tokens_end(); ++Iter) {
    auto II = Iter->getIdentifierInfo();
    if (!II)
      continue;
    if (II->hasMacroDefinition() && (II->getName().str() == "__host__" ||
                                     II->getName().str() == "__device__" ||
                                     II->getName().str() == "__global__")) {
      AttrExpansionFilter.insert(Range.getBegin());
    }
  }
}

void IncludesCallbacks::Ifdef(SourceLocation Loc, const Token &MacroNameTok,
                              const MacroDefinition &MD) {
  ReplaceCuMacro(MacroNameTok);
}
void IncludesCallbacks::Ifndef(SourceLocation Loc, const Token &MacroNameTok,
                               const MacroDefinition &MD) {
  ReplaceCuMacro(MacroNameTok);
}

void IncludesCallbacks::Defined(const Token &MacroNameTok,
                                const MacroDefinition &MD, SourceRange Range) {
  ReplaceCuMacro(MacroNameTok);
}

void IncludesCallbacks::ReplaceCuMacro(SourceRange ConditionRange) {
  // __CUDA_ARCH__ is not defined in clang, and need check if it is use
  // in #if and #elif
  auto Begin = SM.getExpansionLoc(ConditionRange.getBegin());
  auto End = SM.getExpansionLoc(ConditionRange.getEnd());
  const char *BP = SM.getCharacterData(Begin);
  const char *EP = SM.getCharacterData(End);
  unsigned int Size = EP - BP + 1;
  std::string E(BP, Size);
  size_t Pos = 0;
  const std::string MacroName = "__CUDA_ARCH__";
  std::string ReplacedMacroName;
  if (MapNames::MacrosMap.find(MacroName) != MapNames::MacrosMap.end()) {
    ReplacedMacroName = MapNames::MacrosMap.at(MacroName);
  } else {
    return;
  }

  std::size_t Found = E.find(MacroName, Pos);
  while (Found != std::string::npos) {
    // found one, insert replace for it
    if (MapNames::MacrosMap.find(MacroName) != MapNames::MacrosMap.end()) {
      SourceLocation IB = Begin.getLocWithOffset(Found);
      SourceLocation IE = IB.getLocWithOffset(MacroName.length());
      CharSourceRange InsertRange(SourceRange(IB, IE), false);
      TransformSet.emplace_back(
          new ReplaceInclude(InsertRange, std::move(ReplacedMacroName)));
    }
    // check next
    Pos = Found + MacroName.length();
    if ((Pos + MacroName.length()) > Size) {
      break;
    }
    Found = E.find(MacroName, Pos);
  }
}
void IncludesCallbacks::If(SourceLocation Loc, SourceRange ConditionRange,
                           ConditionValueKind ConditionValue) {
  std::string InRoot = ATM.InRoot;
  std::string InFile = SM.getFilename(Loc);
  bool IsInRoot = !llvm::sys::fs::is_directory(InFile) &&
                  (isChildPath(InRoot, InFile) || isSamePath(InRoot, InFile));

  if (!IsInRoot) {
    return;
  }
  ReplaceCuMacro(ConditionRange);
}
void IncludesCallbacks::Elif(SourceLocation Loc, SourceRange ConditionRange,
                             ConditionValueKind ConditionValue,
                             SourceLocation IfLoc) {
  std::string InRoot = ATM.InRoot;
  std::string InFile = SM.getFilename(Loc);
  bool IsInRoot = !llvm::sys::fs::is_directory(InFile) &&
                  (isChildPath(InRoot, InFile) || isSamePath(InRoot, InFile));

  if (!IsInRoot) {
    return;
  }

  ReplaceCuMacro(ConditionRange);
}

void IncludesCallbacks::InclusionDirective(
    SourceLocation HashLoc, const Token &IncludeTok, StringRef FileName,
    bool IsAngled, CharSourceRange FilenameRange, const FileEntry *File,
    StringRef SearchPath, StringRef RelativePath, const Module *Imported,
    SrcMgr::CharacteristicKind FileType) {
  // Record the locations of inclusion directives
  // The last inclusion diretive of a file will be remembered
  auto DecomposedLoc = SM.getDecomposedExpansionLoc(FilenameRange.getEnd());
  IncludeLocations[DecomposedLoc.first] = FilenameRange.getEnd();

  std::string IncludePath = SearchPath;
  makeCanonical(IncludePath);
  std::string IncludingFile = SM.getFilename(HashLoc);

  IncludingFile = getAbsolutePath(IncludingFile);
  makeCanonical(IncludingFile);

  // eg. '/home/path/util.h' -> '/home/path'
  StringRef Directory = llvm::sys::path::parent_path(IncludingFile);
  std::string InRoot = ATM.InRoot;

  bool IsIncludingFileInInRoot = !llvm::sys::fs::is_directory(IncludingFile) &&
                                 (isChildPath(InRoot, Directory.str()) ||
                                  isSamePath(InRoot, Directory.str()));

  // If the header file included can not be found, just return.
  if (!File) {
    return;
  }

  std::string FilePath = File->getName();
  makeCanonical(FilePath);
  std::string DirPath = llvm::sys::path::parent_path(FilePath);
  bool IsFileInInRoot =
      !isChildPath(SyclctInstallPath, DirPath) &&
      (isChildPath(InRoot, DirPath) || isSamePath(InRoot, DirPath));

  if (IsFileInInRoot && !StringRef(FilePath).endswith(".cu")) {
    auto Find = IncludeFileMap.find(FilePath);
    if (Find == IncludeFileMap.end()) {
      IncludeFileMap[FilePath] = false;
    }
  }

  if (!SM.isWrittenInMainFile(HashLoc) && !IsIncludingFileInInRoot) {
    return;
  }

  // Insert SYCL headers for file inputted or file included.
  // E.g. A.cu included B.cu, both A.cu and B.cu are inserted "#include
  // <CL/sycl.hpp>\n#include <syclct/syclct.hpp>"
  if (!SyclHeaderInserted || SeenFiles.find(IncludingFile) == end(SeenFiles)) {
    SeenFiles.insert(IncludingFile);
    std::string Replacement = std::string("#include <CL/sycl.hpp>") + getNL() +
                              "#include <syclct/syclct.hpp>" + getNL();
    CharSourceRange InsertRange(SourceRange(HashLoc, HashLoc), false);
    TransformSet.emplace_back(
        new ReplaceInclude(InsertRange, std::move(Replacement)));
    SyclHeaderInserted = true;
  }

  // replace "#include <math.h>" with <cmath>
  if (IsAngled && FileName.compare(StringRef("math.h")) == 0) {
    TransformSet.emplace_back(new ReplaceInclude(
        CharSourceRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                        /*IsTokenRange=*/false),
        "#include <cmath>"));
  }

  // replace "#include <cublas_v2.h>" with <DPCPP_blas_TEMP.h>
  if (IsAngled && FileName.compare(StringRef("cublas_v2.h")) == 0) {
    TransformSet.emplace_back(new ReplaceInclude(
        CharSourceRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                        /*IsTokenRange=*/false),
        "#include <DPCPP_blas_TEMP.h>"));
  }

  if (!isChildPath(CudaPath, IncludePath) &&
      // CudaPath detection have not consider soft link, here do special
      // for /usr/local/cuda
      IncludePath.compare(0, 15, "/usr/local/cuda", 15)) {

    // Replace "#include "*.cuh"" with "include "*.sycl.hpp""
    if (!IsAngled && FileName.endswith(".cuh")) {
      CharSourceRange InsertRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                                  /* IsTokenRange */ false);
      std::string NewFileName = "#include \"" +
                                FileName.drop_back(strlen(".cuh")).str() +
                                ".sycl.hpp\"";
      TransformSet.emplace_back(
          new ReplaceInclude(InsertRange, std::move(NewFileName)));
      return;
    }

    // Replace "#include "*.cu"" with "include "*.sycl.cpp""
    if (!IsAngled && FileName.endswith(".cu")) {
      CharSourceRange InsertRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                                  /* IsTokenRange */ false);
      std::string NewFileName = "#include \"" +
                                FileName.drop_back(strlen(".cu")).str() +
                                ".sycl.cpp\"";
      TransformSet.emplace_back(
          new ReplaceInclude(InsertRange, std::move(NewFileName)));
      return;
    }
  }

  // Extra process thrust headers, map to PSTL mapping headers in runtime.
  // For multi thrust header files, only insert once for PSTL mapping header.
  if (IsAngled && (FileName.find("thrust/") != std::string::npos)) {
    if (!ThrustHeaderInserted) {
      std::string Replacement;
      if (!SyclHeaderInserted) {
        Replacement = std::string("<CL/sycl.hpp>") + getNL() +
                      "#include <syclct/syclct.hpp>" + getNL() +
                      "#include <syclct/syclct_thrust.hpp>";
        SyclHeaderInserted = true;
      } else {
        Replacement = std::string("<syclct/syclct_thrust.hpp>");
      }
      ThrustHeaderInserted = true;
      TransformSet.emplace_back(
          new ReplaceInclude(FilenameRange, std::move(Replacement)));
    } else {
      // Replace the complete include directive with an empty string.
      TransformSet.emplace_back(new ReplaceInclude(
          CharSourceRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                          /*IsTokenRange=*/false),
          ""));
    }
    return;
  }

  // if it's not an include from the Cuda SDK, leave it,
  // unless it's <cuda_runtime.h>, in which case it will be replaced.
  // In other words, <cuda_runtime.h> will be replaced regardless of where it's
  // coming from
  if (!isChildPath(CudaPath, IncludePath) &&
      IncludePath.compare(0, 15, "/usr/local/cuda", 15)) {
    if (!(IsAngled && FileName.compare(StringRef("cuda_runtime.h")) == 0)) {
      return;
    }
  }

  // Multiple CUDA headers in an including file will be replaced with one
  // include of the SYCL header.
  if ((SeenFiles.find(IncludingFile) == end(SeenFiles)) &&
      (!SyclHeaderInserted)) {
    SeenFiles.insert(IncludingFile);
    std::string Replacement =
        std::string("<CL/sycl.hpp>") + getNL() + "#include <syclct/syclct.hpp>";
    TransformSet.emplace_back(
        new ReplaceInclude(FilenameRange, std::move(Replacement)));
    SyclHeaderInserted = true;
  } else {
    // Replace the complete include directive with an empty string.
    TransformSet.emplace_back(new ReplaceInclude(
        CharSourceRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                        /*IsTokenRange=*/false),
        ""));
  }
}

void IncludesCallbacks::FileChanged(SourceLocation Loc, FileChangeReason Reason,
                                    SrcMgr::CharacteristicKind FileType,
                                    FileID PrevFID) {
  // Record the location when a file is entered
  if (Reason == clang::PPCallbacks::EnterFile) {
    auto DecomposedLoc = SM.getDecomposedExpansionLoc(Loc);
    IncludeLocations[DecomposedLoc.first] = Loc;
  }
}

void TranslationRule::print(llvm::raw_ostream &OS) {
  const auto &EmittedTransformations = getEmittedTransformations();
  if (EmittedTransformations.empty()) {
    return;
  }

  OS << "[" << getName() << "]" << getNL();
  constexpr char Indent[] = "  ";
  for (const TextModification *TM : EmittedTransformations) {
    OS << Indent;
    TM->print(OS, getCompilerInstance().getASTContext(),
              /* Print parent */ false);
  }
}

void TranslationRule::printStatistics(llvm::raw_ostream &OS) {
  const auto &EmittedTransformations = getEmittedTransformations();
  if (EmittedTransformations.empty()) {
    return;
  }

  OS << "<Statistics of " << getName() << ">" << getNL();
  std::unordered_map<std::string, size_t> TMNameCountMap;
  for (const TextModification *TM : EmittedTransformations) {
    const std::string Name = TM->getName();
    if (TMNameCountMap.count(Name) == 0) {
      TMNameCountMap.emplace(std::make_pair(Name, 1));
    } else {
      ++TMNameCountMap[Name];
    }
  }

  constexpr char Indent[] = "  ";
  for (const auto &Pair : TMNameCountMap) {
    const std::string &Name = Pair.first;
    const size_t &Numbers = Pair.second;
    OS << Indent << "Emitted # of replacement <" << Name << ">: " << Numbers
       << getNL();
  }
}

void TranslationRule::emplaceTransformation(const char *RuleID,
                                            TextModification *TM) {
  ASTTraversalMetaInfo::getEmittedTransformations()[RuleID].emplace_back(TM);
  TransformSet->emplace_back(TM);
}

void IterationSpaceBuiltinRule::registerMatcher(MatchFinder &MF) {
  // TODO: check that threadIdx is not a local variable.
  MF.addMatcher(
      memberExpr(hasObjectExpression(opaqueValueExpr(hasSourceExpression(
                     declRefExpr(to(varDecl(hasAnyName("threadIdx", "blockDim",
                                                       "blockIdx", "gridDim"))
                                        .bind("varDecl")))))),
                 hasAncestor(functionDecl().bind("func")))
          .bind("memberExpr"),
      this);
}

void IterationSpaceBuiltinRule::run(const MatchFinder::MatchResult &Result) {
  const MemberExpr *ME = getNodeAsType<MemberExpr>(Result, "memberExpr");
  if (!ME)
    return;
  if (auto FD = getAssistNodeAsType<FunctionDecl>(Result, "func"))
    DeviceFunctionDecl::LinkRedecls(FD)->setItem();
  const VarDecl *VD = getAssistNodeAsType<VarDecl>(Result, "varDecl", false);
  assert(ME && VD && "Unknown result");

  ValueDecl *Field = ME->getMemberDecl();
  StringRef FieldName = Field->getName();
  unsigned Dimension;

  if (FieldName == "__fetch_builtin_x")
    Dimension = 0;
  else if (FieldName == "__fetch_builtin_y")
    Dimension = 1;
  else if (FieldName == "__fetch_builtin_z")
    Dimension = 2;
  else
    syclct_unreachable("Unknown field name");

  std::string Replacement = getItemName();
  StringRef BuiltinName = VD->getName();

  if (BuiltinName == "threadIdx")
    Replacement += ".get_local_id(";
  else if (BuiltinName == "blockDim")
    Replacement += ".get_local_range().get(";
  else if (BuiltinName == "blockIdx")
    Replacement += ".get_group(";
  else if (BuiltinName == "gridDim")
    Replacement += ".get_group_range(";
  else
    syclct_unreachable("Unknown builtin variable");

  Replacement += std::to_string(Dimension);
  Replacement += ")";
  emplaceTransformation(new ReplaceStmt(ME, std::move(Replacement)));
}

REGISTER_RULE(IterationSpaceBuiltinRule)

void ErrorHandlingIfStmtRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      // Match if-statement that has no else and has a condition of either an
      // operator!= or a variable of type enum.
      ifStmt(unless(hasElse(anything())),
             hasCondition(
                 anyOf(binaryOperator(hasOperatorName("!=")).bind("op!="),
                       ignoringImpCasts(
                           declRefExpr(hasType(hasCanonicalType(enumType())))
                               .bind("var")))))
          .bind("errIf"),
      this);
  MF.addMatcher(
      // Match if-statement that has no else and has a condition of
      // operator==.
      ifStmt(unless(hasElse(anything())),
             hasCondition(binaryOperator(hasOperatorName("==")).bind("op==")))
          .bind("errIfSpecial"),
      this);
}

static bool isVarRef(const Expr *E) {
  if (auto D = dyn_cast<DeclRefExpr>(E))
    return isa<VarDecl>(D->getDecl());
  else
    return false;
}

static std::string getVarType(const Expr *E) {
  return E->getType().getCanonicalType().getUnqualifiedType().getAsString();
}

static bool isCudaFailureCheck(const BinaryOperator *Op, bool IsEq = false) {
  auto Lhs = Op->getLHS()->IgnoreImplicit();
  auto Rhs = Op->getRHS()->IgnoreImplicit();

  const Expr *Literal = nullptr;
  if (isVarRef(Lhs) && getVarType(Lhs) == "enum cudaError")
    Literal = Rhs;
  else if (isVarRef(Rhs) && getVarType(Rhs) == "enum cudaError")
    Literal = Lhs;
  else
    return false;

  if (auto IntLit = dyn_cast<IntegerLiteral>(Literal)) {
    if (IsEq ^ (IntLit->getValue() != 0))
      return false;
  } else if (auto D = dyn_cast<DeclRefExpr>(Literal)) {
    auto EnumDecl = dyn_cast<EnumConstantDecl>(D->getDecl());
    if (!EnumDecl)
      return false;
    // Check for cudaSuccess or CUDA_SUCCESS.
    if (IsEq ^ (EnumDecl->getInitVal() != 0))
      return false;
  } else {
    // The expression is neither an int literal nor an enum value.
    return false;
  }

  return true;
}

static bool isCudaFailureCheck(const DeclRefExpr *E) {
  return isVarRef(E) && getVarType(E) == "enum cudaError";
}

void ErrorHandlingIfStmtRule::run(const MatchFinder::MatchResult &Result) {
  static std::vector<std::string> NameList = {"errIf", "errIfSpecial"};
  const IfStmt *If = getNodeAsType<IfStmt>(Result, "errIf");
  if (!If)
    if (!(If = getNodeAsType<IfStmt>(Result, "errIfSpecial")))
      return;
  auto EmitNotRemoved = [&](SourceLocation SL, const Stmt *R) {
    report(SL, Diagnostics::STMT_NOT_REMOVED);
  };
  auto isErrorHandlingSafeToRemove = [&](const Stmt *S) {
    if (const auto *CE = dyn_cast<CallExpr>(S)) {
      if (!CE->getDirectCallee()) {
        EmitNotRemoved(S->getSourceRange().getBegin(), S);
        return false;
      }
      auto Name = CE->getDirectCallee()->getNameAsString();
      static const llvm::StringSet<> SafeCallList = {
          "printf", "puts", "exit", "cudaDeviceReset", "fprintf"};
      if (SafeCallList.find(Name) == SafeCallList.end()) {
        EmitNotRemoved(S->getSourceRange().getBegin(), S);
        return false;
      }
#if 0
    //TODO: enable argument check
    for (const auto *S : CE->arguments()) {
      if (!isErrorHandlingSafeToRemove(S->IgnoreImplicit()))
        return false;
    }
#endif
      return true;
    }
#if 0
  //TODO: enable argument check
  else if (isa <DeclRefExpr>(S))
    return true;
  else if (isa<IntegerLiteral>(S))
    return true;
  else if (isa<StringLiteral>(S))
    return true;
#endif
    EmitNotRemoved(S->getSourceRange().getBegin(), S);
    return false;
  };

  auto isErrorHandling = [&](const Stmt *Block) {
    if (!isa<CompoundStmt>(Block))
      return isErrorHandlingSafeToRemove(Block);
    const CompoundStmt *CS = cast<CompoundStmt>(Block);
    for (const auto *S : CS->children()) {
      if (auto *E = dyn_cast_or_null<Expr>(S)) {
        if (!isErrorHandlingSafeToRemove(E->IgnoreImplicit())) {
          return false;
        }
      }
    }
    return true;
  };

  if (![&] {
        bool IsIfstmtSpecialCase = false;
        SourceLocation Ip;
        if (auto Op = getNodeAsType<BinaryOperator>(Result, "op!=")) {
          if (!isCudaFailureCheck(Op))
            return false;
        } else if (auto Op = getNodeAsType<BinaryOperator>(Result, "op==")) {
          if (!isCudaFailureCheck(Op, true))
            return false;
          IsIfstmtSpecialCase = true;
          Ip = Op->getBeginLoc();

        } else {
          auto CondVar = getNodeAsType<DeclRefExpr>(Result, "var");
          if (!isCudaFailureCheck(CondVar))
            return false;
        }
        // We know that it's error checking condition, check the body
        if (!isErrorHandling(If->getThen())) {
          if (IsIfstmtSpecialCase) {
            report(Ip, Diagnostics::IFSTMT_SPECIAL_CASE);
          } else {
            report(If->getSourceRange().getBegin(),
                   Diagnostics::IFSTMT_NOT_REMOVED);
          }
          return false;
        }
        return true;
      }()) {

    return;
  }

  emplaceTransformation(new ReplaceStmt(If, ""));
}

REGISTER_RULE(ErrorHandlingIfStmtRule)

void AlignAttrsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(cxxRecordDecl(hasAttr(attr::Aligned)).bind("classDecl"), this);
}

void AlignAttrsRule::run(const MatchFinder::MatchResult &Result) {
  auto C = getNodeAsType<CXXRecordDecl>(Result, "classDecl");
  if (!C)
    return;
  auto &AV = C->getAttrs();

  for (auto A : AV) {
    if (A->getKind() == attr::Aligned) {
      auto SM = Result.SourceManager;
      auto ExpB = SM->getExpansionLoc(A->getLocation());
      if (!strncmp(SM->getCharacterData(ExpB), "__align__(", 10))
        emplaceTransformation(new ReplaceToken(ExpB, "__sycl_align__"));
    }
  }
}

REGISTER_RULE(AlignAttrsRule)

void FunctionAttrsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      functionDecl(anyOf(hasAttr(attr::CUDAGlobal), hasAttr(attr::CUDADevice),
                         hasAttr(attr::CUDAHost)))
          .bind("functionDecl"),
      this);
}

void FunctionAttrsRule::run(const MatchFinder::MatchResult &Result) {
  const FunctionDecl *FD = getNodeAsType<FunctionDecl>(Result, "functionDecl");
  if (!FD)
    return;
  const AttrVec &AV = FD->getAttrs();

  for (const Attr *A : AV) {
    attr::Kind AK = A->getKind();
    if (!A->isImplicit() && (AK == attr::CUDAGlobal || AK == attr::CUDADevice ||
                             AK == attr::CUDAHost)) {
      // If __global__, __host__ and __device__ are defined in other macros,
      // the replacements should happen at spelling locations of these macros
      // instead of expansion locations. In these cases, no work is needed here.
      auto Loc = A->getLocation();
      Loc = Result.SourceManager->getExpansionLoc(Loc);
      if (AttrExpansionFilter.find(Loc) == AttrExpansionFilter.end()) {
        emplaceTransformation(new RemoveAttr(A));
      }
    }
  }
}

REGISTER_RULE(FunctionAttrsRule)

void AtomicFunctionRule::registerMatcher(MatchFinder &MF) {
  std::vector<std::string> AtomicFuncNames(AtomicFuncNamesMap.size());
  std::transform(
      AtomicFuncNamesMap.begin(), AtomicFuncNamesMap.end(),
      AtomicFuncNames.begin(),
      [](const std::pair<std::string, std::string> &p) { return p.first; });

  auto hasAnyAtomicFuncName = [&]() {
    return internal::Matcher<NamedDecl>(
        new internal::HasNameMatcher(AtomicFuncNames));
  };

  // Support all integer type, float and double
  // Type half and half2 are not supported
  auto supportedTypes = [&]() {
    // TODO: investigate usage of __half and __half2 types and support it
    return anyOf(hasType(pointsTo(isInteger())),
                 hasType(pointsTo(asString("float"))),
                 hasType(pointsTo(asString("double"))));
  };

  auto supportedAtomicFunctions = [&]() {
    return allOf(hasAnyAtomicFuncName(), hasParameter(0, supportedTypes()));
  };

  auto unsupportedAtomicFunctions = [&]() {
    return allOf(hasAnyAtomicFuncName(),
                 unless(hasParameter(0, supportedTypes())));
  };

  MF.addMatcher(callExpr(callee(functionDecl(supportedAtomicFunctions())))
                    .bind("supportedAtomicFuncCall"),
                this);

  MF.addMatcher(callExpr(callee(functionDecl(unsupportedAtomicFunctions())))
                    .bind("unsupportedAtomicFuncCall"),
                this);
}

void AtomicFunctionRule::ReportUnsupportedAtomicFunc(const CallExpr *CE) {
  if (!CE)
    return;

  std::ostringstream OSS;
  // Atomic functions with __half and half2 are not supported.
  if (!CE->getDirectCallee())
    return;
  OSS << "half version of " << CE->getDirectCallee()->getName().str();
  report(CE->getBeginLoc(), Comments::API_NOT_MIGRATED, OSS.str());
}

void AtomicFunctionRule::TranslateAtomicFunc(
    const CallExpr *CE, const ast_matchers::MatchFinder::MatchResult &Result) {
  if (!CE)
    return;

  // TODO: 1. Investigate are there usages of atomic functions on local address
  //          space
  //       2. If item 1. shows atomic functions on local address space is
  //          significant, detect whether this atomic operation operates in
  //          global space or local space (currently, all in global space,
  //          see syclct_atomic.hpp for more details)
  if (!CE->getDirectCallee())
    return;
  const std::string AtomicFuncName = CE->getDirectCallee()->getName().str();
  assert(AtomicFuncNamesMap.find(AtomicFuncName) != AtomicFuncNamesMap.end());
  std::string ReplacedAtomicFuncName = AtomicFuncNamesMap.at(AtomicFuncName);

  // Explicitly cast all arguments except first argument
  const Type *Arg0Type = CE->getArg(0)->getType().getTypePtrOrNull();
  // Atomic operation's first argument is always pointer type
  assert(Arg0Type && Arg0Type->isPointerType());
  if (!Arg0Type || !Arg0Type->isPointerType()) {
    return;
  }
  const QualType PointeeType = Arg0Type->getPointeeType();

  std::string TypeName;
  if (auto *SubstedType = dyn_cast<SubstTemplateTypeParmType>(PointeeType)) {
    // Type is substituted in template initialization, use the template
    // parameter name
    if (!SubstedType->getReplacedParameter()->getIdentifier()) {
      return;
    }
    TypeName =
        SubstedType->getReplacedParameter()->getIdentifier()->getName().str();
  } else {
    TypeName = PointeeType.getAsString();
  }
  // add exceptions for atomic tranlastion:
  // eg. source code: atomicMin(double), don't migrate it, its user code.
  //     also: atomic_fetch_min<double> is not available in compute++.
  if ((TypeName == "double" && AtomicFuncName != "atomicAdd") ||
      (TypeName == "float" &&
       !(AtomicFuncName == "atomicAdd" || AtomicFuncName == "atomicExch"))) {

    return;
  }

  emplaceTransformation(new ReplaceCalleeName(
      CE, std::move(ReplacedAtomicFuncName), AtomicFuncName));

  const unsigned NumArgs = CE->getNumArgs();
  for (unsigned i = 1; i < NumArgs; ++i) {
    const Expr *Arg = CE->getArg(i);
    if (auto *ImpCast = dyn_cast<ImplicitCastExpr>(Arg)) {
      if (ImpCast->getCastKind() != clang::CK_LValueToRValue) {
        insertAroundStmt(Arg, "(" + TypeName + ")(", ")");
      }
    }
  }
}

void AtomicFunctionRule::run(const MatchFinder::MatchResult &Result) {
  ReportUnsupportedAtomicFunc(
      getNodeAsType<CallExpr>(Result, "unsupportedAtomicFuncCall"));

  TranslateAtomicFunc(
      getNodeAsType<CallExpr>(Result, "supportedAtomicFuncCall"), Result);
}

REGISTER_RULE(AtomicFunctionRule)

auto TypedefNames =
    hasAnyName("dim3", "cudaError_t", "cudaEvent_t", "cudaStream_t", "__half",
               "__half2", "half", "half2", "cublasStatus_t", "cublasHandle_t",
               "cuComplex", "cuDoubleComplex");
auto EnumTypeNames = hasAnyName("cudaError");
// CUstream_st and CUevent_st are the actual types of cudaStream_t and
// cudaEvent_st respectively
auto RecordTypeNames =
    hasAnyName("cudaDeviceProp", "CUstream_st", "CUevent_st");

// Rule for types replacements in var declarations and field declarations
void TypeInDeclRule::registerMatcher(MatchFinder &MF) {
  auto HasCudaType = anyOf(hasType(typedefDecl(TypedefNames)),
                           hasType(enumDecl(EnumTypeNames)),
                           hasType(cxxRecordDecl(RecordTypeNames)));

  auto HasCudaTypePtr =
      anyOf(hasType(pointsTo(typedefDecl(TypedefNames))),
            hasType(pointsTo(enumDecl(EnumTypeNames))),
            hasType(pointsTo(cxxRecordDecl(RecordTypeNames))));

  auto HasCudaTypePtrPtr =
      anyOf(hasType(pointsTo(pointsTo(typedefDecl(TypedefNames)))),
            hasType(pointsTo(pointsTo(enumDecl(EnumTypeNames)))),
            hasType(pointsTo(pointsTo(cxxRecordDecl(RecordTypeNames)))));

  auto HasCudaTypeRef =
      anyOf(hasType(references(typedefDecl(TypedefNames))),
            hasType(references(enumDecl(EnumTypeNames))),
            hasType(references(cxxRecordDecl(RecordTypeNames))));

  auto Typedefs = typedefType(hasDeclaration(typedefDecl(TypedefNames)));

  auto EnumTypes = enumType(hasDeclaration(enumDecl(EnumTypeNames)));

  auto RecordTypes = recordType(hasDeclaration(cxxRecordDecl(RecordTypeNames)));

  auto HasCudaArrayType =
      anyOf(hasType(arrayType(hasElementType(Typedefs))),
            hasType(arrayType(hasElementType(EnumTypes))),
            hasType(arrayType(hasElementType(RecordTypes))));

  auto HasCudaPtrArrayType =
      anyOf(hasType(arrayType(hasElementType(pointsTo(Typedefs)))),
            hasType(arrayType(hasElementType(pointsTo(EnumTypes)))),
            hasType(arrayType(hasElementType(pointsTo(RecordTypes)))));

  auto HasCudaPtrPtrArrayType = anyOf(
      hasType(arrayType(hasElementType(pointsTo(pointsTo(Typedefs))))),
      hasType(arrayType(hasElementType(pointsTo(pointsTo(EnumTypes))))),
      hasType(arrayType(hasElementType(pointsTo(pointsTo(RecordTypes))))));

  MF.addMatcher(varDecl(anyOf(HasCudaType, HasCudaTypePtr, HasCudaTypePtrPtr,
                              HasCudaTypeRef, HasCudaArrayType,
                              HasCudaPtrArrayType, HasCudaPtrPtrArrayType),
                        unless(hasType(substTemplateTypeParmType())))
                    .bind("TypeInVarDecl"),
                this);
  MF.addMatcher(fieldDecl(anyOf(HasCudaType, HasCudaTypePtr, HasCudaTypePtrPtr,
                                HasCudaTypeRef, HasCudaArrayType,
                                HasCudaPtrArrayType, HasCudaPtrPtrArrayType),
                          unless(hasType(substTemplateTypeParmType())))
                    .bind("TypeInFieldDecl"),
                this);
}

std::string getReplacementForType(std::string TypeStr) {
  std::istringstream ISS(TypeStr);
  std::vector<std::string> Strs(std::istream_iterator<std::string>{ISS},
                                std::istream_iterator<std::string>());
  auto it = std::remove_if(Strs.begin(), Strs.end(), [](llvm::StringRef Str) {
    return (Str.contains("&") || Str.contains("*"));
  });
  if (it != Strs.end())
    Strs.erase(it);

  const std::string &TypeName = Strs.back();
  SrcAPIStaticsMap[TypeName]++;

  auto Search = MapNames::TypeNamesMap.find(TypeName);
  if (Search == MapNames::TypeNamesMap.end())
    return "";

  std::string Replacement = TypeStr;
  assert(Replacement.find(TypeName) != std::string::npos);
  Replacement = Replacement.substr(Replacement.find(TypeName));
  Replacement.replace(0, TypeName.length(), Search->second);

  return Replacement;
}

void TypeInDeclRule::run(const MatchFinder::MatchResult &Result) {
  // DD points to a VarDecl or a FieldDecl
  const DeclaratorDecl *DD = getNodeAsType<VarDecl>(Result, "TypeInVarDecl");
  QualType QT;
  if (DD)
    QT = DD->getType();
  else if ((DD = getNodeAsType<FieldDecl>(Result, "TypeInFieldDecl")))
    QT = DD->getType();
  else
    return;

  auto Loc =
      DD->getTypeSourceInfo()->getTypeLoc().getBeginLoc().getRawEncoding();
  if (DupFilter.find(Loc) != DupFilter.end())
    return;

  std::string TypeStr;
  if (QT->isArrayType()) {
    auto ArrType = Result.Context->getAsArrayType(QT);
    auto EleType = ArrType->getElementType();
    TypeStr = EleType.getAsString();
  } else {
    TypeStr = QT.getAsString();
  }

  if (TypeStr == "cuComplex" || TypeStr == "cuDoubleComplex") {
    SourceManager *SM = Result.SourceManager;
    FileID FID = SM->getDecomposedExpansionLoc(DD->getBeginLoc()).first;
    // Add '#include <complex>' directive to the file only once
    static std::set<FileID> ComplexHeaderFilter;
    SourceLocation IncludeLoc = IncludeLocations[FID];
    if (ComplexHeaderFilter.find(FID) == ComplexHeaderFilter.end()) {
      ComplexHeaderFilter.insert(FID);
      emplaceTransformation(new InsertText(
          IncludeLoc, getNL() + std::string("#include <complex>") + getNL()));
    }
  }

  auto Replacement = getReplacementForType(TypeStr);
  if (Replacement.empty())
    // TODO report migration error
    return;

  emplaceTransformation(new ReplaceTypeInDecl(DD, std::move(Replacement)));
  DupFilter.insert(Loc);
}

REGISTER_RULE(TypeInDeclRule)

// Rule for types replacements in template var declarations and field
// declarations
void TemplateTypeInDeclRule::registerMatcher(MatchFinder &MF) {
  auto Typedefs = typedefType(hasDeclaration(typedefDecl(TypedefNames)));

  auto EnumTypes = enumType(hasDeclaration(enumDecl(EnumTypeNames)));

  auto RecordTypes = recordType(hasDeclaration(cxxRecordDecl(RecordTypeNames)));

  auto HasCudaTemplateType =
      hasType(classTemplateSpecializationDecl(hasAnyTemplateArgument(
          refersToType(anyOf(Typedefs, EnumTypes, RecordTypes,
                             pointsTo(cxxRecordDecl(RecordTypeNames)))))));

  MF.addMatcher(
      varDecl(HasCudaTemplateType, unless(hasType(substTemplateTypeParmType())))
          .bind("TemplateTypeInVarDecl"),
      this);

  MF.addMatcher(fieldDecl(HasCudaTemplateType,
                          unless(hasType(substTemplateTypeParmType())))
                    .bind("TemplateTypeInFieldDecl"),
                this);
}

void TemplateTypeInDeclRule::run(const MatchFinder::MatchResult &Result) {
  // DD points to a VarDecl or a FieldDecl
  const DeclaratorDecl *DD =
      getNodeAsType<VarDecl>(Result, "TemplateTypeInVarDecl");
  QualType QT;
  if (DD)
    QT = DD->getType();
  else if ((DD = getNodeAsType<FieldDecl>(Result, "TemplateTypeInFieldDecl")))
    QT = DD->getType();
  else
    return;

  auto Loc =
      DD->getTypeSourceInfo()->getTypeLoc().getBeginLoc().getRawEncoding();
  if (DupFilter.find(Loc) != DupFilter.end())
    return;

  // std::vector<cudaStream_t> is elaborated to
  // std::vector<CUstream_st *, std::allocator<CUstream_st *>>
  bool isElaboratedType = false;
  if (auto ET = dyn_cast<ElaboratedType>(QT.getTypePtr())) {
    QT = ET->desugar();
    isElaboratedType = true;
  }
  if (auto TST = dyn_cast<TemplateSpecializationType>(QT.getTypePtr())) {
    for (unsigned i = 0; i < TST->getNumArgs(); ++i) {
      auto Args = TST->template_arguments();
      auto Arg = Args[i];
      QT = Arg.getAsType();
      auto TypeStr = QT.getAsString();
      auto Replacement = getReplacementForType(TypeStr);
      if (Replacement.empty())
        // TODO report migration error
        continue;

      auto DTL = DD->getTypeSourceInfo()->getTypeLoc();
      TemplateSpecializationTypeLoc TTTL;
      if (isElaboratedType) {
        auto ETL = DTL.getAs<ElaboratedTypeLoc>();
        TTTL = ETL.getNamedTypeLoc().getAs<TemplateSpecializationTypeLoc>();
      } else {
        TTTL = DTL.getAs<TemplateSpecializationTypeLoc>();
      }
      // Replace each type in the template arguments one by one
      auto TAL = TTTL.getArgLoc(i);
      emplaceTransformation(
          new ReplaceTypeInDecl(DD, TAL, std::move(Replacement)));
      DupFilter.insert(Loc);
    }
  }
}

REGISTER_RULE(TemplateTypeInDeclRule)

// Supported vector types
const std::unordered_set<std::string> SupportedVectorTypes{
    "char1",     "uchar1",     "char2",      "uchar2",     "char3",
    "uchar3",    "char4",      "uchar4",     "short1",     "ushort1",
    "short2",    "ushort2",    "short3",     "ushort3",    "short4",
    "ushort4",   "int1",       "uint1",      "int2",       "uint2",
    "int3",      "uint3",      "int4",       "uint4",      "long1",
    "ulong1",    "long2",      "ulong2",     "long3",      "ulong3",
    "long4",     "ulong4",     "float1",     "float2",     "float3",
    "float4",    "longlong1",  "ulonglong1", "longlong2",  "ulonglong2",
    "longlong3", "ulonglong3", "longlong4",  "ulonglong4", "double1",
    "double2",   "double3",    "double4"};

static internal::Matcher<NamedDecl> vectorTypeName() {
  std::vector<std::string> TypeNames(SupportedVectorTypes.begin(),
                                     SupportedVectorTypes.end());
  return internal::Matcher<NamedDecl>(new internal::HasNameMatcher(TypeNames));
}

namespace clang {
namespace ast_matchers {

AST_MATCHER(QualType, vectorType) {
  return (SupportedVectorTypes.find(Node.getAsString()) !=
          SupportedVectorTypes.end());
}

AST_MATCHER(TypedefDecl, typedefVecDecl) {
  if (!Node.getUnderlyingType().getBaseTypeIdentifier())
    return false;

  const std::string BaseTypeName =
      Node.getUnderlyingType().getBaseTypeIdentifier()->getName().str();
  return (SupportedVectorTypes.find(BaseTypeName) !=
          SupportedVectorTypes.end());
}

} // namespace ast_matchers
} // namespace clang

// Rule for types replacements in var. declarations.
void VectorTypeNamespaceRule::registerMatcher(MatchFinder &MF) {
  auto unlessMemory =
      unless(anyOf(hasAttr(attr::CUDAConstant), hasAttr(attr::CUDADevice),
                   hasAttr(attr::CUDAShared)));

  // basic: eg. int2 xx
  auto basicType = [&]() {
    return allOf(hasType(typedefDecl(vectorTypeName())),
                 unless(hasType(substTemplateTypeParmType())), unlessMemory);
  };

  // pointer: eg. int2 * xx
  auto ptrType = [&]() {
    return allOf(hasType(pointsTo(typedefDecl(vectorTypeName()))),
                 unlessMemory);
  };

  // array: eg. int2 array_[xx]
  auto arrType = [&]() {
    return allOf(hasType(arrayType(hasElementType(typedefType(
                     hasDeclaration(typedefDecl(vectorTypeName())))))),
                 unlessMemory);
  };

  // reference: eg int2 & xx
  auto referenceType = [&]() {
    return allOf(hasType(references(typedefDecl(vectorTypeName()))),
                 unlessMemory);
  };

  MF.addMatcher(
      varDecl(anyOf(basicType(), ptrType(), arrType(), referenceType()))
          .bind("vecVarDecl"),
      this);

  // typedef int2 xxx
  MF.addMatcher(typedefDecl(typedefVecDecl()).bind("typeDefDecl"), this);

  auto vectorTypeAccess = [&]() {
    return anyOf(vectorType(), references(vectorType()),
                 pointsTo(vectorType()));
  };

  // int2 func() => cl::sycl::int2 func()
  MF.addMatcher(
      functionDecl(returns(vectorTypeAccess())).bind("funcReturnsVectorType"),
      this);
}

bool VectorTypeNamespaceRule::isNamespaceInserted(SourceLocation SL) {
  unsigned int Key = SL.getRawEncoding();
  if (DupFilter.find(Key) == end(DupFilter)) {
    DupFilter.insert(Key);
    return false;
  } else {
    return true;
  }
}

void VectorTypeNamespaceRule::replaceTypeName(const QualType &QT,
                                              SourceLocation BeginLoc,
                                              bool isDeclType) {
  if (isNamespaceInserted(BeginLoc))
    return;

  CtTypeInfo Ty(QT);
  auto &TypeName = Ty.getOrginalBaseType();

  if (isDeclType)
    ++SrcAPIStaticsMap[TypeName];

  emplaceTransformation(
      new ReplaceToken(BeginLoc, std::string(MapNames::findReplacedName(
                                     MapNames::TypeNamesMap, TypeName))));
}

void VectorTypeNamespaceRule::run(const MatchFinder::MatchResult &Result) {
  // int2 => cl::sycl::int2
  if (const VarDecl *D = getNodeAsType<VarDecl>(Result, "vecVarDecl"))
    replaceTypeName(D->getType(),
                    D->getTypeSourceInfo()->getTypeLoc().getBeginLoc(), true);

  // typedef int2 xxx => typedef cl::sycl::int2 xxx
  if (const TypedefDecl *TD = getNodeAsType<TypedefDecl>(Result, "typeDefDecl"))
    replaceTypeName(TD->getUnderlyingType(),
                    TD->getTypeSourceInfo()->getTypeLoc().getBeginLoc());

  // int2 func() => cl::sycl::int2 func()
  if (const FunctionDecl *FD =
          getNodeAsType<FunctionDecl>(Result, "funcReturnsVectorType"))
    replaceTypeName(FD->getReturnType(),
                    FD->getReturnTypeSourceRange().getBegin());
}

REGISTER_RULE(VectorTypeNamespaceRule)

void VectorTypeMemberAccessRule::registerMatcher(MatchFinder &MF) {
  auto memberAccess = [&]() {
    return hasObjectExpression(hasType(qualType(hasCanonicalType(
        recordType(hasDeclaration(cxxRecordDecl(vectorTypeName())))))));
  };

  // int2.x => static_cast<int>(int2.x())
  MF.addMatcher(
      memberExpr(allOf(memberAccess(), unless(hasParent(binaryOperator(allOf(
                                           hasLHS(memberExpr(memberAccess())),
                                           isAssignmentOperator()))))))
          .bind("VecMemberExpr"),
      this);

  // int2.x += xxx => int2.x() += static_cast<int>(xxx)
  MF.addMatcher(
      binaryOperator(allOf(hasLHS(memberExpr(memberAccess())
                                      .bind("VecMemberExprAssignmentLHS")),
                           isAssignmentOperator()))
          .bind("VecMemberExprAssignment"),
      this);
}

void VectorTypeMemberAccessRule::renameMemberField(const MemberExpr *ME) {
  auto BaseTy = ME->getBase()->getType().getAsString();
  auto &SM = SyclctGlobalInfo::getSourceManager();
  if (*(BaseTy.end() - 1) == '1') {
    auto Begin = ME->getOperatorLoc();
    auto End = Lexer::getLocForEndOfToken(
        ME->getMemberLoc(), 0, SM,
        SyclctGlobalInfo::getContext().getLangOpts());
    auto Length = SM.getFileOffset(End) - SM.getFileOffset(Begin);
    return emplaceTransformation(new ReplaceText(Begin, Length, ""));
  }
  std::string MemberName = ME->getMemberNameInfo().getAsString();
  if (MapNames::replaceName(MemberNamesMap, MemberName))
    emplaceTransformation(
        new RenameFieldInMemberExpr(ME, std::move(MemberName)));
}

void VectorTypeMemberAccessRule::run(const MatchFinder::MatchResult &Result) {
  // xxx = int2.x => xxx = static_cast<int>(int2.x())
  if (const MemberExpr *ME =
          getNodeAsType<MemberExpr>(Result, "VecMemberExpr")) {

    std::ostringstream CastPrefix;
    CastPrefix << "static_cast<" << ME->getType().getAsString() << ">(";
    insertAroundStmt(ME, CastPrefix.str(), ")");
    renameMemberField(ME);
  }

  if (auto ME = getNodeAsType<MemberExpr>(Result, "VecMemberExprAssignmentLHS"))
    renameMemberField(ME);
}

REGISTER_RULE(VectorTypeMemberAccessRule)

namespace clang {
namespace ast_matchers {

AST_MATCHER(FunctionDecl, overloadedVectorOperator) {
  if (!SyclctGlobalInfo::isInRoot(Node.getBeginLoc()))
    return false;

  switch (Node.getOverloadedOperator()) {
  default: {
    return false;
  }
#define OVERLOADED_OPERATOR_MULTI(...)
#define OVERLOADED_OPERATOR(Name, ...)                                         \
  case OO_##Name: {                                                            \
    break;                                                                     \
  }
#include "clang/Basic/OperatorKinds.def"
#undef OVERLOADED_OPERATOR
#undef OVERLOADED_OPERATOR_MULTI
  }

  // Check parameter is vector type
  auto SupportedParamType = [&](const ParmVarDecl *PD) {
    assert(PD != nullptr);
    const IdentifierInfo *IDInfo =
        PD->getOriginalType().getBaseTypeIdentifier();
    if (!IDInfo)
      return false;

    const std::string TypeName = IDInfo->getName().str();
    return (SupportedVectorTypes.find(TypeName) != SupportedVectorTypes.end());
  };

  assert(Node.getNumParams() < 3);
  // As long as one parameter is vector type
  for (unsigned i = 0, End = Node.getNumParams(); i != End; ++i) {
    if (SupportedParamType(Node.getParamDecl(i))) {
      return true;
    }
  }

  return false;
}

} // namespace ast_matchers
} // namespace clang

void VectorTypeOperatorRule::registerMatcher(MatchFinder &MF) {
  auto vectorTypeOverLoadedOperator = [&]() {
    return functionDecl(overloadedVectorOperator());
  };

  // Matches user overloaded operator declaration
  MF.addMatcher(vectorTypeOverLoadedOperator().bind("overloadedOperatorDecl"),
                this);

  // Matches call of user overloaded operator
  MF.addMatcher(cxxOperatorCallExpr(callee(vectorTypeOverLoadedOperator()))
                    .bind("callOverloadedOperator"),
                this);
}

const char VectorTypeOperatorRule::NamespaceName[] =
    "syclct_operator_overloading";

void VectorTypeOperatorRule::TranslateOverloadedOperatorDecl(
    const MatchFinder::MatchResult &Result, const FunctionDecl *FD) {
  if (!FD)
    return;

  // Helper function to get the scope of function declartion
  // Eg:
  //
  //    void test();
  //   ^            ^
  //   |            |
  // Begin         End
  //
  //    void test() {}
  //   ^              ^
  //   |              |
  // Begin           End
  auto GetFunctionSourceRange = [&](const SourceManager &SM,
                                    const SourceLocation &StartLoc,
                                    const SourceLocation &EndLoc) {
    const std::pair<FileID, unsigned> StartLocInfo =
        SM.getDecomposedExpansionLoc(StartLoc);
    llvm::StringRef Buffer(SM.getCharacterData(EndLoc));
    const std::string Str = std::string(";") + getNL();
    size_t Offset = Buffer.find_first_of(Str);
    assert(Offset != llvm::StringRef::npos);
    const std::pair<FileID, unsigned> EndLocInfo =
        SM.getDecomposedExpansionLoc(EndLoc.getLocWithOffset(Offset + 1));
    assert(StartLocInfo.first == EndLocInfo.first);

    return SourceRange(
        SM.getComposedLoc(StartLocInfo.first, StartLocInfo.second),
        SM.getComposedLoc(EndLocInfo.first, EndLocInfo.second));
  };

  // Add namespace to user overloaded operator declaration
  // double2& operator+=(double2& lhs, const double2& rhs)
  // =>
  // namespace syclct_operator_overloading {
  //
  // double2& operator+=(double2& lhs, const double2& rhs)
  //
  // }
  const auto &SM = *Result.SourceManager;
  const std::string NL = getNL();

  std::ostringstream Prologue;
  // clang-format off
  Prologue << "namespace " << NamespaceName << " {" << NL
           << NL;
  // clang-format on

  std::ostringstream Epilogue;
  // clang-format off
  Epilogue << NL
           << "}  // namespace " << NamespaceName << NL
           << NL;
  // clang-format on

  const SourceRange SR =
      GetFunctionSourceRange(SM, FD->getBeginLoc(), FD->getEndLoc());
  report(SR.getBegin(), Diagnostics::TRNA_WARNING_OVERLOADED_API_FOUND);
  emplaceTransformation(new InsertText(SR.getBegin(), Prologue.str()));
  emplaceTransformation(new InsertText(SR.getEnd(), Epilogue.str()));
}

void VectorTypeOperatorRule::TranslateOverloadedOperatorCall(
    const MatchFinder::MatchResult &Result, const CXXOperatorCallExpr *CE) {
  if (!CE)
    return;

  // Explicitly call user overloaded operator
  //
  // For non-assignment operator:
  // a == b
  // =>
  // syclct_operator_overloading::operator==(a, b)
  //
  // For assignment operator:
  // a += b
  // =>
  // a = syclct_operator_overloading::operator+=(a, b)

  const std::string OperatorName = BinaryOperator::getOpcodeStr(
      BinaryOperator::getOverloadedOpcode(CE->getOperator()));

  std::ostringstream FuncCall;

  if (CE->isAssignmentOp()) {
    const auto &SM = *Result.SourceManager;
    const char *Start = SM.getCharacterData(CE->getBeginLoc());
    const char *End = SM.getCharacterData(CE->getOperatorLoc());
    const std::string LHSText(Start, End - Start);
    FuncCall << LHSText << " = ";
  }

  FuncCall << NamespaceName << "::operator" << OperatorName;

  std::string OperatorReplacement = (CE->getNumArgs() == 1)
                                        ? /* Unary operator */ ""
                                        : /* Binary operator */ ",";
  emplaceTransformation(
      new ReplaceToken(CE->getOperatorLoc(), std::move(OperatorReplacement)));
  insertAroundStmt(CE, FuncCall.str() + "(", ")");
}

void VectorTypeOperatorRule::run(const MatchFinder::MatchResult &Result) {
  // Add namespace to user overloaded operator declaration
  TranslateOverloadedOperatorDecl(
      Result, getNodeAsType<FunctionDecl>(Result, "overloadedOperatorDecl"));

  // Explicitly call user overloaded operator
  TranslateOverloadedOperatorCall(
      Result,
      getNodeAsType<CXXOperatorCallExpr>(Result, "callOverloadedOperator"));
}

REGISTER_RULE(VectorTypeOperatorRule)

void VectorTypeCtorRule::registerMatcher(MatchFinder &MF) {
  // Find sycl sytle vector:eg.int2 constructors which are part of different
  // casts (representing different syntaxes). This includes copy constructors.
  // All constructors will be visited once.
  MF.addMatcher(
      cxxConstructExpr(hasType(typedefDecl(vectorTypeName())),
                       hasParent(cxxFunctionalCastExpr().bind("CtorFuncCast"))),
      this);

  MF.addMatcher(cxxConstructExpr(hasType(typedefDecl(vectorTypeName())),
                                 hasParent(cStyleCastExpr().bind("CtorCCast"))),
                this);

  // (int2 *)&xxx;
  MF.addMatcher(cStyleCastExpr(hasType(pointsTo(typedefDecl(vectorTypeName()))))
                    .bind("PtrCast"),
                this);

  // make_int2
  auto makeVectorFunc = [&]() {
    std::vector<std::string> MakeVectorFuncNames;
    for (const std::string &TypeName : SupportedVectorTypes) {
      MakeVectorFuncNames.emplace_back("make_" + TypeName);
    }

    return internal::Matcher<NamedDecl>(
        new internal::HasNameMatcher(MakeVectorFuncNames));
  };

  // migrate utility for vector type: eg: make_int2
  MF.addMatcher(
      callExpr(callee(functionDecl(makeVectorFunc()))).bind("VecUtilFunc"),
      this);

  // sizeof(int2)
  MF.addMatcher(
      unaryExprOrTypeTraitExpr(allOf(hasArgumentOfType(vectorType()),
                                     has(qualType(hasCanonicalType(type())))))
          .bind("Sizeof"),
      this);
}

std::string
VectorTypeCtorRule::getReplaceTypeName(const std::string &TypeName) {
  return std::string(
      MapNames::findReplacedName(MapNames::TypeNamesMap, TypeName));
}

// Determines which case of construction applies and creates replacements for
// the syntax. Returns the constructor node and a boolean indicating if a
// closed brace needs to be appended.
void VectorTypeCtorRule::run(const MatchFinder::MatchResult &Result) {
  // Most commonly used syntax cases are checked first.
  if (auto Cast =
          getNodeAsType<CXXFunctionalCastExpr>(Result, "CtorFuncCast")) {
    // int2 a = int2(1); // function style cast
    // int2 b = int2(a); // copy constructor
    // func(int(1), int2(a));
    emplaceTransformation(
        new ReplaceToken(Cast->getBeginLoc(),
                         getReplaceTypeName(Cast->getType().getAsString())));
  }

  if (auto Cast = getNodeAsType<CStyleCastExpr>(Result, "CtorCCast")) {
    // int2 a = (int2)1;
    // int2 b = (int2)a; // copy constructor
    // func((int2)1, (int2)a);
    emplaceTransformation(new ReplaceCCast(
        Cast, "(" + getReplaceTypeName(Cast->getType().getAsString()) + ")"));
    return;
  }

  if (const CallExpr *CE = getNodeAsType<CallExpr>(Result, "VecUtilFunc")) {
    if (!CE->getDirectCallee())
      return;
    const llvm::StringRef FuncName = CE->getDirectCallee()->getName();
    assert(FuncName.startswith("make_") &&
           "Found non make_<vector type> function");
    emplaceTransformation(new ReplaceStmt(
        CE->getCallee(), getReplaceTypeName(CE->getType().getAsString())));
    return;
  }

  if (const CStyleCastExpr *CPtrCast =
          getNodeAsType<CStyleCastExpr>(Result, "PtrCast")) {
    emplaceTransformation(new ReplaceToken(
        CPtrCast->getLParenLoc().getLocWithOffset(1),
        getReplaceTypeName(
            CPtrCast->getType()->getPointeeType().getAsString())));
    return;
  }

  if (const UnaryExprOrTypeTraitExpr *ExprSizeof =
          getNodeAsType<UnaryExprOrTypeTraitExpr>(Result, "Sizeof")) {
    if (ExprSizeof->isArgumentType()) {
      emplaceTransformation(new ReplaceToken(
          ExprSizeof->getArgumentTypeInfo()->getTypeLoc().getBeginLoc(),
          getReplaceTypeName(ExprSizeof->getArgumentType().getAsString())));
    }
    return;
  }
}

REGISTER_RULE(VectorTypeCtorRule)

void ReplaceDim3CtorRule::registerMatcher(MatchFinder &MF) {
  // Find dim3 constructors which are part of different casts (representing
  // different syntaxes). This includes copy constructors. All constructors
  // will be visited once.
  MF.addMatcher(cxxConstructExpr(hasType(typedefDecl(hasName("dim3"))),
                                 argumentCountIs(1),
                                 unless(hasAncestor(cxxConstructExpr(
                                     hasType(typedefDecl(hasName("dim3")))))))
                    .bind("dim3Top"),
                this);

  MF.addMatcher(cxxConstructExpr(hasType(typedefDecl(hasName("dim3"))),
                                 argumentCountIs(3), hasParent(varDecl()),
                                 unless(hasAncestor(cxxConstructExpr(
                                     hasType(typedefDecl(hasName("dim3")))))))
                    .bind("dim3CtorDecl"),
                this);

  MF.addMatcher(
      cxxConstructExpr(
          hasType(typedefDecl(hasName("dim3"))), argumentCountIs(3),
          // skip fields in a struct.  The source loc is
          // messed up (points to the start of the struct)
          unless(hasAncestor(cxxRecordDecl())), unless(hasParent(varDecl())),
          unless(hasAncestor(
              cxxConstructExpr(hasType(typedefDecl(hasName("dim3")))))))
          .bind("dim3CtorNoDecl"),
      this);
}

ReplaceDim3Ctor *ReplaceDim3CtorRule::getReplaceDim3Modification(
    const MatchFinder::MatchResult &Result) {
  if (auto Ctor = getNodeAsType<CXXConstructExpr>(Result, "dim3CtorDecl")) {
    // dim3 a(1);
    if (Ctor->getParenOrBraceRange().isInvalid()) {
      // dim3 a;
      // No replacements are needed
      return nullptr;
    } else {
      // dim3 a(1);
      return new ReplaceDim3Ctor(Ctor, true /*isDecl*/);
    }
  } else if (auto Ctor =
                 getNodeAsType<CXXConstructExpr>(Result, "dim3CtorNoDecl")) {
    return new ReplaceDim3Ctor(Ctor);
  } else if (auto Ctor = getNodeAsType<CXXConstructExpr>(Result, "dim3Top")) {
    if (auto A = ReplaceDim3Ctor::getConstructExpr(Ctor->getArg(0))) {
      // strip the top CXXConstructExpr, if there's a CXXConstructExpr further
      // down
      return new ReplaceDim3Ctor(Ctor, A);
    } else {
      // Copy constructor case: dim3 a(copyfrom)
      // No replacements are needed
      return nullptr;
    }
  }
  return nullptr;
}

void ReplaceDim3CtorRule::run(const MatchFinder::MatchResult &Result) {
  ReplaceDim3Ctor *R = getReplaceDim3Modification(Result);
  if (R) {
    // add a transformation that will filter out all nested transformations
    emplaceTransformation(R->getEmpty());
    // all the nested transformations will be applied when R->getReplacement()
    // is called
    emplaceTransformation(R);
  }
}

REGISTER_RULE(ReplaceDim3CtorRule)

void Dim3MemberFieldsRule::FieldsRename(const MatchFinder::MatchResult &Result,
                                        std::string Str, const MemberExpr *ME) {
  auto SM = Result.SourceManager;
  SourceLocation Begin = SM->getSpellingLoc(ME->getBeginLoc());
  SourceLocation End = SM->getSpellingLoc(ME->getEndLoc());
  std::string Ret =
      std::string(SM->getCharacterData(Begin), SM->getCharacterData(End));

  std::size_t Position = std::string::npos;
  std::size_t Current = Ret.find(Str);

  // Find the last position of dot '.'
  while (Current != std::string::npos) {
    Position = Current;
    Current = Ret.find(Str, Position + 1);
  }

  if (Position != std::string::npos) {
    auto Search = MapNames::Dim3MemberNamesMap.find(
        ME->getMemberNameInfo().getAsString());
    if (Search != MapNames::Dim3MemberNamesMap.end()) {
      emplaceTransformation(
          new RenameFieldInMemberExpr(ME, Search->second + "", Position));
      std::string NewMemberStr = Ret.substr(0, Position) + Search->second;
    }
  }
}

// rule for dim3 types member fields replacements.
void Dim3MemberFieldsRule::registerMatcher(MatchFinder &MF) {
  // dim3->x/y/z => dim3->operator[](0)/(1)/(2)
  MF.addMatcher(
      memberExpr(
          has(implicitCastExpr(hasType(pointsTo(typedefDecl(hasName("dim3")))))
                  .bind("ImplCast")))
          .bind("Dim3MemberPointerExpr"),
      this);

  // dim3.x/y/z => dim3[0]/[1]/[2]
  MF.addMatcher(
      memberExpr(
          hasObjectExpression(hasType(qualType(hasCanonicalType(
              recordType(hasDeclaration(cxxRecordDecl(hasName("dim3")))))))))
          .bind("Dim3MemberDotExpr"),
      this);
}

void Dim3MemberFieldsRule::run(const MatchFinder::MatchResult &Result) {
  if (const MemberExpr *ME =
          getNodeAsType<MemberExpr>(Result, "Dim3MemberPointerExpr")) {
    // E.g.
    // dim3 *pd3;
    // pd3->x;
    // will migrate to:
    // cl::sycl::range<3> *pd3;
    // (*pd3)[0];
    auto Impl = getAssistNodeAsType<ImplicitCastExpr>(Result, "ImplCast");
    insertAroundStmt(Impl, "(*", ")");
    FieldsRename(Result, "->", ME);
  }

  if (const MemberExpr *ME =
          getNodeAsType<MemberExpr>(Result, "Dim3MemberDotExpr")) {
    FieldsRename(Result, ".", ME);
  }
}

REGISTER_RULE(Dim3MemberFieldsRule)

// Rule for return types replacements.
void ReturnTypeRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      functionDecl(
          returns(hasCanonicalType(
              anyOf(recordType(hasDeclaration(
                        cxxRecordDecl(hasName("cudaDeviceProp")))),
                    enumType(hasDeclaration(enumDecl(hasName("cudaError"))))))))
          .bind("functionDecl"),
      this);
}

void ReturnTypeRule::run(const MatchFinder::MatchResult &Result) {
  const FunctionDecl *FD = getNodeAsType<FunctionDecl>(Result, "functionDecl");
  if (!FD)
    return;
  const clang::Type *Type = FD->getReturnType().getTypePtr();
  std::string TypeName =
      Type->getCanonicalTypeInternal().getBaseTypeIdentifier()->getName().str();

  SrcAPIStaticsMap[TypeName]++;

  auto Search = MapNames::TypeNamesMap.find(TypeName);
  if (Search == MapNames::TypeNamesMap.end()) {
    // TODO report migration error
    return;
  }
  std::string Replacement = Search->second;
  emplaceTransformation(new ReplaceReturnType(FD, std::move(Replacement)));
}

REGISTER_RULE(ReturnTypeRule)

// Rule for cudaDeviceProp variables.
void DevicePropVarRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      memberExpr(
          hasObjectExpression(hasType(qualType(hasCanonicalType(recordType(
              hasDeclaration(cxxRecordDecl(hasName("cudaDeviceProp")))))))))
          .bind("DevicePropVar"),
      this);
}

void DevicePropVarRule::run(const MatchFinder::MatchResult &Result) {
  const MemberExpr *ME = getNodeAsType<MemberExpr>(Result, "DevicePropVar");
  if (!ME)
    return;
  auto Parents = Result.Context->getParents(*ME);
  assert(Parents.size() == 1);
  if(Parents.size() != 1) {
      return;
  }
  auto Search = PropNamesMap.find(ME->getMemberNameInfo().getAsString());
  if (Search == PropNamesMap.end()) {
    // TODO report migration error
    return;
  }
  if(Parents[0].get<clang::ImplicitCastExpr>()) {
    // migrate to get_XXX() eg. "b=a.minor" to "b=a.get_minor_version()"
    emplaceTransformation(new RenameFieldInMemberExpr(ME, "get_" + Search->second + "()"));
  } else if (auto *BO = Parents[0].get<clang::BinaryOperator>()) {
    // migrate to set_XXX() eg. "a.minor = 1" to "a.set_minor_version(1)"
    if(BO->getOpcode()== clang::BO_Assign) {
        emplaceTransformation(new RenameFieldInMemberExpr(ME, "set_" + Search->second ));
        emplaceTransformation(new ReplaceText(BO->getOperatorLoc(), 1, "("));
        emplaceTransformation(new InsertAfterStmt(BO, ")"));
    }
  }
  if ((Search->second.compare(0, 13, "major_version") == 0) ||
      (Search->second.compare(0, 13, "minor_version") == 0)) {
    report(ME->getBeginLoc(), Comments::VERSION_COMMENT);
  }
  if (Search->second.compare(0, 10, "integrated") == 0) {
    report(ME->getBeginLoc(), Comments::NOT_SUPPORT_API_INTEGRATEDORNOT);
  }
}

REGISTER_RULE(DevicePropVarRule)

// Rule for enums constants.
void EnumConstantRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(declRefExpr(to(enumConstantDecl(
                                hasType(enumDecl(hasName("cudaComputeMode"))))))
                    .bind("EnumConstant"),
                this);
}

void EnumConstantRule::run(const MatchFinder::MatchResult &Result) {
  const DeclRefExpr *E = getNodeAsType<DeclRefExpr>(Result, "EnumConstant");
  if (!E)
    return;
  assert(E && "Unknown result");
  auto Search = EnumNamesMap.find(E->getNameInfo().getName().getAsString());
  if (Search == EnumNamesMap.end()) {
    // TODO report migration error
    return;
  }
  emplaceTransformation(new ReplaceStmt(E, "syclct::" + Search->second));
}

REGISTER_RULE(EnumConstantRule)

void ErrorConstantsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      declRefExpr(to(enumConstantDecl(hasType(enumDecl(hasName("cudaError"))))))
          .bind("ErrorConstants"),
      this);
}

void ErrorConstantsRule::run(const MatchFinder::MatchResult &Result) {
  const DeclRefExpr *DE = getNodeAsType<DeclRefExpr>(Result, "ErrorConstants");
  if (!DE)
    return;
  assert(DE && "Unknown result");
  auto *EC = cast<EnumConstantDecl>(DE->getDecl());
  emplaceTransformation(new ReplaceStmt(DE, EC->getInitVal().toString(10)));
}

REGISTER_RULE(ErrorConstantsRule)

// Rule for BLAS status enum constants.
// All status enum constants have the prefix CUBLAS_STATUS
// Example: migrate CUBLAS_STATUS_SUCCESS to 0
void BLASStatusRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      declRefExpr(to(enumConstantDecl(matchesName("CUBLAS_STATUS.*"))))
          .bind("BLASStatusConstants"),
      this);
}

void BLASStatusRule::run(const MatchFinder::MatchResult &Result) {
  const DeclRefExpr *DE =
      getNodeAsType<DeclRefExpr>(Result, "BLASStatusConstants");
  if (!DE)
    return;
  assert(DE && "Unknown result");
  auto *EC = cast<EnumConstantDecl>(DE->getDecl());
  emplaceTransformation(new ReplaceStmt(DE, EC->getInitVal().toString(10)));
}

REGISTER_RULE(BLASStatusRule)

// Rule for BLAS opertaion enum constants.
void BLASOperationRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(declRefExpr(to(enumConstantDecl(matchesName("CUBLAS_OP.*"))))
                    .bind("BLASOperationConstants"),
                this);
}

void BLASOperationRule::run(const MatchFinder::MatchResult &Result) {
  const DeclRefExpr *DE =
      getNodeAsType<DeclRefExpr>(Result, "BLASOperationConstants");
  if (!DE)
    return;
  assert(DE && "Unknown result");
  auto *EC = cast<EnumConstantDecl>(DE->getDecl());
  std::string Name = EC->getNameAsString();
  if ("CUBLAS_OP_N" == Name) {
    emplaceTransformation(new ReplaceStmt(DE, "mkl::transpose::nontrans"));
  }
  if ("CUBLAS_OP_T" == Name) {
    emplaceTransformation(new ReplaceStmt(DE, "mkl::transpose::trans"));
  }
  if ("CUBLAS_OP_C" == Name) {
    emplaceTransformation(new ReplaceStmt(DE, "mkl::transpose::conjtrans"));
  }
}

REGISTER_RULE(BLASOperationRule)

void FunctionCallRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName(
        "cudaGetDeviceCount", "cudaGetDeviceProperties", "cudaDeviceReset",
        "cudaSetDevice", "cudaDeviceGetAttribute", "cudaDeviceGetP2PAttribute",
        "cudaDeviceGetPCIBusId",
        "cudaGetDevice", "cudaDeviceSetLimit", "cudaGetLastError",
        "cudaPeekAtLastError", "cudaDeviceSynchronize", "cudaThreadSynchronize",
        "cudaGetErrorString", "cudaGetErrorName", "cudaDeviceSetCacheConfig",
        "cudaDeviceGetCacheConfig", "clock", "cudaThreadSetLimit",
        "cublasSgemm_v2", "cublasDgemm_v2", "cublasCgemm_v2", "cublasZgemm_v2",
        "cublasCreate_v2", "cublasDestroy_v2");
  };

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(functionName())), parentStmt))
          .bind("FunctionCall"),
      this);
  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(functionName())), unless(parentStmt)))
          .bind("FunctionCallUsed"),
      this);
}

void FunctionCallRule::run(const MatchFinder::MatchResult &Result) {
  bool IsAssigned = false;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "FunctionCall");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "FunctionCallUsed")))
      return;
    IsAssigned = true;
  }
  assert(CE && "Unknown result");

  if (!CE->getDirectCallee())
    return;
  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();

  std::string Prefix = "";
  std::string Poststr = "";
  if (IsAssigned) {
    Prefix = "(";
    Poststr = ", 0)";
  }

  if (FuncName == "cudaGetDeviceCount") {
    std::string ResultVarName = DereferenceArg(CE->getArg(0), *Result.Context);
    emplaceTransformation(new InsertBeforeStmt(CE, ResultVarName + " = "));
    emplaceTransformation(
        new ReplaceStmt(CE, "syclct::get_device_manager().device_count()"));
  } else if (FuncName == "cudaGetDeviceProperties") {
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
    }
    std::string ResultVarName = DereferenceArg(CE->getArg(0), *Result.Context);
    emplaceTransformation(new ReplaceStmt(
        CE->getCallee(), Prefix + "syclct::get_device_manager().get_device"));
    emplaceTransformation(new RemoveArg(CE, 0));
    emplaceTransformation(new InsertAfterStmt(
        CE, ".get_device_info(" + ResultVarName + ")" + Poststr));
  } else if (FuncName == "cudaDeviceReset") {
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
    }
    emplaceTransformation(new ReplaceStmt(
        CE, Prefix + "syclct::get_device_manager().current_device().reset()" +
                Poststr));
  } else if (FuncName == "cudaSetDevice") {
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
    }
    emplaceTransformation(
        new ReplaceStmt(CE->getCallee(),
                        Prefix + "syclct::get_device_manager().select_device"));
    if (IsAssigned)
      emplaceTransformation(new InsertAfterStmt(CE, ", 0)"));

  } else if (FuncName == "cudaDeviceGetAttribute") {
    std::string ResultVarName = DereferenceArg(CE->getArg(0), *Result.Context);
    std::string AttributeName = ((const clang::DeclRefExpr *)CE->getArg(1))
                                    ->getNameInfo()
                                    .getName()
                                    .getAsString();
    auto Search = EnumConstantRule::EnumNamesMap.find(AttributeName);
    if (Search == EnumConstantRule::EnumNamesMap.end()) {
      // TODO report migration error
      return;
    }
    emplaceTransformation(new InsertBeforeStmt(CE, ResultVarName + " = "));
    emplaceTransformation(new ReplaceStmt(
        CE->getCallee(), "syclct::get_device_manager().get_device"));
    emplaceTransformation(new RemoveArg(CE, 0));
    emplaceTransformation(new RemoveArg(CE, 1));
    emplaceTransformation(new InsertAfterStmt(CE, "." + Search->second + "()"));
  } else if (FuncName == "cudaDeviceGetP2PAttribute") {
    std::string ResultVarName = DereferenceArg(CE->getArg(0), *Result.Context);
    emplaceTransformation(new ReplaceStmt(CE, ResultVarName + " = 0"));
    report(CE->getBeginLoc(), Comments::NOTSUPPORTED, "P2P Access");
  } else if(FuncName == "cudaDeviceGetPCIBusId") {
      report(CE->getBeginLoc(), Comments::NOTSUPPORTED, "Get PCI BusId");
  }else if (FuncName == "cudaGetDevice") {
    std::string ResultVarName = DereferenceArg(CE->getArg(0), *Result.Context);
    emplaceTransformation(new InsertBeforeStmt(CE, ResultVarName + " = "));
    emplaceTransformation(new ReplaceStmt(
        CE, "syclct::get_device_manager().current_device_id()"));
  } else if (FuncName == "cudaDeviceSynchronize" ||
             FuncName == "cudaThreadSynchronize") {
    std::string ReplStr = "syclct::get_device_manager()."
                          "current_device().queues_wait_"
                          "and_throw()";
    if (IsAssigned) {
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
    }
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));

  } else if (FuncName == "cudaGetLastError" ||
             FuncName == "cudaPeekAtLastError") {
    report(CE->getBeginLoc(),
           Comments::TRNA_WARNING_ERROR_HANDLING_API_REPLACED_0, FuncName);
    emplaceTransformation(new ReplaceStmt(CE, "0"));
  } else if (FuncName == "cudaGetErrorString" ||
             FuncName == "cudaGetErrorName") {
    // Insert warning messages into the spelling locations in case
    // that these functions are contained in macro definitions
    auto Loc = Result.SourceManager->getSpellingLoc(CE->getBeginLoc());
    report(Loc, Comments::TRNA_WARNING_ERROR_HANDLING_API_COMMENTED, FuncName);
    emplaceTransformation(
        new InsertBeforeStmt(CE, "\"" + FuncName + " not supported\"/*"));
    emplaceTransformation(new InsertAfterStmt(CE, "*/"));
  } else if (FuncName == "cudaDeviceSetCacheConfig" ||
             FuncName == "cudaDeviceGetCacheConfig") {
    // SYCL has no corresponding implementation for
    // "cudaDeviceSetCacheConfig/cudaDeviceGetCacheConfig", so simply migrate
    // "cudaDeviceSetCacheConfig/cudaDeviceGetCacheConfig" into expression "0;".
    std::string Replacement = "0";
    emplaceTransformation(new ReplaceStmt(CE, std::move(Replacement)));
  } else if (FuncName == "clock") {
    report(CE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED_SYCL_UNDEF);
    // Add '#include <time.h>' directive to the file only once
    static std::set<FileID> TimeHeaderFilter;
    auto Loc = CE->getBeginLoc();
    auto FID = Result.SourceManager->getDecomposedExpansionLoc(Loc).first;
    auto IncludeLoc = IncludeLocations[FID];
    if (TimeHeaderFilter.find(FID) == TimeHeaderFilter.end()) {
      TimeHeaderFilter.insert(FID);
      emplaceTransformation(new InsertText(
          IncludeLoc, getNL() +
                          std::string("#include <time.h> // For clock_t, "
                                      "clock and CLOCKS_PER_SEC") +
                          getNL()));
    }
  } else if (FuncName == "cudaDeviceSetLimit" ||
             FuncName == "cudaThreadSetLimit") {
    report(CE->getBeginLoc(), Diagnostics::NOTSUPPORTED, FuncName);
    emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ""));
  } else if (FuncName == "cublasSgemm_v2" || FuncName == "cublasDgemm_v2" ||
             FuncName == "cublasCgemm_v2" || FuncName == "cublasZgemm_v2") {
    // There are some macros like "#define cublasSgemm cublasSgemm_v2"
    // in "cublas_v2.h", so the function names we match should with the
    // suffix "_v2".
    const SourceManager *SM = Result.SourceManager;
    SourceLocation FuncNameBegin(CE->getBeginLoc());
    SourceLocation FuncCallEnd(CE->getEndLoc());
    if (FuncNameBegin.isMacroID())
      FuncNameBegin = SM->getExpansionLoc(FuncNameBegin);
    if (FuncCallEnd.isMacroID())
      FuncCallEnd = SM->getExpansionLoc(FuncCallEnd);
    Token Tok;
    Lexer::getRawToken(FuncNameBegin, Tok, *SM, LangOptions());
    SourceLocation FuncNameEnd = Tok.getEndLoc();
    auto FuncNameLength =
        SM->getCharacterData(FuncNameEnd) - SM->getCharacterData(FuncNameBegin);
    auto Search = MapNames::BLASFunctionNamesMap.find(FuncName);
    if (Search == MapNames::BLASFunctionNamesMap.end()) {
      // TODO report migration error
      return;
    }
    std::string Replacement = Search->second;
    if (IsAssigned) {
      emplaceTransformation(new InsertText(FuncNameBegin, "("));
      emplaceTransformation(
          new InsertText(FuncCallEnd.getLocWithOffset(1), ", 0)"));
      report(FuncNameBegin, Diagnostics::NOERROR_RETURN_COMMA_OP);
    }
    emplaceTransformation(
        new ReplaceText(FuncNameBegin, FuncNameLength, std::move(Replacement)));

  } else if (FuncName == "cublasCreate_v2" || FuncName == "cublasDestroy_v2") {
    // Remove these two function calls.
    // There are some macros like "#define cublasCreate cublasCreate_v2"
    // in "cublas_v2.h", so the function names we match should with the
    // suffix "_v2".
    if (IsAssigned) {
      emplaceTransformation(
          new ReplaceStmt(CE, /*IsReplaceCompatibilityAPI*/ true, FuncName,
                          /*IsProcessMacro*/ true, "0"));
    } else {
      emplaceTransformation(
          new ReplaceStmt(CE, /*IsReplaceCompatibilityAPI*/ true, FuncName,
                          /*IsProcessMacro*/ true, ""));
    }
  } else {
    syclct_unreachable("Unknown function name");
  }
}

REGISTER_RULE(FunctionCallRule)

void EventAPICallRule::registerMatcher(MatchFinder &MF) {
  auto eventAPIName = [&]() {
    return hasAnyName("cudaEventCreate", "cudaEventCreateWithFlags",
                      "cudaEventDestroy", "cudaEventRecord",
                      "cudaEventElapsedTime", "cudaEventSynchronize");
  };

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(eventAPIName())), parentStmt))
          .bind("eventAPICall"),
      this);
  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(eventAPIName())), unless(parentStmt)))
          .bind("eventAPICallUsed"),
      this);
}

void EventAPICallRule::run(const MatchFinder::MatchResult &Result) {
  bool IsAssigned = false;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "eventAPICall");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "eventAPICallUsed")))
      return;
    IsAssigned = true;
  }
  assert(CE && "Unknown result");

  if (!CE->getDirectCallee())
    return;
  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();

  if (FuncName == "cudaEventCreate" || FuncName == "cudaEventCreateWithFlags" ||
      FuncName == "cudaEventDestroy") {
    std::string ReplStr;
    if (IsAssigned) {
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
      emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
    } else {
      emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
    }
  } else if (FuncName == "cudaEventRecord") {
    handleEventRecord(CE, Result, IsAssigned);
  } else if (FuncName == "cudaEventElapsedTime") {
    handleEventElapsedTime(CE, Result, IsAssigned);
  } else if (FuncName == "cudaEventSynchronize") {
    std::string ReplStr{getStmtSpelling(CE->getArg(0), *Result.Context)};
    ReplStr += ".wait_and_throw()";
    if (IsAssigned) {
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
    }
    emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
  } else {
    syclct_unreachable("Unknown function name");
  }
}

void EventAPICallRule::handleEventRecord(const CallExpr *CE,
                                         const MatchFinder::MatchResult &Result,
                                         bool IsAssigned) {
  report(CE->getBeginLoc(), Diagnostics::TIME_MEASUREMENT_FOUND);
  std::string ReplStr;

  // Define the helper variable if it is used in the block for first time,
  // otherwise, just use it.
  static std::set<std::pair<const CompoundStmt *, const std::string>> DupFilter;
  const auto *CS = findImmediateBlock(CE);
  auto StmtStr = getStmtSpelling(CE->getArg(0), *Result.Context);
  auto Pair = std::make_pair(CS, StmtStr);

  if (DupFilter.find(Pair) == DupFilter.end()) {
    DupFilter.insert(Pair);
    ReplStr += "auto ";
  }

  ReplStr += "syclct_";
  ReplStr += StmtStr;
  ReplStr += "_";
  ReplStr += getHashAsString(StmtStr).substr(0, 6);
  ReplStr += " = clock()";
  if (IsAssigned) {
    ReplStr = "(" + ReplStr + ", 0)";
    report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
  }
  const std::string Name =
      CE->getCalleeDecl()->getAsFunction()->getNameAsString();
  emplaceTransformation(new ReplaceStmt(CE, true, Name, ReplStr));
}

void EventAPICallRule::handleEventElapsedTime(
    const CallExpr *CE, const MatchFinder::MatchResult &Result,
    bool IsAssigned) {
  auto StmtStrArg0 = getStmtSpelling(CE->getArg(0), *Result.Context);
  auto StmtStrArg1 = getStmtSpelling(CE->getArg(1), *Result.Context);
  auto StmtStrArg2 = getStmtSpelling(CE->getArg(2), *Result.Context);
  std::string ReplStr{"*("};
  ReplStr += StmtStrArg0;
  ReplStr += ") = (float)(syclct_";
  ReplStr += StmtStrArg2;
  ReplStr += "_";
  ReplStr += getHashAsString(StmtStrArg2).substr(0, 6);
  ReplStr += " - syclct_";
  ReplStr += StmtStrArg1;
  ReplStr += "_";
  ReplStr += getHashAsString(StmtStrArg1).substr(0, 6);
  ReplStr += ") / CLOCKS_PER_SEC * 1000";
  if (IsAssigned) {
    ReplStr = "(" + ReplStr + ", 0)";
    report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
  }
  const std::string Name =
      CE->getCalleeDecl()->getAsFunction()->getNameAsString();
  emplaceTransformation(new ReplaceStmt(CE, true, Name, ReplStr));
  handleTimeMeasurement(CE, Result);
}

void EventAPICallRule::handleTimeMeasurement(
    const CallExpr *CE, const MatchFinder::MatchResult &Result) {
  auto CELoc = CE->getBeginLoc().getRawEncoding();
  auto Parents = Result.Context->getParents(*CE);
  assert(Parents.size() == 1);
  auto *Parent = Parents[0].get<Stmt>();
  if (!Parent) {
    return;
  }
  const CallExpr *RecordBegin = nullptr, *RecordEnd = nullptr;
  // Find the last cudaEventRecord call on start and stop
  for (auto Iter = Parent->child_begin(); Iter != Parent->child_end(); ++Iter) {
    if (Iter->getBeginLoc().getRawEncoding() > CELoc)
      continue;

    if (const CallExpr *RecordCall = dyn_cast<CallExpr>(*Iter)) {
      if (!RecordCall->getDirectCallee())
        return;
      std::string RecordFuncName =
          RecordCall->getDirectCallee()->getNameInfo().getName().getAsString();
      // Find the last call of cudaEventRecord on start and stop before
      // call to cudaElpasedTime
      if (RecordFuncName == "cudaEventRecord") {
        auto Arg0 = getStmtSpelling(RecordCall->getArg(0), *Result.Context);
        if (Arg0 == getStmtSpelling(CE->getArg(1), *Result.Context))
          RecordBegin = RecordCall;
        else if (Arg0 == getStmtSpelling(CE->getArg(2), *Result.Context))
          RecordEnd = RecordCall;
      }
    }
  }
  if (!RecordBegin || !RecordEnd)
    return;

  // Find the kernel calls between start and stop
  auto RecordBeginLoc = RecordBegin->getBeginLoc().getRawEncoding();
  auto RecordEndLoc = RecordEnd->getBeginLoc().getRawEncoding();
  for (auto Iter = Parent->child_begin(); Iter != Parent->child_end(); ++Iter) {
    if (auto *Expr = dyn_cast<ExprWithCleanups>(*Iter)) {
      auto *SubExpr = Expr->getSubExpr();
      if (auto *KCall = dyn_cast<CUDAKernelCallExpr>(SubExpr)) {
        auto KCallLoc = KCall->getBeginLoc().getRawEncoding();
        // Only the kernel calls between begin and end are set to be synced
        if (KCallLoc > RecordBeginLoc && KCallLoc < RecordEndLoc) {
          SyclctGlobalInfo::getInstance()
              .insertKernelCallExpr(KCall)
              ->setSync();
        }
      }
    }
  }
}

REGISTER_RULE(EventAPICallRule)

void StreamAPICallRule::registerMatcher(MatchFinder &MF) {
  auto streamFunctionName = [&]() {
    return hasAnyName("cudaStreamCreate", "cudaStreamCreateWithFlags",
                      "cudaStreamCreateWithPriority", "cudaStreamDestroy",
                      "cudaStreamSynchronize", "cudaStreamGetPriority",
                      "cudaStreamGetFlags", "cudaDeviceGetStreamPriorityRange",
                      "cudaStreamAttachMemAsync", "cudaStreamBeginCapture",
                      "cudaStreamEndCapture", "cudaStreamIsCapturing",
                      "cudaStreamQuery", "cudaStreamWaitEvent",
                      "cudaStreamAddCallback");
  };

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(streamFunctionName())), parentStmt))
          .bind("streamAPICall"),
      this);
  MF.addMatcher(callExpr(allOf(callee(functionDecl(streamFunctionName())),
                               unless(parentStmt)))
                    .bind("streamAPICallUsed"),
                this);
}

void StreamAPICallRule::run(const MatchFinder::MatchResult &Result) {
  bool IsAssigned = false;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "streamAPICall");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "streamAPICallUsed")))
      return;
    IsAssigned = true;
  }
  assert(CE && "Unknown result");

  if (!CE->getDirectCallee())
    return;
  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();

  if (FuncName == "cudaStreamCreate" ||
      FuncName == "cudaStreamCreateWithFlags" ||
      FuncName == "cudaStreamCreateWithPriority" ||
      FuncName == "cudaStreamDestroy") {
    auto Arg0 = CE->getArg(0);
    auto DRE = getInnerValueDecl(Arg0);
    std::string ReplStr;
    // If DRE is nullptr which means the stream is a complex expression
    // (e.g. iterator getters), skip the scope analysis
    if (DRE && isInSameScope(CE, DRE->getDecl())) {
      if (IsAssigned) {
        ReplStr = "(0, 0)";
        report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
      }
    } else {
      auto StmtStr0 = getStmtSpelling(CE->getArg(0), *Result.Context);
      // TODO: simplify expression
      if (FuncName == "cudaStreamDestroy") {
        ReplStr = StmtStr0;
      } else {
        if (StmtStr0[0] == '&')
          ReplStr = StmtStr0.substr(1);
        else
          ReplStr = "*(" + StmtStr0 + ")";
      }

      ReplStr += " = cl::sycl::queue{}";
      if (IsAssigned) {
        ReplStr = "(" + ReplStr + ", 0)";
        report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
      }
    }
    emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
    if (FuncName == "cudaStreamCreateWithFlags" ||
        FuncName == "cudaStreamCreateWithPriority") {
      report(CE->getBeginLoc(),
             Diagnostics::STREAM_FLAG_PRIORITY_NOT_SUPPORTED);
    }
  } else if (FuncName == "cudaStreamSynchronize") {
    auto StmtStr = getStmtSpelling(CE->getArg(0), *Result.Context);
    std::string ReplStr{StmtStr};
    ReplStr += ".wait()";
    const std::string Name =
        CE->getCalleeDecl()->getAsFunction()->getNameAsString();
    if (IsAssigned) {
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
    }
    emplaceTransformation(new ReplaceStmt(CE, true, Name, ReplStr));
  } else if (FuncName == "cudaStreamGetFlags" ||
             FuncName == "cudaStreamGetPriority") {
    report(CE->getBeginLoc(), Diagnostics::STREAM_FLAG_PRIORITY_NOT_SUPPORTED);
    auto StmtStr1 = getStmtSpelling(CE->getArg(1), *Result.Context);
    std::string ReplStr{"*("};
    ReplStr += StmtStr1;
    ReplStr += ") = 0";
    if (IsAssigned) {
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
    }
    const std::string Name =
        CE->getCalleeDecl()->getAsFunction()->getNameAsString();
    emplaceTransformation(new ReplaceStmt(CE, true, Name, ReplStr));
  } else if (FuncName == "cudaDeviceGetStreamPriorityRange") {
    report(CE->getBeginLoc(), Diagnostics::STREAM_FLAG_PRIORITY_NOT_SUPPORTED);
    auto StmtStr0 = getStmtSpelling(CE->getArg(0), *Result.Context);
    auto StmtStr1 = getStmtSpelling(CE->getArg(1), *Result.Context);
    std::string ReplStr{"*("};
    ReplStr += StmtStr0;
    ReplStr += ") = 0, *(";
    ReplStr += StmtStr1;
    ReplStr += ") = 0";
    if (IsAssigned) {
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
    }
    const std::string Name =
        CE->getCalleeDecl()->getAsFunction()->getNameAsString();
    emplaceTransformation(new ReplaceStmt(CE, true, Name, ReplStr));
  } else if (FuncName == "cudaStreamAttachMemAsync" ||
             FuncName == "cudaStreamBeginCapture" ||
             FuncName == "cudaStreamEndCapture" ||
             FuncName == "cudaStreamIsCapturing" ||
             FuncName == "cudaStreamQuery" ||
             FuncName == "cudaStreamWaitEvent") {
    report(CE->getBeginLoc(), Diagnostics::NOTSUPPORTED, FuncName);
    emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ""));
  } else if (FuncName == "cudaStreamAddCallback") {
    auto StmtStr0 = getStmtSpelling(CE->getArg(0), *Result.Context);
    auto StmtStr1 = getStmtSpelling(CE->getArg(1), *Result.Context);
    auto StmtStr2 = getStmtSpelling(CE->getArg(2), *Result.Context);
    std::string ReplStr{StmtStr0};
    ReplStr += ".wait(), ";
    ReplStr += StmtStr1;
    ReplStr += "(";
    ReplStr += StmtStr0;
    ReplStr += ", 0, ";
    ReplStr += StmtStr2;
    ReplStr += ")";
    if (IsAssigned) {
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
    }
    report(CE->getBeginLoc(), Diagnostics::CALLBACK_FOR_QUEUE_NOT_SUPPORTED);
    emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
  } else {
    syclct_unreachable("Unknown function name");
  }
}

REGISTER_RULE(StreamAPICallRule)

// kernel call information collection
void KernelCallRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  MF.addMatcher(
      cudaKernelCallExpr(hasAncestor(functionDecl().bind("callContext")))
          .bind("kernelCall"),
      this);
}

void KernelCallRule::run(const ast_matchers::MatchFinder::MatchResult &Result) {
  auto FD = getAssistNodeAsType<FunctionDecl>(Result, "callContext");
  if (auto KCall =
          getAssistNodeAsType<CUDAKernelCallExpr>(Result, "kernelCall")) {
    emplaceTransformation(new ReplaceStmt(KCall, ""));
    if (!FD->isImplicitlyInstantiable())
      SyclctGlobalInfo::getInstance().insertKernelCallExpr(KCall);

    removeTrailingSemicolon(KCall, Result);
  }
}

// Find and remove the semicolon after the kernel call
void KernelCallRule::removeTrailingSemicolon(
    const CUDAKernelCallExpr *KCall,
    const ast_matchers::MatchFinder::MatchResult &Result) {
  const auto &SM = (*Result.Context).getSourceManager();
  auto KELoc = KCall->getEndLoc();
  auto Tok = Lexer::findNextToken(KELoc, SM, LangOptions()).getValue();
  assert(Tok.is(tok::TokenKind::semi));
  emplaceTransformation(new ReplaceToken(Tok.getLocation(), ""));
}

REGISTER_RULE(KernelCallRule)

// __device__ function call information collection
void DeviceFunctionCallRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  MF.addMatcher(
      callExpr(hasAncestor(functionDecl(anyOf(hasAttr(attr::CUDADevice),
                                              hasAttr(attr::CUDAGlobal)),
                                        unless(hasAttr(attr::CUDAHost)))
                               .bind("funcDecl")))
          .bind("callExpr"),
      this);
}

void DeviceFunctionCallRule::run(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  auto CE = getAssistNodeAsType<CallExpr>(Result, "callExpr");
  auto FD = getAssistNodeAsType<FunctionDecl>(Result, "funcDecl");
  if (CE && FD) {
    DeviceFunctionDecl::LinkRedecls(FD)->addCallee(CE);
  }
}

REGISTER_RULE(DeviceFunctionCallRule)

/// __constant__/__shared__/__device__ var information collection
void MemVarRule::registerMatcher(MatchFinder &MF) {
  auto DeclMatcher = varDecl(
      anyOf(hasAttr(attr::CUDAConstant), hasAttr(attr::CUDADevice),
            hasAttr(attr::CUDAShared)),
      unless(hasAnyName("threadIdx", "blockDim", "blockIdx", "gridDim")));
  MF.addMatcher(DeclMatcher.bind("var"), this);
  MF.addMatcher(
      declRefExpr(anyOf(hasParent(implicitCastExpr(
                                      unless(hasParent(arraySubscriptExpr())))
                                      .bind("impl")),
                        anything()),
                  to(DeclMatcher.bind("var")),
                  hasAncestor(functionDecl().bind("func")))
          .bind("used"),
      this);
}

void MemVarRule::insertExplicitCast(const ImplicitCastExpr *Impl,
                                    const QualType &Type) {
  if (Impl->getCastKind() == CastKind::CK_LValueToRValue) {
    if (!Type->isArrayType()) {
      auto TypeName = Type.getAsString();
      if (Type->isPointerType()) {
        TypeName = Type->getPointeeType().getAsString();
      }
      auto Itr = MapNames::TypeNamesMap.find(TypeName);
      if (Itr != MapNames::TypeNamesMap.end())
        TypeName = Itr->second;
      if (Type->isPointerType()) {
        TypeName += "*";
      }
      emplaceTransformation(new InsertBeforeStmt(Impl, "(" + TypeName + ")"));
    }
  }
}

void MemVarRule::run(const MatchFinder::MatchResult &Result) {
  if (auto MemVar = getNodeAsType<VarDecl>(Result, "var")) {
    emplaceTransformation(ReplaceVarDecl::getVarDeclReplacement(
        MemVar,
        MemVarInfo::buildMemVarInfo(MemVar)->getDeclarationReplacement()));
  }
  auto MemVarRef = getNodeAsType<DeclRefExpr>(Result, "used");
  auto Func = getAssistNodeAsType<FunctionDecl>(Result, "func");
  SyclctGlobalInfo &Global = SyclctGlobalInfo::getInstance();
  if (MemVarRef && Func) {
    if (Func->hasAttr<CUDAGlobalAttr>() ||
        (Func->hasAttr<CUDADeviceAttr>() && !Func->hasAttr<CUDAHostAttr>())) {
      auto VD = dyn_cast<VarDecl>(MemVarRef->getDecl());
      if (auto Var = Global.findMemVarInfo(VD))
        DeviceFunctionDecl::LinkRedecls(Func)->addVar(Var);
      if (auto Impl = getAssistNodeAsType<ImplicitCastExpr>(Result, "impl"))
        insertExplicitCast(Impl, VD->getType());
    }
  }
}

REGISTER_RULE(MemVarRule)

// Migration rule for GetVector, SetVector, GetMatrix, SetMatrix, etc.
void BLASGetSetRule::registerMatcher(MatchFinder &MF) {
  auto memoryAPI = [&]() {
    return hasAnyName("cublasSetVector", "cublasGetVector", "cublasSetMatrix",
                      "cublasGetMatrix");
  };
  MF.addMatcher(callExpr(allOf(callee(functionDecl(memoryAPI())), parentStmt))
                    .bind("call"),
                this);

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(memoryAPI())), unless(parentStmt)))
          .bind("callUsed"),
      this);
}

void BLASGetSetRule::run(const MatchFinder::MatchResult &Result) {
  auto TranslateCallExpr = [&](const CallExpr *C, const bool IsAssigned) {
    if (!C)
      return;
    if (C->getCalleeDecl() == nullptr)
      return;
    if (C->getCalleeDecl()->getAsFunction() == nullptr)
      return;
    const std::string Name =
        C->getCalleeDecl()->getAsFunction()->getNameAsString();
    assert(TranslationDispatcher.find(Name) != TranslationDispatcher.end());

    TranslationDispatcher.at(Name)(Result, C, IsAssigned);
  };

  TranslateCallExpr(getNodeAsType<CallExpr>(Result, "call"),
                    /* IsAssigned */ false);
  TranslateCallExpr(getNodeAsType<CallExpr>(Result, "callUsed"),
                    /* IsAssigned */ true);
}

BLASGetSetRule::BLASGetSetRule() {
  SetRuleProperty(ApplyToCudaFile | ApplyToCppFile);
  TranslationDispatcher["cublasGetVector"] = std::bind(
      &BLASGetSetRule::GetSetVectorTranslation, this, std::placeholders::_1,
      std::placeholders::_2, std::placeholders::_3);
  TranslationDispatcher["cublasSetVector"] = std::bind(
      &BLASGetSetRule::GetSetVectorTranslation, this, std::placeholders::_1,
      std::placeholders::_2, std::placeholders::_3);
  TranslationDispatcher["cublasGetMatrix"] = std::bind(
      &BLASGetSetRule::GetSetMatrixTranslation, this, std::placeholders::_1,
      std::placeholders::_2, std::placeholders::_3);
  TranslationDispatcher["cublasSetMatrix"] = std::bind(
      &BLASGetSetRule::GetSetMatrixTranslation, this, std::placeholders::_1,
      std::placeholders::_2, std::placeholders::_3);
}

void BLASGetSetRule::GetSetVectorTranslation(
    const MatchFinder::MatchResult &Result, const CallExpr *CE,
    const bool IsAssigned) {
  assert(CE && "Unknown result");
  const FunctionDecl *FD = CE->getDirectCallee();
  if (FD == nullptr)
    return;
  std::string FuncName = FD->getNameInfo().getName().getAsString();
  if (FuncName == "cublasSetVector" || FuncName == "cublasGetVector") {
    // The 4th and 6th param (incx and incy) of cublasSetVector/cublasGetVector
    // specify the space between two consequent elements when stored.
    // We migrate the original code when incx and incy both equal to 1 (all
    // elements are stored consequently).
    // Otherwise, the codes are kept originally.
    std::vector<std::string> ParamsStrVec =
        GetParamsAsStrs(CE, *(Result.Context));
    // CopySize equals to n*(elemSize+incx)-incx
    std::string CopySize = "(" + ParamsStrVec[0] + ")*((" + ParamsStrVec[1] +
                           ")+(" + ParamsStrVec[3] + "))-(" + ParamsStrVec[3] +
                           ")";
    std::string XStr = "(void*)(" + ParamsStrVec[2] + ")";
    std::string YStr = "(void*)(" + ParamsStrVec[4] + ")";
    const Expr *IncxExpr = CE->getArg(3);
    const Expr *IncyExpr = CE->getArg(5);
    Expr::EvalResult IncxExprResult, IncyExprResult;

    if (IncxExpr->EvaluateAsInt(IncxExprResult, *Result.Context) &&
        IncyExpr->EvaluateAsInt(IncyExprResult, *Result.Context)) {
      std::string IncxStr =
          IncxExprResult.Val.getAsString(*Result.Context, IncxExpr->getType());
      std::string IncyStr =
          IncyExprResult.Val.getAsString(*Result.Context, IncyExpr->getType());
      if (IncxStr != IncyStr) {
        // Keep original code, give a comment to let user migrate code manually
        report(CE->getBeginLoc(), Diagnostics::NOT_SUPPORTED_PARAMETERS_VALUE,
               FuncName, NOT_SUPPORTED_PARAMETERS_VALUE_CASE_0);
        return;
      }
      if ((IncxStr == IncyStr) && (IncxStr != "1")) {
        // incx equals to incy, but does not equal to 1. Performance issue may
        // occur.
        report(CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMACE_ISSUE,
               FuncName, POTENTIAL_PERFORMACE_ISSUE_CASE_0);
      }
    } else {
      // Keep original code, give a comment to let user migrate code manually
      report(CE->getBeginLoc(), Diagnostics::NOT_SUPPORTED_PARAMETERS_VALUE,
             FuncName, NOT_SUPPORTED_PARAMETERS_VALUE_CASE_1);
      return;
    }

    emplaceTransformation(
        new ReplaceCalleeName(CE, "syclct::sycl_memcpy", FuncName));
    std::string Replacement =
        "syclct::sycl_memcpy(" + YStr + "," + XStr + "," + CopySize + ",";
    if (FuncName == "cublasGetVector") {
      Replacement = Replacement + "syclct::device_to_host)";
      emplaceTransformation(new ReplaceStmt(CE, std::move(Replacement)));
    }
    if (FuncName == "cublasSetVector") {
      Replacement = Replacement + "syclct::host_to_device)";
      emplaceTransformation(new ReplaceStmt(CE, std::move(Replacement)));
    }

    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
      insertAroundStmt(CE, "(", ", 0)");
    }

  } else {
    syclct_unreachable("Unknown function name");
  }
}

void BLASGetSetRule::GetSetMatrixTranslation(
    const MatchFinder::MatchResult &Result, const CallExpr *CE,
    const bool IsAssigned) {
  assert(CE && "Unknown result");
  const FunctionDecl *FD = CE->getDirectCallee();
  if (FD == nullptr)
    return;
  std::string FuncName = FD->getNameInfo().getName().getAsString();
  if (FuncName == "cublasSetMatrix" || FuncName == "cublasGetMatrix") {
    std::vector<std::string> ParamsStrVec =
        GetParamsAsStrs(CE, *(Result.Context));
    // CopySize equals to lda*cols*elemSize
    std::string CopySize = "(" + ParamsStrVec[4] + ")*(" + ParamsStrVec[1] +
                           ")*(" + ParamsStrVec[2] + ")";
    std::string AStr = "(void*)(" + ParamsStrVec[3] + ")";
    std::string BStr = "(void*)(" + ParamsStrVec[5] + ")";

    const Expr *LdaExpr = CE->getArg(4);
    const Expr *LdbExpr = CE->getArg(6);
    Expr::EvalResult LdaExprResult, LdbExprResult;
    if (LdaExpr->EvaluateAsInt(LdaExprResult, *Result.Context) &&
        LdbExpr->EvaluateAsInt(LdbExprResult, *Result.Context)) {
      std::string LdaStr =
          LdaExprResult.Val.getAsString(*Result.Context, LdaExpr->getType());
      std::string LdbStr =
          LdbExprResult.Val.getAsString(*Result.Context, LdbExpr->getType());
      if (LdaStr != LdbStr) {
        // Keep original code, give a comment to let user migrate code manually
        report(CE->getBeginLoc(), Diagnostics::NOT_SUPPORTED_PARAMETERS_VALUE,
               FuncName, NOT_SUPPORTED_PARAMETERS_VALUE_CASE_2);
        return;
      }

      const Expr *RowsExpr = CE->getArg(0);
      Expr::EvalResult RowsExprResult;
      if (RowsExpr->EvaluateAsInt(RowsExprResult, *Result.Context)) {
        std::string RowsStr = RowsExprResult.Val.getAsString(
            *Result.Context, RowsExpr->getType());
        if (std::stoi(LdaStr) > std::stoi(RowsStr)) {
          // lda > rows. Performance issue may occur.
          report(CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMACE_ISSUE,
                 FuncName, POTENTIAL_PERFORMACE_ISSUE_CASE_1);
        }
      } else {
        // rows cannot be evaluated. Performance issue may occur.
        report(CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMACE_ISSUE,
               FuncName, POTENTIAL_PERFORMACE_ISSUE_CASE_2);
      }
    } else {
      // Keep original code, give a comment to let user migrate code manually
      report(CE->getBeginLoc(), Diagnostics::NOT_SUPPORTED_PARAMETERS_VALUE,
             FuncName, NOT_SUPPORTED_PARAMETERS_VALUE_CASE_3);
      return;
    }

    std::string Replacement =
        "syclct::sycl_memcpy(" + BStr + "," + AStr + "," + CopySize + ",";
    if (FuncName == "cublasGetMatrix") {
      Replacement = Replacement + "syclct::device_to_host)";
      emplaceTransformation(new ReplaceStmt(CE, std::move(Replacement)));
    }
    if (FuncName == "cublasSetMatrix") {
      Replacement = Replacement + "syclct::host_to_device)";
      emplaceTransformation(new ReplaceStmt(CE, std::move(Replacement)));
    }
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
      insertAroundStmt(CE, "(", ", 0)");
    }
  } else {
    syclct_unreachable("Unknown function name");
  }
}

std::vector<std::string>
BLASGetSetRule::GetParamsAsStrs(const CallExpr *CE, const ASTContext &Context) {
  std::vector<std::string> ParamsStrVec;
  for (auto Arg : CE->arguments())
    ParamsStrVec.emplace_back(getStmtSpelling(Arg, Context));
  return ParamsStrVec;
}

REGISTER_RULE(BLASGetSetRule)

void MemoryTranslationRule::MallocTranslation(
    const MatchFinder::MatchResult &Result, const CallExpr *C) {
  const std::string Name =
      C->getCalleeDecl()->getAsFunction()->getNameAsString();
  SyclctGlobalInfo::getInstance().insertCudaMalloc(C);
  emplaceTransformation(new ReplaceCalleeName(C, "syclct::sycl_malloc", Name));
}

void MemoryTranslationRule::MemcpyTranslation(
    const MatchFinder::MatchResult &Result, const CallExpr *C) {
  // Input:
  //   cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  //   cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
  //   cudaMemcpy(x_A, y_A, size, someDynamicCudaMemcpyKindValue);
  //
  // Desired output:
  //   sycl_memcpy<float>(d_A, h_A, numElements);
  //   sycl_memcpy_back<float>(h_A, d_A, numElements);
  //   sycl_memcpy<float>(x_A, y_A, numElements,
  //   someDynamicCudaMemcpyKindValue);
  //
  // Current output:
  //   sycl_memcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  //   sycl_memcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
  //   sycl_memcpy(x_A, y_A, size, someDynamicCudaMemcpyKindValue);

  // Migrate C->getArg(3) if this is enum constant.
  // TODO: this is a hack until we get pass ordering and make
  // different passes work with each other well together.
  const Expr *Direction = C->getArg(3);
  std::string DirectionName;
  const DeclRefExpr *DD = dyn_cast_or_null<DeclRefExpr>(Direction);
  if (DD && isa<EnumConstantDecl>(DD->getDecl())) {
    DirectionName = DD->getNameInfo().getName().getAsString();
    auto Search = EnumConstantRule::EnumNamesMap.find(DirectionName);
    assert(Search != EnumConstantRule::EnumNamesMap.end());
    Direction = nullptr;
    DirectionName = "syclct::" + Search->second;
  }
  const std::string Name =
      C->getCalleeDecl()->getAsFunction()->getNameAsString();
  emplaceTransformation(new ReplaceCalleeName(C, "syclct::sycl_memcpy", Name));
  insertAroundStmt(C->getArg(0), "(void*)(", ")");
  insertAroundStmt(C->getArg(1), "(void*)(", ")");
  emplaceTransformation(
      new ReplaceStmt(C->getArg(3), std::move(DirectionName)));
}

void MemoryTranslationRule::MemcpyToSymbolTranslation(
    const MatchFinder::MatchResult &Result, const CallExpr *C) {
  // Input:
  //   cudaMemcpyToSymbol(d_A, h_A, size, offset, cudaMemcpyHostToDevice);
  //   cudaMemcpyToSymbol(d_B, d_C, size, offset, cudaMemcpyDeviceToDevice);
  //   cudaMemcpyToSymbol(h_A, d_B, size, offset, cudaMemcpyDefault);

  // Desired output:
  //   syclct::sycl_memcpy_to_symbol(d_A.get_ptr(), (void*)(h_A), size,
  //                                 offset, syclct::host_to_device);
  //
  //   syclct::sycl_memcpy_to_symbol(d_B.get_ptr(), d_C, size, offset,
  //                                 syclct::device_to_device);
  //
  //   syclct::sycl_memcpy_to_symbol(h_A.get_ptr(), (void*)(d_B), size,
  //                                 offset, syclct::automatic);
  const Expr *Direction = C->getArg(4);
  std::string DirectionName;
  const DeclRefExpr *DD = dyn_cast_or_null<DeclRefExpr>(Direction);
  if (DD && isa<EnumConstantDecl>(DD->getDecl())) {
    DirectionName = DD->getNameInfo().getName().getAsString();
    auto Search = EnumConstantRule::EnumNamesMap.find(DirectionName);
    assert(Search != EnumConstantRule::EnumNamesMap.end());
    Direction = nullptr;
    DirectionName = "syclct::" + Search->second;
  }

  SyclctGlobalInfo &Global = SyclctGlobalInfo::getInstance();
  auto MallocInfo = Global.findCudaMalloc(C->getArg(1));
  auto VD = CudaMallocInfo::getDecl(C->getArg(0));
  if (MallocInfo && VD) {
    if (auto Var = Global.findMemVarInfo(VD)) {
      emplaceTransformation(new ReplaceStmt(
          C, Var->getName() + ".assign(" +
                 MallocInfo->getAssignArgs(Var->getType()->getBaseName()) +
                 ")"));
      return;
    }
  }

  std::string VarName = getStmtSpelling(C->getArg(0), *Result.Context);
  // Migrate variable name such as "&const_angle[0]", "&const_one"
  // into "const_angle.get_ptr()", "const_one.get_ptr()".
  VarName.erase(std::remove(VarName.begin(), VarName.end(), '&'),
                VarName.end());
  std::size_t pos = VarName.find("[");
  VarName = (pos != std::string::npos) ? VarName.substr(0, pos) : VarName;
  VarName += ".get_ptr()";

  const std::string Name =
      C->getCalleeDecl()->getAsFunction()->getNameAsString();
  emplaceTransformation(
      new ReplaceCalleeName(C, "syclct::sycl_memcpy_to_symbol", Name));
  emplaceTransformation(new ReplaceToken(C->getArg(0)->getBeginLoc(),
                                         C->getArg(0)->getEndLoc(),
                                         std::move(VarName)));
  insertAroundStmt(C->getArg(1), "(void*)(", ")");
  emplaceTransformation(
      new ReplaceStmt(C->getArg(4), std::move(DirectionName)));
}

void MemoryTranslationRule::MemcpyFromSymbolTranslation(
    const MatchFinder::MatchResult &Result, const CallExpr *C) {
  // Input:
  //   cudaMemcpyToSymbol(h_A, d_A, size, offset, cudaMemcpyDeviceToHost);
  //   cudaMemcpyToSymbol(d_B, d_A, size, offset, cudaMemcpyDeviceToDevice);

  // Desired output:
  //   syclct::sycl_memcpy_to_symbol((void*)(h_A), d_A.get_ptr(), size,
  //   offset,
  //                                 syclct::device_to_host);
  //
  //   syclct::sycl_memcpy_to_symbol((void*)(d_B), d_A.get_ptr(), size,
  //   offset,
  //                                 syclct::device_to_device);
  const Expr *Direction = C->getArg(4);
  std::string DirectionName;
  const DeclRefExpr *DD = dyn_cast_or_null<DeclRefExpr>(Direction);
  if (DD && isa<EnumConstantDecl>(DD->getDecl())) {
    DirectionName = DD->getNameInfo().getName().getAsString();
    auto Search = EnumConstantRule::EnumNamesMap.find(DirectionName);
    assert(Search != EnumConstantRule::EnumNamesMap.end());
    Direction = nullptr;
    DirectionName = "syclct::" + Search->second;
  }

  std::string VarName = getStmtSpelling(C->getArg(1), *Result.Context);
  // Migrate variable name such as "&const_angle[0]", "&const_one"
  // into "const_angle.get_ptr()", "const_one.get_ptr()".
  VarName.erase(std::remove(VarName.begin(), VarName.end(), '&'),
                VarName.end());
  std::size_t pos = VarName.find("[");
  VarName = (pos != std::string::npos) ? VarName.substr(0, pos) : VarName;
  VarName += ".get_ptr()";

  insertAroundStmt(C->getArg(0), "(void*)(", ")");

  const std::string Name =
      C->getCalleeDecl()->getAsFunction()->getNameAsString();
  emplaceTransformation(
      new ReplaceCalleeName(C, "syclct::sycl_memcpy_from_symbol", Name));
  emplaceTransformation(new ReplaceToken(C->getArg(1)->getBeginLoc(),
                                         C->getArg(1)->getEndLoc(),
                                         std::move(VarName)));
  emplaceTransformation(
      new ReplaceStmt(C->getArg(4), std::move(DirectionName)));
}

void MemoryTranslationRule::FreeTranslation(
    const MatchFinder::MatchResult &Result, const CallExpr *C) {
  const std::string Name =
      C->getCalleeDecl()->getAsFunction()->getNameAsString();
  emplaceTransformation(new ReplaceCalleeName(C, "syclct::sycl_free", Name));
}

void MemoryTranslationRule::MemsetTranslation(
    const MatchFinder::MatchResult &Result, const CallExpr *C) {
  const std::string Name =
      C->getCalleeDecl()->getAsFunction()->getNameAsString();
  emplaceTransformation(new ReplaceCalleeName(C, "syclct::sycl_memset", Name));
  insertAroundStmt(C->getArg(0), "(void*)(", ")");
  insertAroundStmt(C->getArg(1), "(int)(", ")");
  insertAroundStmt(C->getArg(2), "(size_t)(", ")");
}

// Memory migration rules live here.
void MemoryTranslationRule::registerMatcher(MatchFinder &MF) {
  auto memoryAPI = [&]() {
    return hasAnyName("cudaMalloc", "cudaMemcpy", "cudaMemcpyToSymbol",
                      "cudaMemcpyFromSymbol", "cudaFree", "cudaMemset");
  };

  MF.addMatcher(callExpr(allOf(callee(functionDecl(memoryAPI())), parentStmt))
                    .bind("call"),
                this);

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(memoryAPI())), unless(parentStmt)))
          .bind("callUsed"),
      this);
}

void MemoryTranslationRule::run(const MatchFinder::MatchResult &Result) {
  auto TranslateCallExpr = [&](const CallExpr *C, const bool IsAssigned) {
    if (!C)
      return;

    if (IsAssigned) {
      report(C->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
      insertAroundStmt(C, "(", ", 0)");
    }

    const std::string Name =
        C->getCalleeDecl()->getAsFunction()->getNameAsString();
    assert(TranslationDispatcher.find(Name) != TranslationDispatcher.end());
    TranslationDispatcher.at(Name)(Result, C);
  };

  TranslateCallExpr(getNodeAsType<CallExpr>(Result, "call"),
                    /* IsAssigned */ false);
  TranslateCallExpr(getNodeAsType<CallExpr>(Result, "callUsed"),
                    /* IsAssigned */ true);
}

MemoryTranslationRule::MemoryTranslationRule() {
  SetRuleProperty(ApplyToCudaFile | ApplyToCppFile);
  TranslationDispatcher["cudaMalloc"] =
      std::bind(&MemoryTranslationRule::MallocTranslation, this,
                std::placeholders::_1, std::placeholders::_2);
  TranslationDispatcher["cudaMemcpy"] =
      std::bind(&MemoryTranslationRule::MemcpyTranslation, this,
                std::placeholders::_1, std::placeholders::_2);
  TranslationDispatcher["cudaMemcpyToSymbol"] =
      std::bind(&MemoryTranslationRule::MemcpyToSymbolTranslation, this,
                std::placeholders::_1, std::placeholders::_2);
  TranslationDispatcher["cudaMemcpyFromSymbol"] =
      std::bind(&MemoryTranslationRule::MemcpyFromSymbolTranslation, this,
                std::placeholders::_1, std::placeholders::_2);
  TranslationDispatcher["cudaFree"] =
      std::bind(&MemoryTranslationRule::FreeTranslation, this,
                std::placeholders::_1, std::placeholders::_2);
  TranslationDispatcher["cudaMemset"] =
      std::bind(&MemoryTranslationRule::MemsetTranslation, this,
                std::placeholders::_1, std::placeholders::_2);
}

REGISTER_RULE(MemoryTranslationRule)

static const CXXConstructorDecl *getIfConstructorDecl(const Decl *ND) {
  if (const auto *Tmpl = dyn_cast<FunctionTemplateDecl>(ND))
    ND = Tmpl->getTemplatedDecl();
  return dyn_cast<CXXConstructorDecl>(ND);
}

void ErrorTryCatchRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  MF.addMatcher(functionDecl(hasBody(compoundStmt()),
                             unless(anyOf(hasAttr(attr::CUDAGlobal),
                                          hasAttr(attr::CUDADevice),
                                          hasAncestor(lambdaExpr(anything())))))
                    .bind("functionDecl"),
                this);
}

void ErrorTryCatchRule::run(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  const FunctionDecl *FD = getNodeAsType<FunctionDecl>(Result, "functionDecl");
  if (!FD)
    return;

  // Filter out compiler generated methods
  if (const CXXMethodDecl *CXXMDecl = dyn_cast<CXXMethodDecl>(FD)) {
    if (!CXXMDecl->isUserProvided()) {
      return;
    }
  }

  auto BodySLoc = FD->getBody()->getSourceRange().getBegin().getRawEncoding();
  if (Insertions.find(BodySLoc) != Insertions.end())
    return;

  Insertions.insert(BodySLoc);

  // First check if this is a constructor decl
  if (const CXXConstructorDecl *CDecl = getIfConstructorDecl(FD)) {
    emplaceTransformation(new InsertBeforeCtrInitList(CDecl, "try "));
  } else {
    emplaceTransformation(new InsertBeforeStmt(FD->getBody(), "try "));
  }

  std::string ReplaceStr =
      getNL() + std::string("catch (cl::sycl::exception const &exc) {") +
      getNL() +
      std::string("  std::cerr << exc.what() << \"EOE at line \" << ") +
      std::string("__LINE__ << std::endl;") + getNL() +
      std::string("  std::exit(1);") + getNL() + "}";

  emplaceTransformation(
      new InsertAfterStmt(FD->getBody(), std::move(ReplaceStr)));
}

REGISTER_RULE(ErrorTryCatchRule)

void UnnamedTypesRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      cxxRecordDecl(unless(has(cxxRecordDecl(isImplicit()))), hasDefinition())
          .bind("unnamedType"),
      this);
}

void UnnamedTypesRule::run(const MatchFinder::MatchResult &Result) {
  auto D = getNodeAsType<CXXRecordDecl>(Result, "unnamedType");
  if (D && D->getName().empty())
    emplaceTransformation(new InsertClassName(D));
}

REGISTER_RULE(UnnamedTypesRule)

void MathFunctionsRule::registerMatcher(MatchFinder &MF) {
  std::vector<std::string> ExceptionalFunctionNames;
  for (auto Function : ExceptionalFunctionNamesMap)
    ExceptionalFunctionNames.push_back(Function.first);

  std::vector<std::string> HalfFunctionNames;
  for (auto Function : HalfFunctionNamesMap)
    HalfFunctionNames.push_back(Function.first);

  std::vector<std::string> SingleDoubleFunctionNames;
  for (auto Function : SingleDoubleFunctionNamesMap)
    SingleDoubleFunctionNames.push_back(Function.first);

  std::vector<std::string> IntegerFunctionNames;
  for (auto Function : IntegerFunctionNamesMap)
    IntegerFunctionNames.push_back(Function.first);

  std::vector<std::string> TypecastFunctionNames;
  for (auto Function : TypecastFunctionNamesMap)
    TypecastFunctionNames.push_back(Function.first);

  MF.addMatcher(
      callExpr(callee(functionDecl(
                   internal::Matcher<NamedDecl>(
                       new internal::HasNameMatcher(ExceptionalFunctionNames)),
                   unless(hasDeclContext(namespaceDecl(anything()))))))
          .bind("mathExceptional"),
      this);
  MF.addMatcher(
      callExpr(callee(functionDecl(
                   internal::Matcher<NamedDecl>(
                       new internal::HasNameMatcher(HalfFunctionNames)),
                   unless(hasDeclContext(namespaceDecl(anything()))))))
          .bind("mathHalf"),
      this);
  MF.addMatcher(
      callExpr(callee(functionDecl(
                   internal::Matcher<NamedDecl>(
                       new internal::HasNameMatcher(SingleDoubleFunctionNames)),
                   unless(hasDeclContext(namespaceDecl(anything()))))))
          .bind("mathSingleDouble"),
      this);
  MF.addMatcher(
      callExpr(callee(functionDecl(
                   internal::Matcher<NamedDecl>(
                       new internal::HasNameMatcher(IntegerFunctionNames)),
                   unless(hasDeclContext(namespaceDecl(anything()))))))
          .bind("mathInteger"),
      this);
  MF.addMatcher(
      callExpr(callee(functionDecl(
                   internal::Matcher<NamedDecl>(
                       new internal::HasNameMatcher(TypecastFunctionNames)),
                   unless(hasDeclContext(namespaceDecl(anything()))))))
          .bind("mathTypecast"),
      this);
}

bool endsWith(std::string const &Str, std::string const &Ending) {
  if (Ending.size() > Str.size())
    return false;
  return std::equal(Ending.rbegin(), Ending.rend(), Str.rbegin());
}

void MathFunctionsRule::run(const MatchFinder::MatchResult &Result) {
  const CallExpr *CE = nullptr;
  if ((CE = getNodeAsType<CallExpr>(Result, "mathExceptional")))
    return handleExceptionalFunctions(CE, Result);

  if ((CE = getNodeAsType<CallExpr>(Result, "mathHalf")))
    return handleHalfFunctions(CE, Result);

  if ((CE = getNodeAsType<CallExpr>(Result, "mathSingleDouble")))
    return handleSingleDoubleFunctions(CE, Result);

  if ((CE = getNodeAsType<CallExpr>(Result, "mathInteger")))
    return handleIntegerFunctions(CE, Result);

  if ((CE = getNodeAsType<CallExpr>(Result, "mathTypecast")))
    return handleTypecastFunctions(CE, Result);
}

void MathFunctionsRule::handleExceptionalFunctions(
    const CallExpr *CE, const MatchFinder::MatchResult &Result) {
  auto FD = CE->getDirectCallee();
  if (!FD)
    return;

  const std::string FuncName = FD->getNameAsString();
  std::string NamespaceStr;
  auto Qualifier =
      dyn_cast<DeclRefExpr>(CE->getCallee()->IgnoreImpCasts())->getQualifier();
  if (Qualifier) {
    auto Namespace = Qualifier->getAsNamespace();
    if (Namespace)
      NamespaceStr = Namespace->getName();
  }

  if (NamespaceStr == "std")
    return;

  // The calls to some functions do not need to be migrated because they
  // are functions in C standard library. However, in UB18.04 these functions
  // are not recognized by ComputeCpp. As a result, these calls are temporarily
  // migrated to sycl alternatives.
  // TODO: Find a better way to deal with it or investigate why
  // ComputeCpp behaves differently in UB16.04 and UB18.04.
  if (ExceptionalFunctionNamesMap.find(FuncName) !=
      ExceptionalFunctionNamesMap.end()) {
    std::string NewFuncName = ExceptionalFunctionNamesMap.at(FuncName);
    if (FuncName == "abs") {
      // further check the type of the args.
      if (!CE->getArg(0)->getType()->isIntegerType()) {
        NewFuncName = "cl::sycl::fabs";
      }
    }

    emplaceTransformation(
        new ReplaceCalleeName(CE, std::move(NewFuncName), FuncName));

    if (FuncName == "min") {
      const LangOptions &LO = Result.Context->getLangOpts();
      std::string FT = CE->getType().getAsString(PrintingPolicy(LO));
      for (unsigned i = 0; i < CE->getNumArgs(); i++) {
        std::string ArgT =
            CE->getArg(i)->getType().getAsString(PrintingPolicy(LO));
        std::string ArgExpr = CE->getArg(i)->getStmtClassName();
        if (ArgT != FT || ArgExpr == "BinaryOperator") {
          insertAroundStmt(CE->getArg(i), "(" + FT + ")(", ")");
        }
      }
    }
  }
}

void MathFunctionsRule::handleHalfFunctions(
    const CallExpr *CE, const MatchFinder::MatchResult &Result) {
  auto FD = CE->getDirectCallee();
  if (!FD || !FD->hasAttr<CUDADeviceAttr>())
    return;

  const std::string FuncName = FD->getNameAsString();

  if (HalfFunctionNamesMap.find(FuncName) != HalfFunctionNamesMap.end()) {
    std::string NewFuncName = HalfFunctionNamesMap.at(FuncName);
    if (NewFuncName == StringLiteralUnsupported) {
      report(CE->getBeginLoc(), Diagnostics::NOTSUPPORTED, FuncName);
      return;
    }
    if (FuncName == "__h2div" || FuncName == "__hdiv" || FuncName == "__hmul" ||
        FuncName == "__hsub" || FuncName == "__hmul2" ||
        FuncName == "__hsub2" || FuncName == "__heq" || FuncName == "__hge" ||
        FuncName == "__hgt" || FuncName == "__hle" || FuncName == "__hlt" ||
        FuncName == "__hne" || FuncName == "__heq2" || FuncName == "__hge2" ||
        FuncName == "__hgt2" || FuncName == "__hle2" || FuncName == "__hlt2" ||
        FuncName == "__hne2") {
      std::string Operator = HalfFunctionNamesMap.at(FuncName);
      auto StmtStrArg0 = getStmtSpelling(CE->getArg(0), *Result.Context);
      auto StmtStrArg1 = getStmtSpelling(CE->getArg(1), *Result.Context);
      std::string ReplStr{StmtStrArg0};
      ReplStr += " ";
      ReplStr += Operator;
      ReplStr += " ";
      ReplStr += StmtStrArg1;
      report(CE->getBeginLoc(), Diagnostics::ROUNDING_MODE_UNSUPPORTED,
             "operator" + Operator);
      emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
      return;
    } else if (FuncName == "__hneg" || FuncName == "__hneg2") {
      std::string Operator = HalfFunctionNamesMap.at(FuncName);
      auto StmtStrArg0 = getStmtSpelling(CE->getArg(0), *Result.Context);
      std::string ReplStr{Operator};
      ReplStr += StmtStrArg0;
      report(CE->getBeginLoc(), Diagnostics::ROUNDING_MODE_UNSUPPORTED,
             "operator" + Operator);
      emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
      return;
    }

    emplaceTransformation(
        new ReplaceCalleeName(CE, std::move(NewFuncName), FuncName));
  }
}

void MathFunctionsRule::handleSingleDoubleFunctions(
    const CallExpr *CE, const MatchFinder::MatchResult &Result) {
  auto FD = CE->getDirectCallee();
  if (!FD || !FD->hasAttr<CUDADeviceAttr>())
    return;

  const std::string FuncName = FD->getNameAsString();

  if (SingleDoubleFunctionNamesMap.find(FuncName) !=
      SingleDoubleFunctionNamesMap.end()) {
    std::string NewFuncName = SingleDoubleFunctionNamesMap.at(FuncName);
    if (NewFuncName == StringLiteralUnsupported) {
      report(CE->getBeginLoc(), Diagnostics::NOTSUPPORTED, FuncName);
    } else if (FuncName == "__dadd_rd" || FuncName == "__fadd_rd" ||
               FuncName == "__dadd_rn" || FuncName == "__fadd_rn" ||
               FuncName == "__dadd_ru" || FuncName == "__fadd_ru" ||
               FuncName == "__dadd_rz" || FuncName == "__fadd_rz" ||
               FuncName == "__dsub_rd" || FuncName == "__fsub_rd" ||
               FuncName == "__dsub_rn" || FuncName == "__fsub_rn" ||
               FuncName == "__dsub_ru" || FuncName == "__fsub_ru" ||
               FuncName == "__dsub_rz" || FuncName == "__fsub_rz" ||
               FuncName == "__dmul_rd" || FuncName == "__fmul_rd" ||
               FuncName == "__dmul_rn" || FuncName == "__fmul_rn" ||
               FuncName == "__dmul_ru" || FuncName == "__fmul_ru" ||
               FuncName == "__dmul_rz" || FuncName == "__fmul_rz" ||
               FuncName == "__ddiv_rd" || FuncName == "__fdiv_rd" ||
               FuncName == "__ddiv_rn" || FuncName == "__fdiv_rn" ||
               FuncName == "__ddiv_ru" || FuncName == "__fdiv_ru" ||
               FuncName == "__ddiv_rz" || FuncName == "__fdiv_rz") {
      std::string Operator = SingleDoubleFunctionNamesMap.at(FuncName);
      auto StmtStrArg0 = getStmtSpelling(CE->getArg(0), *Result.Context);
      auto StmtStrArg1 = getStmtSpelling(CE->getArg(1), *Result.Context);
      std::string ReplStr{StmtStrArg0};
      ReplStr += " ";
      ReplStr += Operator;
      ReplStr += " ";
      ReplStr += StmtStrArg1;
      report(CE->getBeginLoc(), Diagnostics::ROUNDING_MODE_UNSUPPORTED,
             "operator+");
      emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
    } else if (FuncName == "frexp" || FuncName == "frexpf") {
      auto StmtStrArg0 = getStmtSpelling(CE->getArg(0), *Result.Context);
      auto StmtStrArg1 = getStmtSpelling(CE->getArg(1), *Result.Context);
      std::string ReplStr{"cl::sycl::frexp("};
      ReplStr += StmtStrArg0;
      ReplStr += ", cl::sycl::make_ptr<int, "
                 "cl::sycl::access::address_space::local_space>(";
      ReplStr += StmtStrArg1;
      ReplStr += "))";
      emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
    } else if (FuncName == "modf" || FuncName == "modff") {
      auto StmtStrArg0 = getStmtSpelling(CE->getArg(0), *Result.Context);
      auto StmtStrArg1 = getStmtSpelling(CE->getArg(1), *Result.Context);
      std::string ReplStr{"cl::sycl::modf("};
      ReplStr += StmtStrArg0;
      if (FuncName == "modf")
        ReplStr += ", cl::sycl::make_ptr<double, "
                   "cl::sycl::access::address_space::local_space>(";
      else
        ReplStr += ", cl::sycl::make_ptr<float, "
                   "cl::sycl::access::address_space::local_space>(";
      ReplStr += StmtStrArg1;
      ReplStr += "))";
      emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
    } else if (FuncName == "nan" || FuncName == "nanf") {
      emplaceTransformation(
          new ReplaceStmt(CE, true, FuncName, "cl::sycl::nan(0u)"));
    } else if (FuncName == "sincos" || FuncName == "sincosf" ||
               FuncName == "__sincosf") {
      auto StmtStrArg0 = getStmtSpelling(CE->getArg(0), *Result.Context);
      auto StmtStrArg1 = getStmtSpelling(CE->getArg(1), *Result.Context);
      auto StmtStrArg2 = getStmtSpelling(CE->getArg(2), *Result.Context);
      std::string ReplStr;
      if (StmtStrArg1[0] == '&')
        ReplStr = StmtStrArg1.substr(1);
      else
        ReplStr = "*(" + StmtStrArg1 + ")";
      ReplStr += " = cl::sycl::sincos(";
      ReplStr += StmtStrArg0;
      if (FuncName == "sincos")
        ReplStr += ", cl::sycl::make_ptr<double, "
                   "cl::sycl::access::address_space::local_space>(";
      else
        ReplStr += ", cl::sycl::make_ptr<float, "
                   "cl::sycl::access::address_space::local_space>(";
      ReplStr += StmtStrArg2;
      ReplStr += "))";
      emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
    } else if (FuncName == "sincospi" || FuncName == "sincospif") {
      auto StmtStrArg0 = getStmtSpelling(CE->getArg(0), *Result.Context);
      auto StmtStrArg1 = getStmtSpelling(CE->getArg(1), *Result.Context);
      auto StmtStrArg2 = getStmtSpelling(CE->getArg(2), *Result.Context);
      std::string ReplStr;
      if (StmtStrArg1[0] == '&')
        ReplStr = StmtStrArg1.substr(1);
      else
        ReplStr = "*(" + StmtStrArg1 + ")";
      ReplStr += " = cl::sycl::sincos(";
      ReplStr += StmtStrArg0;
      if (FuncName == "sincospi")
        ReplStr += " * SYCLCT_PI";
      else
        ReplStr += " * SYCLCT_PI_F";

      if (FuncName == "sincospi")
        ReplStr += ", cl::sycl::make_ptr<double, "
                   "cl::sycl::access::address_space::local_space>(";
      else
        ReplStr += ", cl::sycl::make_ptr<float, "
                   "cl::sycl::access::address_space::local_space>(";
      ReplStr += StmtStrArg2;
      ReplStr += "))";
      report(CE->getBeginLoc(), Diagnostics::MATH_SIMULATION, FuncName,
             "cl::sycl::sincos");
      emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
    } else if (FuncName == "remquo" || FuncName == "remquof") {
      auto StmtStrArg0 = getStmtSpelling(CE->getArg(0), *Result.Context);
      auto StmtStrArg1 = getStmtSpelling(CE->getArg(1), *Result.Context);
      auto StmtStrArg2 = getStmtSpelling(CE->getArg(2), *Result.Context);
      std::string ReplStr{"cl::sycl::remquo("};
      ReplStr += StmtStrArg0;
      ReplStr += ", ";
      ReplStr += StmtStrArg1;
      ReplStr += ", cl::sycl::make_ptr<int, "
                 "cl::sycl::access::address_space::local_space>(";
      ReplStr += StmtStrArg2;
      ReplStr += "))";
      emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
    } else if (FuncName == "nearbyint" || FuncName == "nearbyintf") {
      auto StmtStrArg0 = getStmtSpelling(CE->getArg(0), *Result.Context);
      std::string ReplStr{"cl::sycl::floor("};
      ReplStr += StmtStrArg0;
      ReplStr += " + 0.5)";
      report(CE->getBeginLoc(), Diagnostics::MATH_SIMULATION, FuncName,
             "cl::sycl::floor");
      emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
    } else if (FuncName == "rhypot" || FuncName == "rhypotf") {
      auto StmtStrArg0 = getStmtSpelling(CE->getArg(0), *Result.Context);
      auto StmtStrArg1 = getStmtSpelling(CE->getArg(1), *Result.Context);
      std::string ReplStr{"1 / cl::sycl::hypot("};
      ReplStr += StmtStrArg0;
      ReplStr += ", ";
      ReplStr += StmtStrArg1;
      ReplStr += ")";
      report(CE->getBeginLoc(), Diagnostics::MATH_SIMULATION, FuncName,
             "cl::sycl::hypot");
      emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
    } else {
      // For the rest math functions, replace callee names with DPC++ ones
      if (endsWith(FuncName, "_rd") || endsWith(FuncName, "_rn") ||
          endsWith(FuncName, "_ru") || endsWith(FuncName, "_rz")) {
        report(CE->getBeginLoc(), Diagnostics::ROUNDING_MODE_UNSUPPORTED,
               NewFuncName);
      }
      emplaceTransformation(
          new ReplaceCalleeName(CE, std::move(NewFuncName), FuncName));
    }
  }
}

void MathFunctionsRule::handleIntegerFunctions(
    const CallExpr *CE, const MatchFinder::MatchResult &Result) {
  auto FD = CE->getDirectCallee();
  if (!FD || !FD->hasAttr<CUDADeviceAttr>())
    return;

  const std::string FuncName = FD->getNameAsString();

  if (IntegerFunctionNamesMap.find(FuncName) != IntegerFunctionNamesMap.end()) {
    std::string NewFuncName = IntegerFunctionNamesMap.at(FuncName);
    if (NewFuncName == StringLiteralUnsupported) {
      report(CE->getBeginLoc(), Diagnostics::NOTSUPPORTED, FuncName);
      return;
    }

    emplaceTransformation(
        new ReplaceCalleeName(CE, std::move(NewFuncName), FuncName));
  }
}

void MathFunctionsRule::handleTypecastFunctions(
    const CallExpr *CE, const MatchFinder::MatchResult &Result) {
  auto FD = CE->getDirectCallee();
  if (!FD || !FD->hasAttr<CUDADeviceAttr>())
    return;

  using SSMap = std::map<std::string, std::string>;
  static SSMap RoundingModeMap{{"", "automatic"},
                               {"rd", "rtn"},
                               {"rn", "rte"},
                               {"ru", "rtp"},
                               {"rz", "rtz"}};
  const std::string FuncName = CE->getDirectCallee()->getNameAsString();
  if (TypecastFunctionNamesMap.find(FuncName) !=
      TypecastFunctionNamesMap.end()) {
    std::string NewFuncName = TypecastFunctionNamesMap.at(FuncName);
    if (NewFuncName == StringLiteralUnsupported) {
      report(CE->getBeginLoc(), Diagnostics::NOTSUPPORTED, FuncName);
      return;
    } else if (startsWith(NewFuncName, "syclct::")) {
      emplaceTransformation(
          new ReplaceCalleeName(CE, std::move(NewFuncName), FuncName));
      return;
    }
    assert(NewFuncName.empty());

    if (FuncName == "__float22half2_rn") {
      auto StmtStrArg0 = getStmtSpelling(CE->getArg(0), *Result.Context);
      std::string ReplStr{StmtStrArg0};
      ReplStr += ".convert<cl::sycl::half, cl::sycl::rounding_mode::rte>()";
      emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
    } else if (FuncName == "__float2half2_rn") {
      auto StmtStrArg0 = getStmtSpelling(CE->getArg(0), *Result.Context);
      std::string ReplStr{"cl::sycl::float2{"};
      ReplStr += StmtStrArg0;
      ReplStr += ",";
      ReplStr += StmtStrArg0;
      ReplStr += "}.convert<cl::sycl::half, cl::sycl::rounding_mode::rte>()";
      emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
    } else if (FuncName == "__floats2half2_rn") {
      auto StmtStrArg0 = getStmtSpelling(CE->getArg(0), *Result.Context);
      auto StmtStrArg1 = getStmtSpelling(CE->getArg(1), *Result.Context);
      std::string ReplStr{"cl::sycl::float2{"};
      ReplStr += StmtStrArg0;
      ReplStr += ",";
      ReplStr += StmtStrArg1;
      ReplStr += "}.convert<cl::sycl::half, cl::sycl::rounding_mode::rte>()";
      emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
    } else if (FuncName == "__half22float2") {
      auto StmtStrArg0 = getStmtSpelling(CE->getArg(0), *Result.Context);
      std::string ReplStr{StmtStrArg0};
      ReplStr += ".convert<float, cl::sycl::rounding_mode::automatic>()";
      emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
    } else if (FuncName == "__half2half2") {
      auto StmtStrArg0 = getStmtSpelling(CE->getArg(0), *Result.Context);
      std::string ReplStr{"cl::sycl::half2{"};
      ReplStr += StmtStrArg0;
      ReplStr += ",";
      ReplStr += StmtStrArg0;
      ReplStr += "}";
      emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
    } else if (FuncName == "__halves2half2") {
      auto StmtStrArg0 = getStmtSpelling(CE->getArg(0), *Result.Context);
      auto StmtStrArg1 = getStmtSpelling(CE->getArg(1), *Result.Context);
      std::string ReplStr{"cl::sycl::half2{"};
      ReplStr += StmtStrArg0;
      ReplStr += ",";
      ReplStr += StmtStrArg1;
      ReplStr += "}";
      emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
    } else if (FuncName == "__high2float") {
      auto StmtStrArg0 = getStmtSpelling(CE->getArg(0), *Result.Context);
      std::string ReplStr{StmtStrArg0};
      ReplStr += ".get_value(0)";
      emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
    } else if (FuncName == "__high2half") {
      auto StmtStrArg0 = getStmtSpelling(CE->getArg(0), *Result.Context);
      std::string ReplStr{StmtStrArg0};
      ReplStr += ".get_value(0)";
      emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
    } else if (FuncName == "__high2half2") {
      auto StmtStrArg0 = getStmtSpelling(CE->getArg(0), *Result.Context);
      std::string ReplStr{"cl::sycl::half2{"};
      ReplStr += StmtStrArg0;
      ReplStr += ".get_value(0), ";
      ReplStr += StmtStrArg0;
      ReplStr += ".get_value(0)}";
      emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
    } else if (FuncName == "__highs2half2") {
      auto StmtStrArg0 = getStmtSpelling(CE->getArg(0), *Result.Context);
      auto StmtStrArg1 = getStmtSpelling(CE->getArg(1), *Result.Context);
      std::string ReplStr{"cl::sycl::half2{"};
      ReplStr += StmtStrArg0;
      ReplStr += ".get_value(0), ";
      ReplStr += StmtStrArg1;
      ReplStr += ".get_value(0)}";
      emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
    } else if (FuncName == "__low2float") {
      auto StmtStrArg0 = getStmtSpelling(CE->getArg(0), *Result.Context);
      std::string ReplStr{StmtStrArg0};
      ReplStr += ".get_value(1)";
      emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
    } else if (FuncName == "__low2half") {
      auto StmtStrArg0 = getStmtSpelling(CE->getArg(0), *Result.Context);
      std::string ReplStr{StmtStrArg0};
      ReplStr += ".get_value(1)";
      emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
    } else if (FuncName == "__low2half2") {
      auto StmtStrArg0 = getStmtSpelling(CE->getArg(0), *Result.Context);
      std::string ReplStr{"cl::sycl::half2{"};
      ReplStr += StmtStrArg0;
      ReplStr += ".get_value(1), ";
      ReplStr += StmtStrArg0;
      ReplStr += ".get_value(1)}";
      emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
    } else if (FuncName == "__lowhigh2highlow") {
      auto StmtStrArg0 = getStmtSpelling(CE->getArg(0), *Result.Context);
      std::string ReplStr{"cl::sycl::half2{"};
      ReplStr += StmtStrArg0;
      ReplStr += ".get_value(1), ";
      ReplStr += StmtStrArg0;
      ReplStr += ".get_value(0)}";
      emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
    } else if (FuncName == "__lows2half2") {
      auto StmtStrArg0 = getStmtSpelling(CE->getArg(0), *Result.Context);
      auto StmtStrArg1 = getStmtSpelling(CE->getArg(1), *Result.Context);
      std::string ReplStr{"cl::sycl::half2{"};
      ReplStr += StmtStrArg0;
      ReplStr += ".get_value(1), ";
      ReplStr += StmtStrArg1;
      ReplStr += ".get_value(1)}";
      emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
    } else {
      //__half2short_rd and __half2float
      static SSMap TypeMap{{"ll", "long long"},
                           {"ull", "unsigned long long"},
                           {"ushort", "unsigned short"},
                           {"uint", "unsigned int"},
                           {"half", "cl::sycl::half"}};
      auto StmtStrArg0 = getStmtSpelling(CE->getArg(0), *Result.Context);
      std::string RoundingMode;
      if (FuncName[FuncName.size() - 3] == '_')
        RoundingMode = FuncName.substr(FuncName.size() - 2);
      auto FN = FuncName.substr(2, FuncName.find('_', 2) - 2);
      auto Types = split(FN, '2');
      assert(Types.size() == 2);
      std::string ReplStr;
      auto T0 = TypeMap[Types[0]];
      auto T1 = TypeMap[Types[1]];
      if (!T0.empty())
        Types[0] = T0;
      if (!T1.empty())
        Types[1] = T1;
      ReplStr += "cl::sycl::vec<";
      ReplStr += Types[0];
      ReplStr += ", 1>{";
      ReplStr += StmtStrArg0;
      ReplStr += "}.convert<";
      ReplStr += Types[1];
      ReplStr += ", cl::sycl::rounding_mode::";
      ReplStr += RoundingModeMap[RoundingMode];
      ReplStr += ">().get_value(0)";
      emplaceTransformation(new ReplaceStmt(CE, true, FuncName, ReplStr));
    }
  }
}

REGISTER_RULE(MathFunctionsRule)

void SyncThreadsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(callExpr(callee(functionDecl(hasAnyName("__syncthreads"))),
                         hasAncestor(functionDecl().bind("func")))
                    .bind("syncthreads"),
                this);
}

void SyncThreadsRule::run(const MatchFinder::MatchResult &Result) {
  if (auto CE = getNodeAsType<CallExpr>(Result, "syncthreads")) {
    if (auto FD = getAssistNodeAsType<FunctionDecl>(Result, "func"))
      DeviceFunctionDecl::LinkRedecls(FD)->setItem();
    std::string Replacement = getItemName() + ".barrier()";
    emplaceTransformation(new ReplaceStmt(CE, std::move(Replacement)));
  }
}

REGISTER_RULE(SyncThreadsRule)

void KernelFunctionInfoRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      varDecl(hasType(recordDecl(hasName("cudaFuncAttributes")))).bind("decl"),
      this);
  MF.addMatcher(
      callExpr(callee(functionDecl(hasAnyName("cudaFuncGetAttributes"))))
          .bind("call"),
      this);
  MF.addMatcher(memberExpr(hasObjectExpression(hasType(
                               recordDecl(hasName("cudaFuncAttributes")))))
                    .bind("member"),
                this);
}

void KernelFunctionInfoRule::run(const MatchFinder::MatchResult &Result) {
  if (auto V = getNodeAsType<VarDecl>(Result, "decl"))
    emplaceTransformation(
        new ReplaceTypeInDecl(V, "sycl_kernel_function_info"));
  else if (auto C = getNodeAsType<CallExpr>(Result, "call")) {
    emplaceTransformation(
        new ReplaceToken(C->getBeginLoc(), "(get_kernel_function_info"));
    emplaceTransformation(new InsertAfterStmt(C, ", 0)"));
    auto FuncArg = C->getArg(1);
    emplaceTransformation(new InsertBeforeStmt(FuncArg, "(const void *)"));
  } else if (auto M = getNodeAsType<MemberExpr>(Result, "member")) {
    auto MemberName = M->getMemberNameInfo();
    auto NameMap = AttributesNamesMap.find(MemberName.getAsString());
    if (NameMap != AttributesNamesMap.end())
      emplaceTransformation(new ReplaceToken(MemberName.getBeginLoc(),
                                             std::string(NameMap->second)));
  }
}

REGISTER_RULE(KernelFunctionInfoRule)

void TypeCastRule::registerMatcher(MatchFinder &MF) {

  MF.addMatcher(
      declRefExpr(hasParent(implicitCastExpr(
                      hasParent(cStyleCastExpr(unless(
                          hasType(pointsTo(typedefDecl(hasName("double2"))))))),
                      hasType(pointsTo(typedefDecl(hasName("double2")))))))

          .bind("Double2CastExpr"),
      this);
}

void TypeCastRule::run(const MatchFinder::MatchResult &Result) {

  if (const DeclRefExpr *E =
          getNodeAsType<DeclRefExpr>(Result, "Double2CastExpr")) {
    std::string Name = E->getNameInfo().getName().getAsString();

    insertAroundStmt(E, "(&", "[0])");
  }
}

REGISTER_RULE(TypeCastRule)

void RecognizeAPINameRule::registerMatcher(MatchFinder &MF) {
  std::vector<std::string> AllAPINames =
      TranslationStatistics::GetAllAPINames();
  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(internal::Matcher<NamedDecl>(
                         new internal::HasNameMatcher(AllAPINames)))),
                     unless(hasAncestor(cudaKernelCallExpr())),
                     unless(callee(hasDeclContext(namedDecl(hasName("std")))))))
          .bind("APINamesUsed"),
      this);
}

const std::string
RecognizeAPINameRule::GetFunctionSignature(const FunctionDecl *Func) {

  std::string Buf;
  llvm::raw_string_ostream OS(Buf);
  OS << Func->getReturnType().getAsString() << " "
     << Func->getQualifiedNameAsString() << "(";

  for (unsigned int Index = 0; Index < Func->getNumParams(); Index++) {
    if (Index > 0) {
      OS << ",";
    }
    OS << QualType::getAsString(Func->parameters()[Index]->getType().split(),
                                PrintingPolicy{{}})
       << " " << Func->parameters()[Index]->getQualifiedNameAsString();
  }
  OS << ")";
  return OS.str();
}

void RecognizeAPINameRule::run(const MatchFinder::MatchResult &Result) {
  const CallExpr *C = getNodeAsType<CallExpr>(Result, "APINamesUsed");
  if (!C) {
    return;
  }

  std::string Namespace;
  const NamedDecl *ND = dyn_cast<NamedDecl>(C->getCalleeDecl());
  if (ND) {
    const auto *NSD = dyn_cast<NamespaceDecl>(ND->getDeclContext());
    if (NSD && !NSD->isInlineNamespace()) {
      Namespace = NSD->getName().str();
    }
  }

  std::string APIName = C->getCalleeDecl()->getAsFunction()->getNameAsString();

  if (!Namespace.empty()) {
    APIName = Namespace + "::" + APIName;
  }

  SrcAPIStaticsMap[GetFunctionSignature(C->getCalleeDecl()->getAsFunction())]++;

  if (!TranslationStatistics::IsTranslated(APIName)) {

    const SourceManager &SM = (*Result.Context).getSourceManager();
    const SourceLocation FileLoc = SM.getFileLoc(C->getBeginLoc());

    std::string SLStr = FileLoc.printToString(SM);

    std::size_t Pos = SLStr.find(':');
    std::string FileName = SLStr.substr(0, Pos);
    LOCStaticsMap[FileName][2]++;
    report(C->getBeginLoc(), Comments::API_NOT_MIGRATED, APIName.c_str());
  }
}

REGISTER_RULE(RecognizeAPINameRule)

void ASTTraversalManager::matchAST(ASTContext &Context, TransformSetTy &TS,
                                   StmtStringMap &SSM) {
  this->Context = &Context;
  for (auto &I : Storage) {
    I->registerMatcher(Matchers);
    if (auto TR = dyn_cast<TranslationRule>(&*I)) {
      TR->TM = this;
      TR->setTransformSet(TS);
    }
  }

  DebugInfo::printTranslationRules(Storage);

  Matchers.matchAST(Context);

  DebugInfo::printMatchedRules(Storage);
}

void ASTTraversalManager::emplaceAllRules(int SourceFileFlag) {
  std::vector<std::vector<std::string>> Rules;

  for (auto &F : ASTTraversalMetaInfo::getConstructorTable()) {

    auto RuleObj = (TranslationRule *)F.second();
    CommonRuleProperty RuleProperty = RuleObj->GetRuleProperty();

    auto RType = RuleProperty.RType;
    auto RulesDependon = RuleProperty.RulesDependon;

    if (RType & SourceFileFlag) {
      std::string CurrentRuleName = ASTTraversalMetaInfo::getName(F.first);
      std::vector<std::string> Vec;
      Vec.push_back(CurrentRuleName);
      for (auto const &RuleName : RulesDependon) {
        Vec.push_back(RuleName);
      }
      Rules.push_back(Vec);
    }
  }

  std::vector<std::string> SortedRules = ruleTopoSort(Rules);

  for (std::vector<std::string>::reverse_iterator it = SortedRules.rbegin();
       it != SortedRules.rend(); it++) {
    auto *ID = ASTTraversalMetaInfo::getID(*it);
    if (!ID) {
      llvm::errs() << "[ERROR] Rule\"" << *it << "\" not found\n";
      std::exit(MigrationError);
    }
    emplaceTranslationRule(ID);
  }
}

const CompilerInstance &TranslationRule::getCompilerInstance() {
  return TM->CI;
}
