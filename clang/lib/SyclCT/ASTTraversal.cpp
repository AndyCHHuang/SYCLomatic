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

auto parentStmt = []() {
  return anyOf(hasParent(compoundStmt()), hasParent(forStmt()),
               hasParent(whileStmt()), hasParent(doStmt()),
               hasParent(ifStmt()));
};

std::unordered_map<std::string, std::unordered_set</* Comment ID */ int>>
    TranslationRule::ReportedComment;

static std::set<SourceLocation> AttrExpansionFilter;

// Remember the location of the last inclusion directive for each file
static std::map<std::string, SourceLocation> IncludeLocations;

// Remember if a file has already included the cmath header or not
static std::set<std::string> MathHeaderFilter;
// Remember whether <complex> is included in one file.
static std::set<std::string> ComplexHeaderFilter;
// Remember whether <future> is included in one file.
static std::set<std::string> FutureHeaderFilter;
// Remember whether MKL headers are included in one file.
static std::set<std::string> MKLHeadersFilter;
// Remember whether <time.h> is included in one file.
static std::set<std::string> TimeHeaderFilter;

// Add '#include <cmath>' directive to the file where CE is located
InsertText *getCmathHeaderInsertTextForCallExpr(const CallExpr *CE,
                                                const SourceManager *SM) {
  auto Loc = CE->getBeginLoc();
  std::string Path = SyclctGlobalInfo::getSourceManager().getFilename(
      SyclctGlobalInfo::getSourceManager().getExpansionLoc(Loc));
  makeCanonical(Path);
  // No need to insert or already inserted
  if (MathHeaderFilter.find(Path) != MathHeaderFilter.end())
    return nullptr;

  auto IncludeLoc = IncludeLocations[Path];
  MathHeaderFilter.insert(Path);
  return new InsertText(IncludeLoc,
                        getNL() + std::string("#include <cmath>") + getNL());
}

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

bool IncludesCallbacks::ShouldEnter(StringRef FileName, bool IsAngled) {
#ifdef _WIN32
  std::string Name = FileName.str();
  return !IsAngled || !MapNames::isInSet(MapNames::ThrustFileExcludeSet, Name);
#else
  return true;
#endif
}

void IncludesCallbacks::InclusionDirective(
    SourceLocation HashLoc, const Token &IncludeTok, StringRef FileName,
    bool IsAngled, CharSourceRange FilenameRange, const FileEntry *File,
    StringRef SearchPath, StringRef RelativePath, const Module *Imported,
    SrcMgr::CharacteristicKind FileType) {
  // Record the locations of inclusion directives
  // The last inclusion diretive of a file will be remembered
  std::string Path = SyclctGlobalInfo::getSourceManager().getFilename(
      SyclctGlobalInfo::getSourceManager().getExpansionLoc(HashLoc));
  makeCanonical(Path);
  IncludeLocations[Path] = FilenameRange.getEnd();

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

  // Record that math header is included in this file
  if (IsAngled && (FileName.compare(StringRef("math.h")) == 0 ||
                   FileName.compare(StringRef("cmath")) == 0)) {
    std::string Path = SyclctGlobalInfo::getSourceManager().getFilename(
        SyclctGlobalInfo::getSourceManager().getExpansionLoc(HashLoc));
    makeCanonical(Path);
    MathHeaderFilter.insert(Path);
  }

  // Replace "#include <cublas_v2.h>" and "#include <cublas.h>" with
  // <mkl_blas_sycl.hpp>, <mkl_lapack_sycl.hpp> and <sycl_types.hpp>
  if ((IsAngled && FileName.compare(StringRef("cublas_v2.h")) == 0) ||
      (IsAngled && FileName.compare(StringRef("cublas.h")) == 0) ||
      (IsAngled && FileName.compare(StringRef("cusolverDn.h")) == 0)) {
    std::string Path = SyclctGlobalInfo::getSourceManager().getFilename(
        SyclctGlobalInfo::getSourceManager().getExpansionLoc(HashLoc));
    makeCanonical(Path);
    if (MKLHeadersFilter.find(Path) == MKLHeadersFilter.end()) {
      MKLHeadersFilter.insert(Path);
      std::string Replacement = std::string("#include <mkl_blas_sycl.hpp>") +
                                getNL() + "#include <mkl_lapack_sycl.hpp>" +
                                getNL() + "#include <sycl_types.hpp>" +
                                getNL() + "#include <syclct/syclct_blas.hpp>";
      TransformSet.emplace_back(new ReplaceInclude(
          CharSourceRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                          /*IsTokenRange=*/false),
          std::move(Replacement)));
    } else {
      TransformSet.emplace_back(new ReplaceInclude(
          CharSourceRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                          /*IsTokenRange=*/false),
          ""));
    }
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
      std::string Replacement = std::string("<dpstd/algorithm>") + getNL() +
                                "#include <dpstd/execution>" + getNL() +
                                "#include <syclct/syclct_dpstd_utils.hpp>";
      if (!SyclHeaderInserted) {
        Replacement = std::string("<CL/sycl.hpp>") + getNL() +
                      "#include <syclct/syclct.hpp>" + getNL() + "#include " +
                      Replacement;
        SyclHeaderInserted = true;
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
    std::string Path = SyclctGlobalInfo::getSourceManager().getFilename(
        SyclctGlobalInfo::getSourceManager().getExpansionLoc(Loc));
    makeCanonical(Path);
    IncludeLocations[Path] = Loc;
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

  if (isVarRef(Lhs) && (getVarType(Lhs) == "enum cudaError" ||
                        getVarType(Lhs) == "enum cudaError_enum")) {
    Literal = Rhs;
  } else if (isVarRef(Rhs) && (getVarType(Rhs) == "enum cudaError" ||
                               getVarType(Rhs) == "enum cudaError_enum")) {
    Literal = Lhs;
  } else
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
  return isVarRef(E) && (getVarType(E) == "enum cudaError" ||
                         getVarType(E) == "enum cudaError_enum");
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
  report(CE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, OSS.str());
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
  for (unsigned i = 0; i < NumArgs; ++i) {
    const Expr *Arg = CE->getArg(i);
    if (auto *ImpCast = dyn_cast<ImplicitCastExpr>(Arg)) {
      if (ImpCast->getCastKind() != clang::CK_LValueToRValue) {
        if (i == 0) {
          insertAroundStmt(Arg, "(" + TypeName + "*)(", ")");
        } else {
          insertAroundStmt(Arg, "(" + TypeName + ")(", ")");
        }
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

void ThrustFunctionRule::registerMatcher(MatchFinder &MF) {
  std::vector<std::string> ThrustFuncNames(MapNames::ThrustFuncNamesMap.size());
  std::transform(
      MapNames::ThrustFuncNamesMap.begin(), MapNames::ThrustFuncNamesMap.end(),
      ThrustFuncNames.begin(),
      [](const std::pair<std::string, MapNames::ThrustFuncReplInfo> &p) {
        return p.first;
      });

  auto hasAnyThrustFuncName = [&]() {
    return internal::Matcher<NamedDecl>(
        new internal::HasNameMatcher(ThrustFuncNames));
  };

  MF.addMatcher(callExpr(callee(functionDecl(
                             hasAnyThrustFuncName(),
                             hasDeclContext(namespaceDecl(hasName("thrust"))))))
                    .bind("thrustFuncCall"),
                this);
}

void ThrustFunctionRule::run(const MatchFinder::MatchResult &Result) {
  auto UniqueName = [](const Stmt *S) {
    auto &SM = SyclctGlobalInfo::getSourceManager();
    SourceLocation Loc = S->getBeginLoc();
    return getHashAsString(Loc.printToString(SM)).substr(0, 6);
  };
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "thrustFuncCall");
  if (!CE) {
    return;
  }
  const std::string ThrustFuncName = CE->getDirectCallee()->getName().str();
  auto ReplInfo = MapNames::ThrustFuncNamesMap.find(ThrustFuncName);
  assert(ReplInfo != MapNames::ThrustFuncNamesMap.end());
  auto NewName = ReplInfo->second.ReplName;
  emplaceTransformation(
      new ReplaceCalleeName(CE, std::move(NewName), ThrustFuncName));
  assert(CE->getNumArgs() > 0);
  auto ExtraParam = ReplInfo->second.ExtraParam;
  if (!ExtraParam.empty()) {
    // This is a temporary fix until, the DPC++ compiler and the DPC++ library
    // support creating a SYCL execution policy without creating a unique
    // one for every use
    if (ExtraParam == "dpstd::execution::sycl") {
      std::string Name = UniqueName(CE);
      ExtraParam = "dpstd::execution::make_sycl_policy<class Policy_" +
                   UniqueName(CE) + ">(dpstd::execution::sycl)";
    }
    emplaceTransformation(
        new InsertBeforeStmt(CE->getArg(0), ExtraParam + ", "));
  }
}

REGISTER_RULE(ThrustFunctionRule)

auto TypedefNames =
    hasAnyName("dim3", "cudaError_t", "CUresult", "CUcontext", "cudaEvent_t",
               "cudaStream_t", "__half", "__half2", "half", "half2",
               "cublasStatus_t", "cuComplex", "cuDoubleComplex",
               "cublasFillMode_t", "cublasDiagType_t", "cublasSideMode_t",
               "cublasOperation_t", "cublasStatus", "cusolverDnHandle_t",
               "cusolverStatus_t", "cusolverEigType_t", "cusolverEigMode_t");
auto EnumTypeNames = hasAnyName("cudaError", "cufftResult_t", "cudaError_enum");
// CUstream_st and CUevent_st are the actual types of cudaStream_t and
// cudaEvent_st respectively
auto RecordTypeNames =
    hasAnyName("cudaDeviceProp", "CUstream_st", "CUevent_st");
auto HandleTypeNames = hasAnyName("cublasHandle_t", "cusolverDnHandle_t");

auto TemplateRecordTypeNames =
    hasAnyName("device_vector", "device_ptr", "host_vector");

// Rule for types replacements in var declarations and field declarations
void TypeInDeclRule::registerMatcher(MatchFinder &MF) {
  auto HasCudaType = anyOf(
      hasType(typedefDecl(TypedefNames)), hasType(enumDecl(EnumTypeNames)),
      hasType(cxxRecordDecl(RecordTypeNames)),
      hasType(classTemplateSpecializationDecl(TemplateRecordTypeNames)));

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

  auto HandleTypedefs =
      typedefType(hasDeclaration(typedefDecl(HandleTypeNames)));

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

  MF.addMatcher(unaryExprOrTypeTraitExpr(
                    hasArgumentOfType(anyOf(
                        asString("cublasStatus_t"), asString("cublasStatus"),
                        asString("cusolverStatus_t"), asString("cuComplex"),
                        asString("cuDoubleComplex"), asString("cublasHandle_t"),
                        asString("cusolverDnHandle_t"))))
                    .bind("TypeInUnaryExprOrTypeTraitExpr"),
                this);

  MF.addMatcher(
      cStyleCastExpr(
          hasDestinationType(anyOf(
              asString("cublasFillMode_t"), asString("cublasDiagType_t"),
              asString("cublasSideMode_t"), asString("cublasOperation_t"),
              asString("cusolverEigType_t"), asString("cusolverEigMode_t"))))
          .bind("cStyleCastExpr"),
      this);
  // TODO: HandleType in template, in macro body, assigined, as function param
  // and as macro argument
  MF.addMatcher(
      varDecl(
          allOf(anyOf(hasType(typedefDecl(HandleTypeNames)),
                      hasType(pointsTo(typedefDecl(HandleTypeNames))),
                      hasType(pointsTo(pointsTo(typedefDecl(HandleTypeNames)))),
                      hasType(references(typedefDecl(HandleTypeNames))),
                      hasType(arrayType(hasElementType(HandleTypedefs))),
                      hasType(
                          arrayType(hasElementType(pointsTo(HandleTypedefs)))),
                      hasType(arrayType(
                          hasElementType(pointsTo(pointsTo(HandleTypedefs)))))),
                unless(hasType(substTemplateTypeParmType()))),
          hasAncestor(functionDecl(
              anyOf(hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))))
          .bind("TypeInVarDeclDevice"),
      this);

  MF.addMatcher(
      varDecl(
          allOf(anyOf(hasType(typedefDecl(HandleTypeNames)),
                      hasType(pointsTo(typedefDecl(HandleTypeNames))),
                      hasType(pointsTo(pointsTo(typedefDecl(HandleTypeNames)))),
                      hasType(references(typedefDecl(HandleTypeNames))),
                      hasType(arrayType(hasElementType(HandleTypedefs))),
                      hasType(
                          arrayType(hasElementType(pointsTo(HandleTypedefs)))),
                      hasType(arrayType(
                          hasElementType(pointsTo(pointsTo(HandleTypedefs)))))),
                unless(hasType(substTemplateTypeParmType()))),
          hasAncestor(functionDecl(unless(
              allOf(hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal))))))
          .bind("TypeInVarDecl"),
      this);

  MF.addMatcher(
      fieldDecl(
          allOf(anyOf(hasType(typedefDecl(HandleTypeNames)),
                      hasType(pointsTo(typedefDecl(HandleTypeNames))),
                      hasType(pointsTo(pointsTo(typedefDecl(HandleTypeNames)))),
                      hasType(references(typedefDecl(HandleTypeNames))),
                      hasType(arrayType(hasElementType(HandleTypedefs))),
                      hasType(
                          arrayType(hasElementType(pointsTo(HandleTypedefs)))),
                      hasType(arrayType(
                          hasElementType(pointsTo(pointsTo(HandleTypedefs)))))),
                unless(hasType(substTemplateTypeParmType()))),
          hasAncestor(functionDecl(
              anyOf(hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))))
          .bind("TypeInFieldDeclDevice"),
      this);

  MF.addMatcher(
      fieldDecl(
          allOf(anyOf(hasType(typedefDecl(HandleTypeNames)),
                      hasType(pointsTo(typedefDecl(HandleTypeNames))),
                      hasType(pointsTo(pointsTo(typedefDecl(HandleTypeNames)))),
                      hasType(references(typedefDecl(HandleTypeNames))),
                      hasType(arrayType(hasElementType(HandleTypedefs))),
                      hasType(
                          arrayType(hasElementType(pointsTo(HandleTypedefs)))),
                      hasType(arrayType(
                          hasElementType(pointsTo(pointsTo(HandleTypedefs)))))),
                unless(hasType(substTemplateTypeParmType()))),
          hasAncestor(functionDecl(unless(
              allOf(hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal))))))
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

  std::string TypeName = Strs.back();

  // remove possible template parameters from TypeName
  size_t bracketBeginPos = TypeName.find('<');
  if (bracketBeginPos != std::string::npos) {
    size_t bracketEndPos = TypeName.rfind('>');
    TypeName.erase(bracketBeginPos, bracketEndPos - bracketBeginPos + 1);
  }
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
  const DeclaratorDecl *DD =
      getNodeAsType<VarDecl>(Result, "TypeInVarDeclDevice");
  const UnaryExprOrTypeTraitExpr *UETTE;
  const CStyleCastExpr *CSCE;
  QualType QT;
  bool HasDeviceAttr = false;
  bool IsUETTE = false;
  bool IsCSCE = false;
  if ((DD) ||
      ((DD = getNodeAsType<VarDecl>(Result, "TypeInFieldDeclDevice")))) {
    QT = DD->getType();
    HasDeviceAttr = true;
  } else if ((DD = getNodeAsType<VarDecl>(Result, "TypeInVarDecl"))) {
    QT = DD->getType();
  } else if ((DD = getNodeAsType<FieldDecl>(Result, "TypeInFieldDecl"))) {
    QT = DD->getType();
  } else if ((UETTE = getNodeAsType<UnaryExprOrTypeTraitExpr>(
                  Result, "TypeInUnaryExprOrTypeTraitExpr"))) {
    IsUETTE = true;
  } else if ((CSCE = getNodeAsType<CStyleCastExpr>(Result, "cStyleCastExpr"))) {
    IsCSCE = true;
  } else {
    return;
  }
  SourceManager *SM = Result.SourceManager;
  unsigned int Loc;
  std::string TypeStr;
  SourceLocation BeginLoc;
  int Len = 0;
  bool IsMacro = false;

  TypeSourceInfo *ArgTypeInfo = nullptr;
  if (IsUETTE) {
    if ((ArgTypeInfo = UETTE->getArgumentTypeInfo())) {
      BeginLoc = ArgTypeInfo->getTypeLoc().getBeginLoc();
    } else {
      return;
    }
  } else if (IsCSCE) {
    BeginLoc =
        CSCE->getTypeInfoAsWritten()->getTypeLoc().getSourceRange().getBegin();
  } else {
    if ((ArgTypeInfo = DD->getTypeSourceInfo())) {
      BeginLoc = ArgTypeInfo->getTypeLoc().getSourceRange().getBegin();
    } else {
      return;
    }
  }

  if (BeginLoc.isMacroID()) {
    IsMacro = true;
    auto SpellingLocation = SM->getSpellingLoc(BeginLoc);
    if (SyclctGlobalInfo::isInCudaPath(SpellingLocation)) {
      BeginLoc = SM->getExpansionLoc(BeginLoc);
    } else {
      BeginLoc = SpellingLocation;
    }
  }

  Loc = BeginLoc.getRawEncoding();
  if (DupFilter.find(Loc) != DupFilter.end())
    return;

  auto BeginLocChar = SM->getCharacterData(SM->getExpansionLoc(BeginLoc));
  Len = Lexer::MeasureTokenLength(BeginLoc, *SM, LangOptions());

  if (IsUETTE) {
    TypeStr = std::string(BeginLocChar, Len);
  } else if (IsCSCE) {
    TypeStr = CSCE->getType().getAsString();
  } else {
    if (QT->isArrayType()) {
      auto ArrType = Result.Context->getAsArrayType(QT);
      auto EleType = ArrType->getElementType();
      TypeStr = EleType.getAsString();
    } else {
      TypeStr = QT.getAsString();
    }
  }

  // Add '#include <complex>' directive to the file only once
  if (TypeStr == "cuComplex" || TypeStr == "cuDoubleComplex") {
    insertIncludeFile(BeginLoc, ComplexHeaderFilter,
                      getNL() + std::string("#include <complex>") + getNL());
  }

  auto Replacement = getReplacementForType(TypeStr);
  if (Replacement.empty())
    // TODO report migration error
    return;

  if (HasDeviceAttr) {
    auto SL = DD->getTypeSourceInfo()->getTypeLoc().getBeginLoc();
    if (SL.isMacroID())
      SL = (Result.SourceManager)->getExpansionLoc(SL);
    report(SL, Diagnostics::HANDLE_IN_DEVICE, TypeStr);
    return;
  }

  if (IsUETTE || IsCSCE) {
    emplaceTransformation(
        new ReplaceText(BeginLoc, Len, std::move(Replacement)));
  } else {
    if (IsMacro) {
      emplaceTransformation(
          new ReplaceText(BeginLoc, Len, std::move(Replacement)));
    } else {
      emplaceTransformation(new ReplaceTypeInDecl(DD, std::move(Replacement)));
    }
  }
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

static internal::Matcher<NamedDecl> vectorTypeName() {
  std::vector<std::string> TypeNames(MapNames::SupportedVectorTypes.begin(),
                                     MapNames::SupportedVectorTypes.end());
  return internal::Matcher<NamedDecl>(new internal::HasNameMatcher(TypeNames));
}

namespace clang {
namespace ast_matchers {

AST_MATCHER(QualType, vectorType) {
  return (MapNames::SupportedVectorTypes.find(Node.getAsString()) !=
          MapNames::SupportedVectorTypes.end());
}

AST_MATCHER(TypedefDecl, typedefVecDecl) {
  if (!Node.getUnderlyingType().getBaseTypeIdentifier())
    return false;

  const std::string BaseTypeName =
      Node.getUnderlyingType().getBaseTypeIdentifier()->getName().str();
  return (MapNames::SupportedVectorTypes.find(BaseTypeName) !=
          MapNames::SupportedVectorTypes.end());
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

  MF.addMatcher(
      fieldDecl(anyOf(basicType(), ptrType(), arrType(), referenceType()),
                hasParent(cxxRecordDecl().bind("cxxRecordDeclParent")))
          .bind("fieldvecVarDecl"),
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
  if (const VarDecl *D = getNodeAsType<VarDecl>(Result, "vecVarDecl")) {
    replaceTypeName(D->getType(),
                    D->getTypeSourceInfo()->getTypeLoc().getBeginLoc(), true);
  }

  // struct benchtype{
  // ...;
  // uint2 u32;
  // };
  // =>
  // struct benchtype {
  // ...;
  // cl::sycl::uint2 u32;
  // };
  if (const FieldDecl *FD =
          getNodeAsType<FieldDecl>(Result, "fieldvecVarDecl")) {
    auto D = getNodeAsType<CXXRecordDecl>(Result, "cxxRecordDeclParent");
    if (D && D->isUnion()) {
      // To add a default member initializer list "{}" to the
      // vector variant member of the union, because a union contains a
      // non-static data member with a non-trivial default constructor, the
      // default constructor of the union will be deleted by default.
      SourceManager *SM = Result.SourceManager;
      auto Loc = FD->getEndLoc().getLocWithOffset(Lexer::MeasureTokenLength(
          FD->getEndLoc(), *SM, Result.Context->getLangOpts()));
      emplaceTransformation(new ReplaceToken(Loc.getLocWithOffset(-1), "{}"));
    }
    replaceTypeName(FD->getType(),
                    FD->getTypeSourceInfo()->getTypeLoc().getBeginLoc(), true);
  }

  // typedef int2 xxx => typedef cl::sycl::int2 xxx
  if (const TypedefDecl *TD =
          getNodeAsType<TypedefDecl>(Result, "typeDefDecl")) {
    replaceTypeName(TD->getUnderlyingType(),
                    TD->getTypeSourceInfo()->getTypeLoc().getBeginLoc());
  }

  // int2 func() => cl::sycl::int2 func()
  if (const FunctionDecl *FD =
          getNodeAsType<FunctionDecl>(Result, "funcReturnsVectorType")) {
    replaceTypeName(FD->getReturnType(),
                    FD->getReturnTypeSourceRange().getBegin());
  }
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
  if (MapNames::replaceName(MapNames::MemberNamesMap, MemberName))
    emplaceTransformation(
        new RenameFieldInMemberExpr(ME, std::move(MemberName)));
}

void VectorTypeMemberAccessRule::run(const MatchFinder::MatchResult &Result) {
  // xxx = int2.x => xxx = static_cast<int>(int2.x())
  if (const MemberExpr *ME =
          getNodeAsType<MemberExpr>(Result, "VecMemberExpr")) {
    auto Parents = Result.Context->getParents(*ME);
    if (Parents.size() == 0) {
      return;
    }
    auto *UO = Parents[0].get<clang::UnaryOperator>();
    if (UO && UO->getOpcode() == clang::UO_AddrOf) {
      // As access to vector fields’ address is not supported in SYCL spec,
      // Implementation here is by defining a local variable to migrate
      // vector fields’ address.
      // e.g:
      //    uchar4 data;
      //    *(&data.x) = 'a';
      // =>
      //    cl::sycl::uchar4 data;
      //    {
      //    unsigned char x_ct = data.x();
      //    *(&x_ct) = 'a';
      //    data.x() = x_ct;
      //    }
      // TODO: Need to handle the situations below.
      // 1. if/while condition stmt
      // 2. macro stmt
      // 3. vec field address assignment expr, such as int i=&a.x
      // 4. one dimension vec, such as char1
      SourceManager *SM = Result.SourceManager;
      auto EndLoc = ME->getEndLoc().getLocWithOffset(Lexer::MeasureTokenLength(
          ME->getEndLoc(), *SM, Result.Context->getLangOpts()));

      const char *Start = SM->getCharacterData(ME->getBeginLoc());
      const char *End = SM->getCharacterData(EndLoc);
      const std::string MExprStr(Start, End - Start);

      std::string VecField = MExprStr + "()";
      std::string VarType = ME->getType().getAsString();
      std::string VarName =
          ME->getMemberNameInfo().getAsString() + getCTFixedSuffix();

      std::string LocalVarDecl =
          VarType + " " + VarName + " = " + VecField + ";" + getNL();
      std::string LocalVarDeclRef = VecField + " = " + VarName + ";";

      auto SR = getScopeInsertRange(ME);
      SourceLocation StmtBegin = SR.getBegin(), StmtEndAfterSemi = SR.getEnd();

      std::string IndentStr = getIndent(StmtBegin, *SM).str();
      std::string PrefixInsertStr = std::string("{") + getNL();
      PrefixInsertStr += IndentStr + LocalVarDecl;

      std::string SuffixInsertStr =
          getNL() + IndentStr + LocalVarDeclRef + getNL() + IndentStr + "}";

      emplaceTransformation(new ReplaceToken(
          ME->getBeginLoc(), EndLoc.getLocWithOffset(-1), std::move(VarName)));

      insertAroundRange(StmtBegin, StmtEndAfterSemi,
                        PrefixInsertStr + IndentStr,
                        std::move(SuffixInsertStr));
    } else {
      std::ostringstream CastPrefix;
      CastPrefix << "static_cast<" << ME->getType().getAsString() << ">(";
      insertAroundStmt(ME, CastPrefix.str(), ")");
      renameMemberField(ME);
    }
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
  default: { return false; }
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
    return (MapNames::SupportedVectorTypes.find(TypeName) !=
            MapNames::SupportedVectorTypes.end());
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
  const std::string NL = getNL(FD->getBeginLoc(), SM);

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
    for (const std::string &TypeName : MapNames::SupportedVectorTypes) {
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

  MF.addMatcher(cxxConstructExpr(
                    hasType(typedefDecl(hasName("dim3"))), argumentCountIs(3),
                    anyOf(hasParent(varDecl()), hasParent(exprWithCleanups())),
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
          unless(hasParent(exprWithCleanups())),
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
    // deflt = dim3(3);
    return new ReplaceDim3Ctor(Ctor, false /*isDecl*/);
  } else if (auto Ctor = getNodeAsType<CXXConstructExpr>(Result, "dim3Top")) {
    // dim3 d3_6_3 = dim3(ceil(test.x + NUM), NUM + test.y, NUM + test.z + NUM);
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

template <class T>
void NamedTranslationRule<T>::insertIncludeFile(
    SourceLocation SL, std::set<std::string> &HeaderFilter,
    std::string &&InsertFile) {
  std::string Path = SyclctGlobalInfo::getSourceManager().getFilename(
      SyclctGlobalInfo::getSourceManager().getExpansionLoc(SL));
  makeCanonical(Path);
  SourceLocation IncludeLoc = IncludeLocations[Path];
  if (HeaderFilter.find(Path) == HeaderFilter.end()) {
    HeaderFilter.insert(Path);
    emplaceTransformation(new InsertText(IncludeLoc, std::move(InsertFile)));
  }
}

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

  // TODO: cublasHandle_t cannot migrate to cl::sycl::queue simplely.
  // cublasHandle_t is a struct, so it could be returned as value by user
  // defined function. But cl::sycl::queue cannot be return as value.
  // It will be replaced by a handle type later.
  MF.addMatcher(
      functionDecl(
          returns(anyOf(
              asString("cuComplex"), asString("cuDoubleComplex"),
              asString("cublasHandle_t"), asString("cublasStatus_t"),
              asString("cublasFillMode_t"), asString("cublasDiagType_t"),
              asString("cublasSideMode_t"), asString("cublasOperation_t"),
              asString("cusolverDnHandle_t"), asString("cusolverStatus_t"),
              asString("cusolverEigType_t"), asString("cusolverEigMode_t"))))
          .bind("functionDeclWithTypedef"),
      this);
}

void ReturnTypeRule::run(const MatchFinder::MatchResult &Result) {
  const FunctionDecl *FD = nullptr;
  std::string TypeName;
  SourceManager *SM = Result.SourceManager;

  if ((FD = getNodeAsType<FunctionDecl>(Result, "functionDecl"))) {
    const clang::Type *Type = FD->getReturnType().getTypePtr();
    if (Type == nullptr)
      return;
    TypeName = Type->getCanonicalTypeInternal()
                   .getBaseTypeIdentifier()
                   ->getName()
                   .str();
  } else if ((FD = getNodeAsType<FunctionDecl>(Result,
                                               "functionDeclWithTypedef"))) {
    const clang::Type *Type = FD->getReturnType().getTypePtr();
    if (Type == nullptr)
      return;
    auto TDT = static_cast<const TypedefType *>(Type);
    if (TDT == nullptr)
      return;
    TypeName = TDT->getDecl()->getName().str();
  } else {
    return;
  }

  // Add '#include <complex>' directive to the file only once
  if (TypeName == "cuComplex" || TypeName == "cuDoubleComplex") {
    SourceLocation SL = FD->getBeginLoc();
    insertIncludeFile(SL, ComplexHeaderFilter,
                      getNL() + std::string("#include <complex>") + getNL());
  }

  SrcAPIStaticsMap[TypeName]++;
  auto Search = MapNames::TypeNamesMap.find(TypeName);
  if (Search == MapNames::TypeNamesMap.end()) {
    // TODO report migration error
    return;
  }
  std::string Replacement = Search->second;

  auto BeginLoc = FD->getBeginLoc();
  if (BeginLoc.isMacroID()) {
    auto SpellingLocation = SM->getSpellingLoc(BeginLoc);
    if (SyclctGlobalInfo::isInCudaPath(SpellingLocation)) {
      BeginLoc = SM->getExpansionLoc(BeginLoc);
    } else {
      BeginLoc = SpellingLocation;
    }
    auto Len = Lexer::MeasureTokenLength(BeginLoc, *SM, LangOptions());
    emplaceTransformation(
        new ReplaceText(BeginLoc, Len, std::move(Replacement)));
  } else {
    emplaceTransformation(new ReplaceReturnType(FD, std::move(Replacement)));
  }
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
  if (Parents.size() != 1) {
    return;
  }
  auto MemberName = ME->getMemberNameInfo().getAsString();
  if (MemberName == "sharedMemPerBlock") {
    report(ME->getBeginLoc(), Diagnostics::LOCAL_MEM_SIZE);
  } else if (MemberName == "maxGridSize") {
    report(ME->getBeginLoc(), Diagnostics::MAX_GRID_SIZE);
  }

  auto Search = PropNamesMap.find(MemberName);
  if (Search == PropNamesMap.end()) {
    // TODO report migration error
    return;
  }
  if (Parents[0].get<clang::ImplicitCastExpr>()) {
    // migrate to get_XXX() eg. "b=a.minor" to "b=a.get_minor_version()"
    emplaceTransformation(
        new RenameFieldInMemberExpr(ME, "get_" + Search->second + "()"));
  } else if (auto *BO = Parents[0].get<clang::BinaryOperator>()) {
    // migrate to set_XXX() eg. "a.minor = 1" to "a.set_minor_version(1)"
    if (BO->getOpcode() == clang::BO_Assign) {
      emplaceTransformation(
          new RenameFieldInMemberExpr(ME, "set_" + Search->second));
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
  MF.addMatcher(declRefExpr(to(enumConstantDecl(hasType(enumDecl(anyOf(
                                hasName("cudaError"), hasName("cufftResult_t"),
                                hasName("cudaError_enum")))))))
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

// Rule for BLAS enums.
// All BLAS enums have the prefix CUBLAS_
// Migrate BLAS status values to corresponding int values
// Example: migrate CUBLAS_STATUS_SUCCESS to 0
// Other BLAS named values are migrated to corresponding named values
// Example: migrate CUBLAS_OP_N to mkl::transpose::nontrans
void BLASEnumsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      declRefExpr(to(enumConstantDecl(matchesName("CUBLAS_STATUS.*"))))
          .bind("BLASStatusConstants"),
      this);
  MF.addMatcher(declRefExpr(to(enumConstantDecl(matchesName(
                                "(CUBLAS_OP.*)|(CUBLAS_SIDE.*)|(CUBLAS_FILL_"
                                "MODE.*)|(CUBLAS_DIAG.*)"))))
                    .bind("BLASNamedValueConstants"),
                this);
}

void BLASEnumsRule::run(const MatchFinder::MatchResult &Result) {
  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "BLASStatusConstants")) {
    auto *EC = cast<EnumConstantDecl>(DE->getDecl());
    emplaceTransformation(new ReplaceStmt(DE, EC->getInitVal().toString(10)));
  }

  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "BLASNamedValueConstants")) {
    auto *EC = cast<EnumConstantDecl>(DE->getDecl());
    std::string Name = EC->getNameAsString();
    auto Search = MapNames::BLASEnumsMap.find(Name);
    if (Search == MapNames::BLASEnumsMap.end()) {
      syclct_unreachable("migration error");
      return;
    }
    std::string Replacement = Search->second;
    emplaceTransformation(new ReplaceStmt(DE, std::move(Replacement)));
  }
}

REGISTER_RULE(BLASEnumsRule)

void BLASFunctionCallRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName(
        "make_cuComplex", "make_cuDoubleComplex",
        /*Regular BLAS API*/
        /*Regular helper*/
        "cublasCreate_v2", "cublasDestroy_v2", "cublasSetVector",
        "cublasGetVector", "cublasSetVectorAsync", "cublasGetVectorAsync",
        "cublasSetMatrix", "cublasGetMatrix", "cublasSetMatrixAsync",
        "cublasGetMatrixAsync",
        /*Regular level 1*/
        "cublasIsamax_v2", "cublasIdamax_v2", "cublasIcamax_v2",
        "cublasIzamax_v2", "cublasIsamin_v2", "cublasIdamin_v2",
        "cublasIcamin_v2", "cublasIzamin_v2", "cublasSasum_v2",
        "cublasDasum_v2", "cublasScasum_v2", "cublasDzasum_v2",
        "cublasSaxpy_v2", "cublasDaxpy_v2", "cublasCaxpy_v2", "cublasZaxpy_v2",
        "cublasScopy_v2", "cublasDcopy_v2", "cublasCcopy_v2", "cublasZcopy_v2",
        "cublasSdot_v2", "cublasDdot_v2", "cublasCdotu_v2", "cublasCdotc_v2",
        "cublasZdotu_v2", "cublasZdotc_v2", "cublasSnrm2_v2", "cublasDnrm2_v2",
        "cublasScnrm2_v2", "cublasDznrm2_v2", "cublasSrot_v2", "cublasDrot_v2",
        "cublasCsrot_v2", "cublasZdrot_v2", "cublasSrotg_v2", "cublasDrotg_v2",
        "cublasCrotg_v2", "cublasZrotg_v2", "cublasSrotm_v2", "cublasDrotm_v2",
        "cublasSrotmg_v2", "cublasDrotmg_v2", "cublasSscal_v2",
        "cublasDscal_v2", "cublasCscal_v2", "cublasCsscal_v2", "cublasZscal_v2",
        "cublasZdscal_v2", "cublasSswap_v2", "cublasDswap_v2", "cublasCswap_v2",
        "cublasZswap_v2",
        /*Regular level 2*/
        "cublasSgbmv_v2", "cublasDgbmv_v2", "cublasCgbmv_v2", "cublasZgbmv_v2",
        "cublasSgemv_v2", "cublasDgemv_v2", "cublasCgemv_v2", "cublasZgemv_v2",
        "cublasSger_v2", "cublasDger_v2", "cublasCgeru_v2", "cublasCgerc_v2",
        "cublasZgeru_v2", "cublasZgerc_v2", "cublasSsbmv_v2", "cublasDsbmv_v2",
        "cublasSspmv_v2", "cublasDspmv_v2", "cublasSspr_v2", "cublasDspr_v2",
        "cublasSspr2_v2", "cublasDspr2_v2", "cublasSsymv_v2", "cublasDsymv_v2",
        "cublasSsyr_v2", "cublasDsyr_v2", "cublasSsyr2_v2", "cublasDsyr2_v2",
        "cublasStbmv_v2", "cublasDtbmv_v2", "cublasCtbmv_v2", "cublasZtbmv_v2",
        "cublasStbsv_v2", "cublasDtbsv_v2", "cublasCtbsv_v2", "cublasZtbsv_v2",
        "cublasStpmv_v2", "cublasDtpmv_v2", "cublasCtpmv_v2", "cublasZtpmv_v2",
        "cublasStpsv_v2", "cublasDtpsv_v2", "cublasCtpsv_v2", "cublasZtpsv_v2",
        "cublasStrmv_v2", "cublasDtrmv_v2", "cublasCtrmv_v2", "cublasZtrmv_v2",
        "cublasStrsv_v2", "cublasDtrsv_v2", "cublasCtrsv_v2", "cublasZtrsv_v2",
        "cublasChemv_v2", "cublasZhemv_v2", "cublasChbmv_v2", "cublasZhbmv_v2",
        "cublasChpmv_v2", "cublasZhpmv_v2", "cublasCher_v2", "cublasZher_v2",
        "cublasCher2_v2", "cublasZher2_v2", "cublasChpr_v2", "cublasZhpr_v2",
        "cublasChpr2_v2", "cublasZhpr2_v2",
        /*Regular level 3*/
        "cublasSgemm_v2", "cublasDgemm_v2", "cublasCgemm_v2", "cublasZgemm_v2",
        "cublasCgemm3m", "cublasZgemm3m", "cublasSsymm_v2", "cublasDsymm_v2",
        "cublasCsymm_v2", "cublasZsymm_v2", "cublasSsyrk_v2", "cublasDsyrk_v2",
        "cublasCsyrk_v2", "cublasZsyrk_v2", "cublasSsyr2k_v2",
        "cublasDsyr2k_v2", "cublasCsyr2k_v2", "cublasZsyr2k_v2",
        "cublasStrsm_v2", "cublasDtrsm_v2", "cublasCtrsm_v2", "cublasZtrsm_v2",
        "cublasChemm_v2", "cublasZhemm_v2", "cublasCherk_v2", "cublasZherk_v2",
        "cublasCher2k_v2", "cublasZher2k_v2", "cublasSsyrkx", "cublasDsyrkx",
        "cublasStrmm_v2", "cublasDtrmm_v2", "cublasCtrmm_v2", "cublasZtrmm_v2",
        /*Extensions*/
        "cublasSgetrfBatched", "cublasDgetrfBatched", "cublasCgetrfBatched",
        "cublasZgetrfBatched", "cublasSgetriBatched", "cublasDgetriBatched",
        "cublasCgetriBatched", "cublasZgetriBatched", "cublasSgeqrfBatched",
        "cublasDgeqrfBatched", "cublasCgeqrfBatched", "cublasZgeqrfBatched",
        /*Legacy API*/
        "cublasInit", "cublasShutdown", "cublasGetError",
        /*level 1*/
        "cublasSnrm2", "cublasDnrm2", "cublasScnrm2", "cublasDznrm2",
        "cublasSdot", "cublasDdot", "cublasCdotu", "cublasCdotc", "cublasZdotu",
        "cublasZdotc", "cublasSscal", "cublasDscal", "cublasCscal",
        "cublasZscal", "cublasCsscal", "cublasZdscal", "cublasSaxpy",
        "cublasDaxpy", "cublasCaxpy", "cublasZaxpy", "cublasScopy",
        "cublasDcopy", "cublasCcopy", "cublasZcopy", "cublasSswap",
        "cublasDswap", "cublasCswap", "cublasZswap", "cublasIsamax",
        "cublasIdamax", "cublasIcamax", "cublasIzamax", "cublasIsamin",
        "cublasIdamin", "cublasIcamin", "cublasIzamin", "cublasSasum",
        "cublasDasum", "cublasScasum", "cublasDzasum", "cublasSrot",
        "cublasDrot", "cublasCsrot", "cublasZdrot", "cublasSrotg",
        "cublasDrotg", "cublasSrotm", "cublasDrotm", "cublasSrotmg",
        "cublasDrotmg",
        /*level 2*/
        "cublasSgemv", "cublasDgemv", "cublasCgemv", "cublasZgemv",
        "cublasSgbmv", "cublasDgbmv", "cublasCgbmv", "cublasZgbmv",
        "cublasStrmv", "cublasDtrmv", "cublasCtrmv", "cublasZtrmv",
        "cublasStbmv", "cublasDtbmv", "cublasCtbmv", "cublasZtbmv",
        "cublasStpmv", "cublasDtpmv", "cublasCtpmv", "cublasZtpmv",
        "cublasStrsv", "cublasDtrsv", "cublasCtrsv", "cublasZtrsv",
        "cublasStpsv", "cublasDtpsv", "cublasCtpsv", "cublasZtpsv",
        "cublasStbsv", "cublasDtbsv", "cublasCtbsv", "cublasZtbsv",
        "cublasSsymv", "cublasDsymv", "cublasChemv", "cublasZhemv",
        "cublasSsbmv", "cublasDsbmv", "cublasChbmv", "cublasZhbmv",
        "cublasSspmv", "cublasDspmv", "cublasChpmv", "cublasZhpmv",
        "cublasSger", "cublasDger", "cublasCgeru", "cublasCgerc", "cublasZgeru",
        "cublasZgerc", "cublasSsyr", "cublasDsyr", "cublasCher", "cublasZher",
        "cublasSspr", "cublasDspr", "cublasChpr", "cublasZhpr", "cublasSsyr2",
        "cublasDsyr2", "cublasCher2", "cublasZher2", "cublasSspr2",
        "cublasDspr2", "cublasChpr2", "cublasZhpr2",
        /*level 3*/
        "cublasSgemm", "cublasDgemm", "cublasCgemm", "cublasZgemm",
        "cublasSsyrk", "cublasDsyrk", "cublasCsyrk", "cublasZsyrk",
        "cublasCherk", "cublasZherk", "cublasSsyr2k", "cublasDsyr2k",
        "cublasCsyr2k", "cublasZsyr2k", "cublasCher2k", "cublasZher2k",
        "cublasSsymm", "cublasDsymm", "cublasCsymm", "cublasZsymm",
        "cublasChemm", "cublasZhemm", "cublasStrsm", "cublasDtrsm",
        "cublasCtrsm", "cublasZtrsm", "cublasStrmm", "cublasDtrmm",
        "cublasCtrmm", "cublasZtrmm");
  };

  MF.addMatcher(callExpr(allOf(callee(functionDecl(functionName())),
                               hasAncestor(functionDecl(
                                   anyOf(hasAttr(attr::CUDADevice),
                                         hasAttr(attr::CUDAGlobal))))))
                    .bind("kernelCall"),
                this);

  MF.addMatcher(
      callExpr(
          allOf(callee(functionDecl(functionName())), parentStmt(),
                hasAncestor(functionDecl(unless(allOf(
                    hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))))))
          .bind("FunctionCall"),
      this);
  MF.addMatcher(
      callExpr(
          allOf(callee(functionDecl(functionName())), unless(parentStmt()),
                unless(hasAncestor(varDecl())),
                hasAncestor(functionDecl(unless(allOf(
                    hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))))))
          .bind("FunctionCallUsedNotInitializeVarDecl"),
      this);

  MF.addMatcher(
      callExpr(
          allOf(callee(functionDecl(functionName())), hasAncestor(varDecl()),
                hasAncestor(functionDecl(unless(allOf(
                    hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))))))
          .bind("FunctionCallUsedInitializeVarDecl"),
      this);
}

void BLASFunctionCallRule::run(const MatchFinder::MatchResult &Result) {
  bool IsAssigned = false;
  bool IsInitializeVarDecl = false;
  bool HasDeviceAttr = false;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "kernelCall");
  if (CE) {
    HasDeviceAttr = true;
  } else if (!(CE = getNodeAsType<CallExpr>(Result, "FunctionCall"))) {
    if ((CE = getNodeAsType<CallExpr>(
             Result, "FunctionCallUsedNotInitializeVarDecl"))) {
      IsAssigned = true;
    } else if ((CE = getNodeAsType<CallExpr>(
                    Result, "FunctionCallUsedInitializeVarDecl"))) {
      IsAssigned = true;
      IsInitializeVarDecl = true;
    } else {
      return;
    }
  }

  assert(CE && "Unknown result");

  if (!CE->getDirectCallee())
    return;
  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();

  const SourceManager *SM = Result.SourceManager;
  SourceLocation FuncNameBegin(CE->getBeginLoc());
  SourceLocation FuncCallEnd(CE->getEndLoc());
  // There are some macroes like "#define cublasSgemm cublasSgemm_v2"
  // in "cublas_v2.h", so the function names we match should have the
  // suffix "_v2".
  if (FuncNameBegin.isMacroID())
    FuncNameBegin = SM->getExpansionLoc(FuncNameBegin);
  if (FuncCallEnd.isMacroID())
    FuncCallEnd = SM->getExpansionLoc(FuncCallEnd);
  Token Tok;
  Lexer::getRawToken(FuncNameBegin, Tok, *SM, LangOptions());
  SourceLocation FuncNameEnd = Tok.getEndLoc();
  auto FuncNameLength =
      SM->getCharacterData(FuncNameEnd) - SM->getCharacterData(FuncNameBegin);
  auto SR = getScopeInsertRange(CE, FuncNameBegin, FuncCallEnd);
  SourceLocation StmtBegin = SR.getBegin(), StmtEndAfterSemi = SR.getEnd();
  std::string IndentStr = getIndent(StmtBegin, *SM).str();
  std::string PrefixInsertStr, SuffixInsertStr;
  // TODO: Need to process the situation when scalar pointers (alpha, beta)
  // are device pointers.
  if (MapNames::BLASFuncReplInfoMap.find(FuncName) !=
      MapNames::BLASFuncReplInfoMap.end()) {
    auto ReplInfoPair = MapNames::BLASFuncReplInfoMap.find(FuncName);
    MapNames::BLASFuncReplInfo ReplInfo = ReplInfoPair->second;
    std::string Replacement = ReplInfo.ReplName;
    if (HasDeviceAttr) {
      report(FuncNameBegin, Diagnostics::FUNCTION_CALL_IN_DEVICE, FuncName,
             Replacement);
      return;
    }
    int ArgNum = CE->getNumArgs();
    // TODO: If the memory is not allocated by cudaMalloc(), the migrated
    // program will abort at
    // syclct::memory_manager::get_instance().translate_ptr();
    for (int i = 0; i < ArgNum; ++i) {
      int IndexTemp = -1;
      if (isReplIndex(i, ReplInfo.BufferIndexInfo, IndexTemp)) {
        std::string BufferDecl = "";
        std::string BufferName = "";
        if (FuncName == "cublasStrmm_v2" || FuncName == "cublasDtrmm_v2") {
          processTrmmParams(CE, PrefixInsertStr, BufferName, BufferDecl,
                            IndexTemp, i, IndentStr, ReplInfo.BufferTypeInfo,
                            StmtBegin);
        } else {
          BufferName = getBufferNameAndDeclStr(
              CE->getArg(i), *(Result.Context),
              ReplInfo.BufferTypeInfo[IndexTemp], StmtBegin, BufferDecl, i);
        }

        PrefixInsertStr = PrefixInsertStr + BufferDecl;

        if (ReplInfo.BufferTypeInfo[IndexTemp] == "int") {
          PrefixInsertStr = PrefixInsertStr + IndentStr +
                            "cl::sycl::buffer<int64_t,1> "
                            "result_temp_buffer(cl::sycl::range<1>(1));" +
                            getNL();
          SuffixInsertStr = SuffixInsertStr + BufferName +
                            ".get_access<cl::sycl::access::"
                            "mode::write>()[0] = "
                            "(int)result_temp_buffer.get_access<cl::sycl::"
                            "access::mode::read>()[0];" +
                            getNL() + IndentStr;
          emplaceTransformation(
              new ReplaceStmt(CE->getArg(i), "result_temp_buffer"));
          continue;
        }

        emplaceTransformation(
            new ReplaceStmt(CE->getArg(i), std::move(BufferName)));
      }
      if (isReplIndex(i, ReplInfo.PointerIndexInfo, IndexTemp)) {
        emplaceTransformation(new ReplaceStmt(
            CE->getArg(i),
            "*(" + getStmtSpelling(CE->getArg(i), *(Result.Context)) + ")"));
      }
      const CStyleCastExpr *CSCE = nullptr;
      if ((CSCE = dyn_cast<CStyleCastExpr>(CE->getArg(i)))) {
        processParamIntCastToBLASEnum(CE->getArg(i), CSCE, *(Result.Context), i,
                                      IndentStr, ReplInfo.OperationIndexInfo,
                                      ReplInfo.FillModeIndexInfo,
                                      PrefixInsertStr);
      } else if ((FuncName == "cublasSsyrkx" || FuncName == "cublasDsyrkx") &&
                 isReplIndex(i, ReplInfo.OperationIndexInfo, IndexTemp)) {
        std::string TransparamName = "transpose_ct" + std::to_string(i);
        std::string TransStr =
            getStmtSpelling(CE->getArg(i), *(Result.Context));

        auto TransPair = MapNames::BLASEnumsMap.find(TransStr);
        if (TransPair != MapNames::BLASEnumsMap.end()) {
          TransStr = TransPair->second;
        }
        PrefixInsertStr = PrefixInsertStr + IndentStr + "auto " +
                          TransparamName + " = " + TransStr + ";" + getNL();
        Optional<Token> TokSharedPtr;
        TokSharedPtr = Lexer::findNextToken(
            CE->getArg(i)->getEndLoc(), *(Result.SourceManager), LangOptions());
        Token CommaTok = TokSharedPtr.getValue();
        auto CommaEnd = CommaTok.getEndLoc();
        auto Len = SM->getCharacterData(CommaEnd) -
                   SM->getCharacterData(CE->getArg(i)->getBeginLoc());
        emplaceTransformation(new ReplaceText(CE->getArg(i)->getBeginLoc(), Len,
                                              TransparamName + ","));
      }
    }

    if (FuncName == "cublasSsyrkx" || FuncName == "cublasDsyrkx") {
      SourceLocation InsertSL = CE->getArg(3)->getBeginLoc();
      if (InsertSL.isMacroID())
        InsertSL = SM->getExpansionLoc(InsertSL);
      const CStyleCastExpr *CSCE = nullptr;
      if ((CSCE = dyn_cast<CStyleCastExpr>(CE->getArg(2)))) {
        emplaceTransformation(new InsertText(
            InsertSL, "((((int)transpose_ct2)==0)?(mkl::transpose::trans):("
                      "mkl::transpose::nontrans)), "));
      } else {
        emplaceTransformation(new InsertText(
            InsertSL, "((transpose_ct2)==(mkl::transpose::nontrans))?(mkl::"
                      "transpose::trans):(mkl::transpose::nontrans), "));
      }
    }
    if (FuncName == "cublasStrmm_v2" || FuncName == "cublasDtrmm_v2") {
      processTrmmCall(CE, PrefixInsertStr, IndentStr);
    }
    if (IsAssigned) {
      insertAroundRange(FuncNameBegin, FuncCallEnd.getLocWithOffset(1), "(",
                        ", 0)");
      report(FuncNameBegin, Diagnostics::NOERROR_RETURN_COMMA_OP);
    }
    emplaceTransformation(
        new ReplaceText(FuncNameBegin, FuncNameLength, std::move(Replacement)));
    insertAroundRange(StmtBegin, StmtEndAfterSemi,
                      std::string("{") + getNL() + PrefixInsertStr + IndentStr,
                      getNL() + IndentStr + SuffixInsertStr + std::string("}"));
  } else if (MapNames::BLASFuncComplexReplInfoMap.find(FuncName) !=
             MapNames::BLASFuncComplexReplInfoMap.end()) {
    auto ReplInfoPair = MapNames::BLASFuncComplexReplInfoMap.find(FuncName);
    MapNames::BLASFuncComplexReplInfo ReplInfo = ReplInfoPair->second;
    std::string Replacement = ReplInfo.ReplName;
    if (HasDeviceAttr) {
      report(FuncNameBegin, Diagnostics::FUNCTION_CALL_IN_DEVICE, FuncName,
             Replacement);
      return;
    }
    int ArgNum = CE->getNumArgs();
    // TODO: If the memory is not allocated by cudaMalloc(), the migrated
    // program will abort at
    // syclct::memory_manager::get_instance().translate_ptr();

    for (int i = 0; i < ArgNum; ++i) {
      int IndexTemp = -1;
      if (isReplIndex(i, ReplInfo.BufferIndexInfo, IndexTemp)) {
        std::string BufferDecl = "";
        std::string BufferName = "";
        if (FuncName == "cublasCtrmm_v2" || FuncName == "cublasZtrmm_v2") {
          processTrmmParams(CE, PrefixInsertStr, BufferName, BufferDecl,
                            IndexTemp, i, IndentStr, ReplInfo.BufferTypeInfo,
                            StmtBegin);
        } else {
          BufferName = getBufferNameAndDeclStr(
              CE->getArg(i), *(Result.Context),
              ReplInfo.BufferTypeInfo[IndexTemp], StmtBegin, BufferDecl, i);
        }
        PrefixInsertStr = PrefixInsertStr + BufferDecl;

        if (ReplInfo.BufferTypeInfo[IndexTemp] == "int") {
          PrefixInsertStr = PrefixInsertStr + IndentStr +
                            "cl::sycl::buffer<int64_t,1> "
                            "result_temp_buffer(cl::sycl::range<1>(1));" +
                            getNL();
          SuffixInsertStr = SuffixInsertStr + IndentStr + BufferName +
                            ".get_access<cl::sycl::access::"
                            "mode::write>()[0] = "
                            "(int)result_temp_buffer.get_access<cl::sycl::"
                            "access::mode::read>()[0];" +
                            getNL();
          emplaceTransformation(
              new ReplaceStmt(CE->getArg(i), "result_temp_buffer"));
          continue;
        }
        emplaceTransformation(
            new ReplaceStmt(CE->getArg(i), std::move(BufferName)));
      }
      IndexTemp = -1;
      if (isReplIndex(i, ReplInfo.PointerIndexInfo, IndexTemp)) {
        if (ReplInfo.PointerTypeInfo[IndexTemp] == "float" ||
            ReplInfo.PointerTypeInfo[IndexTemp] == "double") {
          emplaceTransformation(new ReplaceStmt(
              CE->getArg(i),
              "*(" + getStmtSpelling(CE->getArg(i), *(Result.Context)) + ")"));
        } else {
          emplaceTransformation(new ReplaceStmt(
              CE->getArg(i),
              ReplInfo.PointerTypeInfo[IndexTemp] + "((" +
                  getStmtSpelling(CE->getArg(i), *(Result.Context)) +
                  ")->x(),(" +
                  getStmtSpelling(CE->getArg(i), *(Result.Context)) +
                  ")->y())"));
        }
      }
      const CStyleCastExpr *CSCE = nullptr;
      if ((CSCE = dyn_cast<CStyleCastExpr>(CE->getArg(i)))) {
        processParamIntCastToBLASEnum(CE->getArg(i), CSCE, *(Result.Context), i,
                                      IndentStr, ReplInfo.OperationIndexInfo,
                                      ReplInfo.FillModeIndexInfo,
                                      PrefixInsertStr);
      }
    }
    if (FuncName == "cublasCtrmm_v2" || FuncName == "cublasZtrmm_v2") {
      processTrmmCall(CE, PrefixInsertStr, IndentStr);
    }
    if (IsAssigned) {
      insertAroundRange(FuncNameBegin, FuncCallEnd.getLocWithOffset(1), "(",
                        ", 0)");
      report(FuncNameBegin, Diagnostics::NOERROR_RETURN_COMMA_OP);
    }
    emplaceTransformation(
        new ReplaceText(FuncNameBegin, FuncNameLength, std::move(Replacement)));
    insertAroundRange(StmtBegin, StmtEndAfterSemi,
                      std::string("{") + getNL() + PrefixInsertStr + IndentStr,
                      getNL() + SuffixInsertStr + IndentStr + std::string("}"));
  } else if (MapNames::LegacyBLASFuncReplInfoMap.find(FuncName) !=
             MapNames::LegacyBLASFuncReplInfoMap.end()) {
    auto ReplInfoPair = MapNames::LegacyBLASFuncReplInfoMap.find(FuncName);
    MapNames::BLASFuncComplexReplInfo ReplInfo = ReplInfoPair->second;
    std::string CallExprReplStr = "";
    PrefixInsertStr = std::string("{") + getNL();
    CallExprReplStr =
        CallExprReplStr + ReplInfo.ReplName + "(syclct::get_default_queue()";
    std::string IndentStr =
        getIndent(StmtBegin, (Result.Context)->getSourceManager()).str();

    std::string VarType;
    std::string VarName;
    const VarDecl *VD = 0;
    if (IsInitializeVarDecl) {
      VD = getAncestralVarDecl(CE);
      if (VD) {
        VarType = VD->getType().getAsString();
        if (VarType == "cuComplex") {
          VarType = "cl::sycl::float2";
        }
        if (VarType == "cuDoubleComplex") {
          VarType = "cl::sycl::double2";
        }
        VarName = VD->getNameAsString();
      } else {
        assert(0 && "Fail to get VarDecl information");
        return;
      }
      PrefixInsertStr =
          VarType + " " + VarName + ";" + getNL() + IndentStr + PrefixInsertStr;
    }
    std::vector<std::string> ParamsStrsVec =
        getParamsAsStrs(CE, *(Result.Context));
    int ArgNum = CE->getNumArgs();
    for (int i = 0; i < ArgNum; ++i) {
      int IndexTemp = -1;
      if (isReplIndex(i, ReplInfo.BufferIndexInfo, IndexTemp)) {
        std::string BufferDecl;
        std::string BufferName = getBufferNameAndDeclStr(
            CE->getArg(i), *(Result.Context),
            ReplInfo.BufferTypeInfo[IndexTemp], StmtBegin, BufferDecl, i);
        CallExprReplStr = CallExprReplStr + ", " + BufferName;
        PrefixInsertStr = PrefixInsertStr + BufferDecl;
      } else if (isReplIndex(i, ReplInfo.PointerIndexInfo, IndexTemp)) {
        if (ReplInfo.PointerTypeInfo[IndexTemp] == "float" ||
            ReplInfo.PointerTypeInfo[IndexTemp] == "double") {
          CallExprReplStr = CallExprReplStr + ", *(" + ParamsStrsVec[i] + ")";
        } else {
          CallExprReplStr =
              CallExprReplStr + ", " + ReplInfo.PointerTypeInfo[IndexTemp] +
              "((" + ParamsStrsVec[i] + ").x(),(" + ParamsStrsVec[i] + ").y())";
        }
      } else if (isReplIndex(i, ReplInfo.OperationIndexInfo, IndexTemp)) {
        std::string TransParamName = "transpose_ct" + std::to_string(i);
        PrefixInsertStr = PrefixInsertStr + IndentStr + "auto " +
                          TransParamName + " = " + ParamsStrsVec[i] + ";" +
                          getNL();
        CallExprReplStr = CallExprReplStr + ", " + "(((" + TransParamName +
                          ")=='N'||(" + TransParamName +
                          ")=='n')?(mkl::transpose::"
                          "nontrans):(((" +
                          TransParamName + ")=='T'||(" + TransParamName +
                          ")=='t')?(mkl::transpose:"
                          ":nontrans):(mkl::transpose::conjtrans)))";
      } else if (ReplInfo.FillModeIndexInfo == i) {
        std::string FillParamName = "fillmode_ct" + std::to_string(i);
        PrefixInsertStr = PrefixInsertStr + IndentStr + "auto " +
                          FillParamName + " = " + ParamsStrsVec[i] + ";" +
                          getNL();
        CallExprReplStr = CallExprReplStr + ", " + "(((" + FillParamName +
                          ")=='L'||(" + FillParamName +
                          ")=='l')?(mkl::uplo::lower):(mkl::uplo::upper))";
      } else if (ReplInfo.SideModeIndexInfo == i) {
        std::string SideParamName = "sidemode_ct" + std::to_string(i);
        PrefixInsertStr = PrefixInsertStr + IndentStr + "auto " +
                          SideParamName + " = " + ParamsStrsVec[i] + ";" +
                          getNL();
        CallExprReplStr = CallExprReplStr + ", " + "(((" + SideParamName +
                          ")=='L'||(" + SideParamName +
                          ")=='l')?(mkl::side::left):(mkl::side::right))";
      } else if (ReplInfo.DiagTypeIndexInfo == i) {
        std::string DiagParamName = "diagtype_ct" + std::to_string(i);
        PrefixInsertStr = PrefixInsertStr + IndentStr + "auto " +
                          DiagParamName + " = " + ParamsStrsVec[i] + ";" +
                          getNL();
        CallExprReplStr = CallExprReplStr + ", " + "(((" + DiagParamName +
                          ")=='N'||(" + DiagParamName +
                          ")=='n')?(mkl::diag::nonunit):(mkl::diag::unit))";
      } else {
        CallExprReplStr = CallExprReplStr + ", " + ParamsStrsVec[i];
      }
    }

    // TODO: If the memory is not allocated by cudaMalloc(), the migrated
    // program will abort at
    // syclct::memory_manager::get_instance().translate_ptr();
    if (FuncName == "cublasSnrm2" || FuncName == "cublasDnrm2" ||
        FuncName == "cublasScnrm2" || FuncName == "cublasDznrm2" ||
        FuncName == "cublasSdot" || FuncName == "cublasDdot" ||
        FuncName == "cublasCdotu" || FuncName == "cublasCdotc" ||
        FuncName == "cublasZdotu" || FuncName == "cublasZdotc" ||
        FuncName == "cublasIsamax" || FuncName == "cublasIdamax" ||
        FuncName == "cublasIcamax" || FuncName == "cublasIzamax" ||
        FuncName == "cublasIsamin" || FuncName == "cublasIdamin" ||
        FuncName == "cublasIcamin" || FuncName == "cublasIzamin" ||
        FuncName == "cublasSasum" || FuncName == "cublasDasum" ||
        FuncName == "cublasScasum" || FuncName == "cublasDzasum") {
      // APIs which have return value
      std::string ResultType =
          ReplInfo.BufferTypeInfo[ReplInfo.BufferTypeInfo.size() - 1];
      PrefixInsertStr =
          PrefixInsertStr + IndentStr + "cl::sycl::buffer<" + ResultType +
          ",1> result_temp_buffer(cl::sycl::range<1>(1));" + getNL() +
          IndentStr + CallExprReplStr + ", result_temp_buffer);" + getNL();
      SuffixInsertStr = getNL() + IndentStr + "}" + getNL();
      insertAroundRange(StmtBegin, StmtEndAfterSemi,
                        PrefixInsertStr + IndentStr,
                        std::move(SuffixInsertStr));
      std::string ReturnValueParamsStr =
          "(result_temp_buffer.get_access<cl::sycl::"
          "access::mode::read>()[0].real(), "
          "result_temp_buffer.get_access<cl::sycl::"
          "access::mode::read>()[0].imag());";
      if (IsInitializeVarDecl) {
        auto ParentNodes = (Result.Context)->getParents(*VD);
        const DeclStmt *DS = 0;
        if ((DS = ParentNodes[0].get<DeclStmt>())) {
          if (FuncName == "cublasCdotu" || FuncName == "cublasCdotc") {
            emplaceTransformation(new ReplaceStmt(
                DS, VarName + " = cl::sycl::float2" + ReturnValueParamsStr));
          } else if (FuncName == "cublasZdotu" || FuncName == "cublasZdotc") {
            emplaceTransformation(new ReplaceStmt(
                DS, VarName + " = cl::sycl::double2" + ReturnValueParamsStr));
          } else {
            emplaceTransformation(new ReplaceStmt(
                DS, VarName + " = result_temp_buffer.get_access<cl::sycl::"
                              "access::mode::read>()[0];"));
          }
        } else {
          assert(0 && "Fail to get Var Decl Stmt");
          return;
        }
      } else {
        if (FuncName == "cublasCdotu" || FuncName == "cublasCdotc") {
          emplaceTransformation(
              new ReplaceStmt(CE, "cl::sycl::float2" + ReturnValueParamsStr));
        } else if (FuncName == "cublasZdotu" || FuncName == "cublasZdotc") {
          emplaceTransformation(
              new ReplaceStmt(CE, "cl::sycl::double2" + ReturnValueParamsStr));
        } else {
          emplaceTransformation(
              new ReplaceStmt(CE, "result_temp_buffer.get_access<cl::sycl::"
                                  "access::mode::read>()[0]"));
        }
      }
    } else {
      // APIs which haven't return value
      // PrefixInsertStr = getNL() + IndentStr + PrefixInsertStr;
      CallExprReplStr = CallExprReplStr + ")";
      emplaceTransformation(new ReplaceStmt(CE, std::move(CallExprReplStr)));
      insertAroundRange(StmtBegin, StmtEndAfterSemi,
                        PrefixInsertStr + IndentStr,
                        getNL() + IndentStr + std::string("}"));
    }
  } else if (MapNames::BLASFuncWrapperReplInfoMap.find(FuncName) !=
             MapNames::BLASFuncWrapperReplInfoMap.end()) {
    auto ReplInfoPair = MapNames::BLASFuncWrapperReplInfoMap.find(FuncName);
    MapNames::BLASFuncReplInfo ReplInfo = ReplInfoPair->second;
    std::string Replacement = ReplInfo.ReplName;
    if (HasDeviceAttr) {
      report(FuncNameBegin, Diagnostics::FUNCTION_CALL_IN_DEVICE, FuncName,
             Replacement);
      return;
    }
    if (IsAssigned) {
      insertAroundRange(FuncNameBegin, FuncCallEnd.getLocWithOffset(1), "(",
                        ", 0)");
      report(FuncNameBegin, Diagnostics::NOERROR_RETURN_COMMA_OP);
    }
    int ArgNum = CE->getNumArgs();
    for (int i = 0; i < ArgNum; ++i) {
      const CStyleCastExpr *CSCE = nullptr;
      if ((CSCE = dyn_cast<CStyleCastExpr>(CE->getArg(i)))) {
        processParamIntCastToBLASEnum(CE->getArg(i), CSCE, *(Result.Context), i,
                                      IndentStr, ReplInfo.OperationIndexInfo,
                                      ReplInfo.FillModeIndexInfo,
                                      PrefixInsertStr);
      }
    }
    emplaceTransformation(
        new ReplaceText(FuncNameBegin, FuncNameLength, std::move(Replacement)));
    if (PrefixInsertStr != "") {
      insertAroundRange(
          StmtBegin, StmtEndAfterSemi,
          std::string("{") + getNL() + PrefixInsertStr + IndentStr,
          getNL() + IndentStr + SuffixInsertStr + std::string("}"));
    }
  } else if (FuncName == "cublasCreate_v2" || FuncName == "cublasDestroy_v2") {
    if (IsAssigned) {
      emplaceTransformation(
          new ReplaceStmt(CE, /*IsReplaceCompatibilityAPI*/ false, FuncName,
                          /*IsProcessMacro*/ true, "0"));
    } else {
      emplaceTransformation(
          new ReplaceStmt(CE, /*IsReplaceCompatibilityAPI*/ false, FuncName,
                          /*IsProcessMacro*/ true, ""));
    }
  } else if (FuncName == "cublasInit" || FuncName == "cublasShutdown" ||
             FuncName == "cublasGetError") {
    // Remove these three function calls.
    // TODO: migrate functions when they are in template
    if (IsAssigned) {
      emplaceTransformation(
          new ReplaceStmt(CE, /*IsReplaceCompatibilityAPI*/ false, FuncName,
                          /*IsProcessMacro*/ false, "0"));
    } else {
      emplaceTransformation(
          new ReplaceStmt(CE, /*IsReplaceCompatibilityAPI*/ false, FuncName,
                          /*IsProcessMacro*/ false, ""));
    }
  } else if (FuncName == "cublasSetVector" || FuncName == "cublasGetVector" ||
             FuncName == "cublasSetVectorAsync" ||
             FuncName == "cublasGetVectorAsync") {
    if (HasDeviceAttr) {
      report(CE->getBeginLoc(), Diagnostics::FUNCTION_CALL_IN_DEVICE, FuncName,
             "syclct::dpct_memcpy");
      return;
    }
    // The 4th and 6th param (incx and incy) of cublasSetVector/cublasGetVector
    // specify the space between two consequent elements when stored.
    // We migrate the original code when incx and incy both equal to 1 (all
    // elements are stored consequently).
    // Otherwise, the codes are kept originally.
    std::vector<std::string> ParamsStrsVec =
        getParamsAsStrs(CE, *(Result.Context));
    // CopySize equals to n*elemSize*incx
    // incx(incy) equals to 1 means elements of x(y) are stored continuously
    std::string CopySize = "(" + ParamsStrsVec[0] + ")*(" + ParamsStrsVec[1] +
                           ")*(" + ParamsStrsVec[3] + ")";
    std::string XStr = "(void*)(" + ParamsStrsVec[2] + ")";
    std::string YStr = "(void*)(" + ParamsStrsVec[4] + ")";
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
               FuncName,
               "parameter " + ParamsStrsVec[3] +
                   " does not equal to parameter " + ParamsStrsVec[5]);
        return;
      }
      if ((IncxStr == IncyStr) && (IncxStr != "1")) {
        // incx equals to incy, but does not equal to 1. Performance issue may
        // occur.
        report(CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMACE_ISSUE,
               FuncName,
               "parameter " + ParamsStrsVec[3] + " equals to parameter " +
                   ParamsStrsVec[5] + " but greater than 1");
      }
    } else {
      // Keep original code, give a comment to let user migrate code manually
      report(CE->getBeginLoc(), Diagnostics::NOT_SUPPORTED_PARAMETERS_VALUE,
             FuncName,
             "parameter(s) " + ParamsStrsVec[3] + " and/or " +
                 ParamsStrsVec[5] + " could not be evaluated");
      return;
    }

    std::string Replacement =
        "syclct::dpct_memcpy(" + YStr + "," + XStr + "," + CopySize + ",";

    if (FuncName == "cublasGetVector" || FuncName == "cublasGetVectorAsync") {
      Replacement = Replacement + "syclct::device_to_host)";
      emplaceTransformation(new ReplaceStmt(CE, std::move(Replacement)));
    }
    if (FuncName == "cublasSetVector" || FuncName == "cublasSetVectorAsync") {
      Replacement = Replacement + "syclct::host_to_device)";
      emplaceTransformation(new ReplaceStmt(CE, std::move(Replacement)));
    }

    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
      insertAroundStmt(CE, "(", ", 0)");
    }
  } else if (FuncName == "cublasSetMatrix" || FuncName == "cublasGetMatrix" ||
             FuncName == "cublasSetMatrixAsync" ||
             FuncName == "cublasGetMatrixAsync") {
    if (HasDeviceAttr) {
      report(CE->getBeginLoc(), Diagnostics::FUNCTION_CALL_IN_DEVICE, FuncName,
             "syclct::dpct_memcpy");
      return;
    }
    std::vector<std::string> ParamsStrsVec =
        getParamsAsStrs(CE, *(Result.Context));
    // CopySize equals to lda*cols*elemSize
    std::string CopySize = "(" + ParamsStrsVec[4] + ")*(" + ParamsStrsVec[1] +
                           ")*(" + ParamsStrsVec[2] + ")";
    std::string AStr = "(void*)(" + ParamsStrsVec[3] + ")";
    std::string BStr = "(void*)(" + ParamsStrsVec[5] + ")";

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
               FuncName,
               "parameter " + ParamsStrsVec[4] +
                   " does not equal to parameter " + ParamsStrsVec[6]);
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
                 FuncName,
                 "parameter " + ParamsStrsVec[0] +
                     " is smaller than parameter " + ParamsStrsVec[4]);
        }
      } else {
        // rows cannot be evaluated. Performance issue may occur.
        report(
            CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMACE_ISSUE,
            FuncName,
            "parameter " + ParamsStrsVec[0] +
                " could not be evaluated and may be smaller than parameter " +
                ParamsStrsVec[4]);
      }
    } else {
      // Keep original code, give a comment to let user migrate code manually
      report(CE->getBeginLoc(), Diagnostics::NOT_SUPPORTED_PARAMETERS_VALUE,
             FuncName,
             "parameter(s) " + ParamsStrsVec[4] + " and/or " +
                 ParamsStrsVec[6] + " could not be evaluated");
      return;
    }

    std::string Replacement =
        "syclct::dpct_memcpy(" + BStr + "," + AStr + "," + CopySize + ",";

    if (FuncName == "cublasGetMatrix" || FuncName == "cublasGetMatrixAsync") {
      Replacement = Replacement + "syclct::device_to_host)";
      emplaceTransformation(new ReplaceStmt(CE, std::move(Replacement)));
    }
    if (FuncName == "cublasSetMatrix" || FuncName == "cublasSetMatrixAsync") {
      Replacement = Replacement + "syclct::host_to_device)";
      emplaceTransformation(new ReplaceStmt(CE, std::move(Replacement)));
    }
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
      insertAroundStmt(CE, "(", ", 0)");
    }
  } else if (FuncName == "make_cuComplex" ||
             FuncName == "make_cuDoubleComplex") {
    if (FuncName == "make_cuDoubleComplex")
      emplaceTransformation(
          new ReplaceCalleeName(CE, "cl::sycl::double2", FuncName));
    else
      emplaceTransformation(
          new ReplaceCalleeName(CE, "cl::sycl::float2", FuncName));
  } else {
    assert(0 && "Unknown function name");
  }
}

bool BLASFunctionCallRule::isReplIndex(int Input,
                                       const std::vector<int> &IndexInfo,
                                       int &IndexTemp) {
  for (int i = 0; i < static_cast<int>(IndexInfo.size()); ++i) {
    if (IndexInfo[i] == Input) {
      IndexTemp = i;
      return true;
    }
  }
  return false;
}

std::string BLASFunctionCallRule::getBufferNameAndDeclStr(
    const Expr *Arg, const ASTContext &AC, const std::string &TypeAsStr,
    SourceLocation SL, std::string &BufferDecl, int DistinctionID) {
  std::string PointerName = getStmtSpelling(Arg, AC);
  return getBufferNameAndDeclStr(PointerName, AC, TypeAsStr, SL, BufferDecl,
                                 DistinctionID);
}

std::string BLASFunctionCallRule::getBufferNameAndDeclStr(
    const std::string &PointerName, const ASTContext &AC,
    const std::string &TypeAsStr, SourceLocation SL, std::string &BufferDecl,
    int DistinctionID) {
  std::string BufferTempName = "buffer_ct" + std::to_string(DistinctionID);
  std::string AllocationTempName =
      "allocation_ct" + std::to_string(DistinctionID);
  // TODO: reinterpret will copy more data
  BufferDecl = getIndent(SL, AC.getSourceManager()).str() + "auto " +
               AllocationTempName +
               " = syclct::memory_manager::get_instance().translate_ptr(" +
               PointerName + ");" + getNL() +
               getIndent(SL, AC.getSourceManager()).str() +
               "cl::sycl::buffer<" + TypeAsStr + ",1> " + BufferTempName +
               " = " + AllocationTempName + ".buffer.reinterpret<" + TypeAsStr +
               ", 1>(cl::sycl::range<1>(" + AllocationTempName +
               ".size/sizeof(" + TypeAsStr + ")));" + getNL();
  return BufferTempName;
}

std::vector<std::string>
BLASFunctionCallRule::getParamsAsStrs(const CallExpr *CE,
                                      const ASTContext &Context) {
  std::vector<std::string> ParamsStrVec;
  for (auto Arg : CE->arguments())
    ParamsStrVec.emplace_back(getStmtSpelling(Arg, Context));
  return ParamsStrVec;
}

// orignal cuda code looks like:
//   cuComplex res1 = cublasLegacyAPI(...);
//   res2 = cublasLegacyAPI(...);
//
// migrated code looks like:
//   cuComplex res1;
//   {
//   buffer res_buffer;
//   mklAPI(res_buffer);
//   res1 = res_buffer.get_access()[0];
//   }
//   {
//   buffer res_buffer;
//   mklAPI(res_buffer);
//   res2 = res_buffer.get_access()[0];
//   }
//
// If the API return value initializes the var declaration, we need to put the
// var declaration out of the scope and assign it in the scope, otherwise users
// cannot use this var out of the scope.
// So we need to find the Decl node of the var, which is the CallExpr node's
// ancestor.
// The initial node is the matched CallExpr node. Then visit the parent node of
// the current node until the current node is a VarDecl node.
const clang::VarDecl *
BLASFunctionCallRule::getAncestralVarDecl(const clang::CallExpr *CE) {
  auto &Context = syclct::SyclctGlobalInfo::getContext();
  auto Parents = Context.getParents(*CE);
  while (Parents.size() == 1) {
    auto *Parent = Parents[0].get<VarDecl>();
    if (Parent) {
      return Parent;
    } else {
      Parents = Context.getParents(Parents[0]);
    }
  }
  return nullptr;
}

void BLASFunctionCallRule::processParamIntCastToBLASEnum(
    const Expr *E, const CStyleCastExpr *CSCE, const ASTContext &Context,
    const int DistinctionID, const std::string IndentStr,
    const std::vector<int> &OperationIndexInfo, const int FillModeIndexInfo,
    std::string &PrefixInsertStr) {
  auto &SM = SyclctGlobalInfo::getSourceManager();
  const Expr *SubExpr = CSCE->getSubExpr();
  std::string SubExprStr = getStmtSpelling(SubExpr, Context);
  SourceLocation BeginLoc = E->getBeginLoc();
  SourceLocation EndLoc = E->getEndLoc();
  if (E->getBeginLoc().isMacroID()) {
    BeginLoc = SM.getExpansionLoc(BeginLoc);
    EndLoc = SM.getExpansionLoc(EndLoc);
  }
  auto Len = SM.getDecomposedLoc(EndLoc).second -
             SM.getDecomposedLoc(BeginLoc).second +
             Lexer::MeasureTokenLength(EndLoc, SM, Context.getLangOpts());

  int IndexTemp = -1;
  if (isReplIndex(DistinctionID, OperationIndexInfo, IndexTemp)) {
    std::string TransParamName = "transpose_ct" + std::to_string(DistinctionID);
    PrefixInsertStr = PrefixInsertStr + IndentStr + "auto " + TransParamName +
                      " = " + SubExprStr + ";" + getNL();
    emplaceTransformation(new ReplaceText(
        BeginLoc, Len,
        "(((int)" + TransParamName +
            ")==2?(mkl::transpose::conjtrans):((mkl::transpose)" +
            TransParamName + "))"));
  }
  if (FillModeIndexInfo == DistinctionID) {
    emplaceTransformation(
        new ReplaceText(BeginLoc, Len,
                        "(((int)" + SubExprStr +
                            ")==0?(mkl::uplo::lower):(mkl::uplo::upper))"));
  }
  // the value of enum in mkl::side/cublasSideMode_t and
  // mkl::diag/cublasDiagType_t is same, so we don't need to
  // transfer
}

void BLASFunctionCallRule::processTrmmParams(
    const CallExpr *CE, std::string &PrefixInsertStr, std::string &BufferName,
    std::string &BufferDecl, int &IndexTemp, int DistinctionID,
    const std::string IndentStr, const std::vector<std::string> &BufferTypeInfo,
    const SourceLocation &StmtBegin) {
  auto &Context = syclct::SyclctGlobalInfo::getContext();
  // decl a temp var for ptrB and ptrC
  PrefixInsertStr = PrefixInsertStr + IndentStr + "auto ptr_ct" +
                    std::to_string(DistinctionID) + " = " +
                    getStmtSpelling(CE->getArg(DistinctionID), Context) + ";" +
                    getNL();
  BufferName = getBufferNameAndDeclStr("ptr_ct" + std::to_string(DistinctionID),
                                       Context, BufferTypeInfo[IndexTemp],
                                       StmtBegin, BufferDecl, DistinctionID);
}

void BLASFunctionCallRule::processTrmmCall(const CallExpr *CE,
                                           std::string &PrefixInsertStr,
                                           const std::string IndentStr) {
  auto &SM = syclct::SyclctGlobalInfo::getSourceManager();
  auto &Context = syclct::SyclctGlobalInfo::getContext();
  // remove parameters ptrB and ldb
  Optional<Token> TokSharedPtr;
  TokSharedPtr =
      Lexer::findNextToken(CE->getArg(11)->getEndLoc(), SM, LangOptions());
  Token CommaTok = TokSharedPtr.getValue();
  auto CommaEnd = CommaTok.getEndLoc();
  auto Len = SM.getCharacterData(CommaEnd) -
             SM.getCharacterData(CE->getArg(10)->getBeginLoc());
  emplaceTransformation(
      new ReplaceText(CE->getArg(10)->getBeginLoc(), Len, ""));
  // decl fout temp vars for ldb, ldc, n and m
  PrefixInsertStr =
      PrefixInsertStr + IndentStr +
      "auto ld_ct13 = " + getStmtSpelling(CE->getArg(13), Context) + ";" +
      " auto m_ct5 = " + getStmtSpelling(CE->getArg(5), Context) +
      "; auto n_ct6 = " + getStmtSpelling(CE->getArg(6), Context) + ";" +
      getNL();
  // insert a stmt copying the data ptrB pointing to where ptrC pointing
  PrefixInsertStr = PrefixInsertStr + IndentStr +
                    "syclct::matrix_mem_copy(ptr_ct12, " +
                    getStmtSpelling(CE->getArg(10), Context) + ", ld_ct13, " +
                    getStmtSpelling(CE->getArg(11), Context) +
                    ", m_ct5, n_ct6, syclct::device_to_device);" + getNL();
  // replace the args in the function call
  emplaceTransformation(new ReplaceStmt(CE->getArg(13), "ld_ct13"));
  emplaceTransformation(new ReplaceStmt(CE->getArg(5), "m_ct5"));
  emplaceTransformation(new ReplaceStmt(CE->getArg(6), "n_ct6"));
}

REGISTER_RULE(BLASFunctionCallRule)

// Rule for SOLVER enums.
// All SOLVER enums have the prefix CUSOLVER_
// Migrate SOLVER status values to corresponding int values
// Example: migrate CUSOLVER_STATUS_SUCCESS to 0
// Other SOLVER named values are migrated to corresponding named values
void SOLVEREnumsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      declRefExpr(to(enumConstantDecl(matchesName("CUSOLVER_STATU.*"))))
          .bind("SOLVERStatusConstants"),
      this);
  MF.addMatcher(
      declRefExpr(to(enumConstantDecl(matchesName(
                      "(CUSOLVER_EIG_TYPE.*)|(CUSOLVER_EIG_MODE.*)"))))
          .bind("SLOVERNamedValueConstants"),
      this);
}

void SOLVEREnumsRule::run(const MatchFinder::MatchResult &Result) {
  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "SOLVERStatusConstants")) {
    auto *EC = cast<EnumConstantDecl>(DE->getDecl());
    emplaceTransformation(new ReplaceStmt(DE, EC->getInitVal().toString(10)));
  }

  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "SLOVERNamedValueConstants")) {
    auto *EC = cast<EnumConstantDecl>(DE->getDecl());
    std::string Name = EC->getNameAsString();
    auto Search = MapNames::SOLVEREnumsMap.find(Name);
    if (Search == MapNames::SOLVEREnumsMap.end()) {
      syclct_unreachable("migration error");
      return;
    }
    std::string Replacement = Search->second;
    emplaceTransformation(new ReplaceStmt(DE, std::move(Replacement)));
  }
}

REGISTER_RULE(SOLVEREnumsRule)

void SOLVERFunctionCallRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName(
        "cusolverDnCreate", "cusolverDnDestroy", "cusolverDnSpotrf_bufferSize",
        "cusolverDnDpotrf_bufferSize", "cusolverDnCpotrf_bufferSize",
        "cusolverDnZpotrf_bufferSize", "cusolverDnSpotri_bufferSize",
        "cusolverDnDpotri_bufferSize", "cusolverDnCpotri_bufferSize",
        "cusolverDnZpotri_bufferSize", "cusolverDnSgetrf_bufferSize",
        "cusolverDnDgetrf_bufferSize", "cusolverDnCgetrf_bufferSize",
        "cusolverDnZgetrf_bufferSize", "cusolverDnSpotrf", "cusolverDnDpotrf",
        "cusolverDnCpotrf", "cusolverDnZpotrf", "cusolverDnSpotrs",
        "cusolverDnDpotrs", "cusolverDnCpotrs", "cusolverDnZpotrs",
        "cusolverDnSpotri", "cusolverDnDpotri", "cusolverDnCpotri",
        "cusolverDnZpotri", "cusolverDnSgetrf", "cusolverDnDgetrf",
        "cusolverDnCgetrf", "cusolverDnZgetrf", "cusolverDnSgetrs",
        "cusolverDnDgetrs", "cusolverDnCgetrs", "cusolverDnZgetrs",
        "cusolverDnSgeqrf_bufferSize", "cusolverDnDgeqrf_bufferSize",
        "cusolverDnCgeqrf_bufferSize", "cusolverDnZgeqrf_bufferSize",
        "cusolverDnSgeqrf", "cusolverDnDgeqrf", "cusolverDnCgeqrf",
        "cusolverDnZgeqrf", "cusolverDnSormqr_bufferSize",
        "cusolverDnDormqr_bufferSize", "cusolverDnSormqr", "cusolverDnDormqr",
        "cusolverDnCunmqr_bufferSize", "cusolverDnZunmqr_bufferSize",
        "cusolverDnCunmqr", "cusolverDnZunmqr", "cusolverDnSorgqr_bufferSize",
        "cusolverDnDorgqr_bufferSize", "cusolverDnCungqr_bufferSize",
        "cusolverDnZungqr_bufferSize", "cusolverDnSorgqr", "cusolverDnDorgqr",
        "cusolverDnCungqr", "cusolverDnZungqr", "cusolverDnSsytrf_bufferSize",
        "cusolverDnDsytrf_bufferSize", "cusolverDnCsytrf_bufferSize",
        "cusolverDnZsytrf_bufferSize", "cusolverDnSsytrf", "cusolverDnDsytrf",
        "cusolverDnCsytrf", "cusolverDnZsytrf", "cusolverDnSgebrd_bufferSize",
        "cusolverDnDgebrd_bufferSize", "cusolverDnCgebrd_bufferSize",
        "cusolverDnZgebrd_bufferSize", "cusolverDnSgebrd", "cusolverDnDgebrd",
        "cusolverDnCgebrd", "cusolverDnZgebrd", "cusolverDnSorgbr_bufferSize",
        "cusolverDnDorgbr_bufferSize", "cusolverDnCungbr_bufferSize",
        "cusolverDnZungbr_bufferSize", "cusolverDnSorgbr", "cusolverDnDorgbr",
        "cusolverDnCungbr", "cusolverDnZungbr", "cusolverDnSsytrd_bufferSize",
        "cusolverDnDsytrd_bufferSize", "cusolverDnChetrd_bufferSize",
        "cusolverDnZhetrd_bufferSize", "cusolverDnSsytrd", "cusolverDnDsytrd",
        "cusolverDnChetrd", "cusolverDnZhetrd", "cusolverDnSormtr_bufferSize",
        "cusolverDnDormtr_bufferSize", "cusolverDnCunmtr_bufferSize",
        "cusolverDnZunmtr_bufferSize", "cusolverDnSormtr", "cusolverDnDormtr",
        "cusolverDnCunmtr", "cusolverDnZunmtr", "cusolverDnSorgtr_bufferSize",
        "cusolverDnDorgtr_bufferSize", "cusolverDnCungtr_bufferSize",
        "cusolverDnZungtr_bufferSize", "cusolverDnSorgtr", "cusolverDnDorgtr",
        "cusolverDnCungtr", "cusolverDnZungtr", "cusolverDnSgesvd_bufferSize",
        "cusolverDnDgesvd_bufferSize", "cusolverDnCgesvd_bufferSize",
        "cusolverDnZgesvd_bufferSize", "cusolverDnSgesvd", "cusolverDnDgesvd",
        "cusolverDnCgesvd", "cusolverDnZgesvd", "cusolverDnSsyevd_bufferSize",
        "cusolverDnDsyevd_bufferSize", "cusolverDnSsyevd_bufferSize",
        "cusolverDnCheevd_bufferSize", "cusolverDnZheevd_bufferSize",
        "cusolverDnDsyevd", "cusolverDnSsyevd", "cusolverDnCheevd",
        "cusolverDnZheevd");
  };

  MF.addMatcher(callExpr(allOf(callee(functionDecl(functionName())),
                               hasAncestor(functionDecl(
                                   anyOf(hasAttr(attr::CUDADevice),
                                         hasAttr(attr::CUDAGlobal))))))
                    .bind("kernelCall"),
                this);

  MF.addMatcher(
      callExpr(
          allOf(callee(functionDecl(functionName())), parentStmt(),
                hasAncestor(functionDecl(unless(allOf(
                    hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))))))
          .bind("FunctionCall"),
      this);
  MF.addMatcher(
      callExpr(
          allOf(callee(functionDecl(functionName())), unless(parentStmt()),
                unless(hasParent(varDecl())),
                hasAncestor(functionDecl(unless(allOf(
                    hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))))))
          .bind("FunctionCallUsedNotInitializeVarDecl"),
      this);

  MF.addMatcher(
      callExpr(
          allOf(callee(functionDecl(functionName())), hasParent(varDecl()),
                hasAncestor(functionDecl(unless(allOf(
                    hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))))))
          .bind("FunctionCallUsedInitializeVarDecl"),
      this);
}

void SOLVERFunctionCallRule::run(const MatchFinder::MatchResult &Result) {
  bool IsAssigned = false;
  bool IsInitializeVarDecl = false;
  bool HasDeviceAttr = false;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "kernelCall");
  if (CE) {
    HasDeviceAttr = true;
  } else if (!(CE = getNodeAsType<CallExpr>(Result, "FunctionCall"))) {
    if ((CE = getNodeAsType<CallExpr>(
             Result, "FunctionCallUsedNotInitializeVarDecl"))) {
      IsAssigned = true;
    } else if ((CE = getNodeAsType<CallExpr>(
                    Result, "FunctionCallUsedInitializeVarDecl"))) {
      IsAssigned = true;
      IsInitializeVarDecl = true;
    } else {
      return;
    }
  }

  assert(CE && "Unknown result");

  // Collect sourceLocations of the function call
  SourceLocation FuncNameBegin(CE->getBeginLoc());
  SourceLocation FuncCallEnd(CE->getEndLoc());

  // Correct sourceLocations for macros
  const SourceManager *SM = Result.SourceManager;
  if (FuncNameBegin.isMacroID())
    FuncNameBegin = SM->getExpansionLoc(FuncNameBegin);
  if (FuncCallEnd.isMacroID())
    FuncCallEnd = SM->getExpansionLoc(FuncCallEnd);

  // Collect sourceLocations for creating new scope
  std::string PrefixBeforeScope, PrefixInsertStr, SuffixInsertStr;
  auto SR = getScopeInsertRange(CE, FuncNameBegin, FuncCallEnd);
  SourceLocation StmtBegin = SR.getBegin(), StmtEndAfterSemi = SR.getEnd();
  std::string IndentStr =
      getIndent(StmtBegin, (Result.Context)->getSourceManager()).str();
  Token Tok;
  Lexer::getRawToken(FuncNameBegin, Tok, *SM, LangOptions());
  SourceLocation FuncNameEnd = Tok.getEndLoc();
  auto FuncNameLength =
      SM->getCharacterData(FuncNameEnd) - SM->getCharacterData(FuncNameBegin);

  // Prepare the prefix and the postfix for assignment
  if (!CE->getDirectCallee())
    return;
  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();
  std::string AssignPrefix = "";
  std::string AssignPostfix = "";

  if (IsAssigned) {
    AssignPrefix = "(";
    AssignPostfix = ", 0)";
  }

  if (HasDeviceAttr) {
    report(CE->getBeginLoc(), Diagnostics::FUNCTION_CALL_IN_DEVICE, FuncName,
           "syclct::sycl_memcpy");
    return;
  }

  const VarDecl *VD = 0;
  if (IsInitializeVarDecl) {
    // Create the prefix for VarDecl before scope and remove the VarDecl inside
    // the scope
    VD = getAncestralVarDecl(CE);
    std::string VarType, VarName;
    if (VD) {
      VarType = VD->getType().getAsString();
      VarName = VD->getNameAsString();
      auto Itr = MapNames::TypeNamesMap.find(VarType);
      if (Itr != MapNames::TypeNamesMap.end())
        VarType = Itr->second;
      PrefixBeforeScope = VarType + " " + VarName + ";" + getNL() + IndentStr +
                          PrefixBeforeScope;
      SourceLocation typeBegin =
          VD->getTypeSourceInfo()->getTypeLoc().getBeginLoc();
      SourceLocation nameBegin = VD->getLocation();
      SourceLocation nameEnd = Lexer::getLocForEndOfToken(
          nameBegin, 0, *SM, Result.Context->getLangOpts());
      auto replLen =
          SM->getCharacterData(nameEnd) - SM->getCharacterData(typeBegin);
      emplaceTransformation(
          new ReplaceText(typeBegin, replLen, std::move(VarName)));
    } else {
      assert(0 && "Fail to get VarDecl information");
      return;
    }
  }

  if (MapNames::SOLVERFuncReplInfoMap.find(FuncName) !=
      MapNames::SOLVERFuncReplInfoMap.end()) {
    // Find replacement string
    auto ReplInfoPair = MapNames::SOLVERFuncReplInfoMap.find(FuncName);
    MapNames::SOLVERFuncReplInfo ReplInfo = ReplInfoPair->second;
    std::string Replacement = ReplInfo.ReplName;

    // Migrate arguments one by one
    int ArgNum = CE->getNumArgs();
    for (int i = 0; i < ArgNum; ++i) {
      int IndexTemp = -1;
      if (isReplIndex(i, ReplInfo.BufferIndexInfo, IndexTemp)) {
        std::string BufferDecl;
        std::string BufferName = getBufferNameAndDeclStr(
            CE->getArg(i), *(Result.Context),
            ReplInfo.BufferTypeInfo[IndexTemp], StmtBegin, BufferDecl, i);
        PrefixInsertStr = PrefixInsertStr + BufferDecl;
        if (ReplInfo.BufferTypeInfo[IndexTemp] == "int") {
          PrefixInsertStr = PrefixInsertStr + IndentStr +
                            "cl::sycl::buffer<int64_t,1> "
                            "result_temp_buffer" +
                            std::to_string(i) + "(cl::sycl::range<1>(1));" +
                            getNL();
          SuffixInsertStr = SuffixInsertStr + BufferName +
                            ".get_access<cl::sycl::access::"
                            "mode::write>()[0] = "
                            "(int)result_temp_buffer" +
                            std::to_string(i) +
                            ".get_access<cl::sycl::"
                            "access::mode::read>()[0];" +
                            getNL() + IndentStr;
          emplaceTransformation(new ReplaceStmt(
              CE->getArg(i), "result_temp_buffer" + std::to_string(i)));
          continue;
        }
        emplaceTransformation(
            new ReplaceStmt(CE->getArg(i), std::move(BufferName)));
      }
      if (isReplIndex(i, ReplInfo.RedundantIndexInfo, IndexTemp)) {
        SourceLocation ParameterEndAfterSemi;
        getParameterEnd(CE->getArg(i)->getEndLoc(), ParameterEndAfterSemi,
                        Result);
        auto ParameterLength =
            SM->getCharacterData(ParameterEndAfterSemi) -
            SM->getCharacterData(CE->getArg(i)->getBeginLoc());
        emplaceTransformation(
            new ReplaceText(CE->getArg(i)->getBeginLoc(), ParameterLength, ""));
      }
    }

    if (!ReplInfo.MissedArgumentFinalLocation.empty()) {
      std::string ReplStr;
      for (size_t i = 0; i < ReplInfo.MissedArgumentFinalLocation.size(); ++i) {
        if (ReplInfo.MissedArgumentIsBuffer[i]) {
          PrefixInsertStr = PrefixInsertStr + IndentStr + "cl::sycl::buffer<" +
                            ReplInfo.MissedArgumentType[i] + ",1> " +
                            ReplInfo.MissedArgumentName[i] +
                            "(cl::sycl::range<1>(1));" + getNL();
        } else {
          PrefixInsertStr = PrefixInsertStr + IndentStr +
                            ReplInfo.MissedArgumentType[i] + " " +
                            ReplInfo.MissedArgumentName[i] + ";" + getNL();
        }
        ReplStr = ReplStr + ReplInfo.MissedArgumentName[i] + ", ";
        if (i == ReplInfo.MissedArgumentFinalLocation.size() ||
            ReplInfo.MissedArgumentInsertBefore[i + 1] !=
                ReplInfo.MissedArgumentInsertBefore[i]) {
          emplaceTransformation(new InsertBeforeStmt(
              CE->getArg(ReplInfo.MissedArgumentInsertBefore[i]),
              std::move(ReplStr)));
          ReplStr = "";
        }
      }
    }

    if (!ReplInfo.CastIndexInfo.empty()) {
      for (size_t i = 0; i < ReplInfo.CastIndexInfo.size(); ++i) {
        std::string CastStr = "(" + ReplInfo.CastTypeInfo[i] + ")";
        emplaceTransformation(new InsertBeforeStmt(
            CE->getArg(ReplInfo.CastIndexInfo[i]), std::move(CastStr)));
      }
    }

    if (IsAssigned) {
      insertAroundRange(FuncNameBegin, FuncCallEnd.getLocWithOffset(1),
                        std::move(AssignPrefix), std::move(AssignPostfix));
      report(FuncNameBegin, Diagnostics::NOERROR_RETURN_COMMA_OP);
    }
    emplaceTransformation(
        new ReplaceText(FuncNameBegin, FuncNameLength, std::move(Replacement)));
    insertAroundRange(StmtBegin, StmtEndAfterSemi,
                      PrefixBeforeScope + std::string("{") + getNL() +
                          PrefixInsertStr + IndentStr,
                      getNL() + IndentStr + SuffixInsertStr + std::string("}"));
  } else if (FuncName == "cusolverDnCreate" ||
             FuncName == "cusolverDnDestroy" ||
             FuncName == "cusolverDnSpotrf_bufferSize" ||
             FuncName == "cusolverDnDpotrf_bufferSize" ||
             FuncName == "cusolverDnCpotrf_bufferSize" ||
             FuncName == "cusolverDnZpotrf_bufferSize" ||
             FuncName == "cusolverDnSpotri_bufferSize" ||
             FuncName == "cusolverDnDpotri_bufferSize" ||
             FuncName == "cusolverDnCpotri_bufferSize" ||
             FuncName == "cusolverDnZpotri_bufferSize" ||
             FuncName == "cusolverDnSgetrf_bufferSize" ||
             FuncName == "cusolverDnDgetrf_bufferSize" ||
             FuncName == "cusolverDnCgetrf_bufferSize" ||
             FuncName == "cusolverDnZgetrf_bufferSize") {
    // Replace helper function calls to "0" or ""
    if (IsAssigned) {
      emplaceTransformation(
          new ReplaceStmt(CE, /*IsReplaceCompatibilityAPI*/ false, FuncName,
                          /*IsProcessMacro*/ true, "0"));
    } else {
      emplaceTransformation(
          new ReplaceStmt(CE, /*IsReplaceCompatibilityAPI*/ false, FuncName,
                          /*IsProcessMacro*/ true, ""));
    }
  }
}

void SOLVERFunctionCallRule::getParameterEnd(
    const SourceLocation &ParameterEnd, SourceLocation &ParameterEndAfterComma,
    const ast_matchers::MatchFinder::MatchResult &Result) {
  Optional<Token> TokSharedPtr;
  TokSharedPtr = Lexer::findNextToken(ParameterEnd, *(Result.SourceManager),
                                      LangOptions());
  Token TokComma = TokSharedPtr.getValue();
  ParameterEndAfterComma = TokComma.getEndLoc();
  // TODO: Check if TokComma is real comma
}

bool SOLVERFunctionCallRule::isReplIndex(int Input, std::vector<int> &IndexInfo,
                                       int &IndexTemp) {
  for (int i = 0; i < static_cast<int>(IndexInfo.size()); ++i) {
    if (IndexInfo[i] == Input) {
      IndexTemp = i;
      return true;
    }
  }
  return false;
}

// TODO: Refactoring with BLASFunctionCallRule for removing duplication
std::string SOLVERFunctionCallRule::getBufferNameAndDeclStr(
    const Expr *Arg, const ASTContext &AC, const std::string &TypeAsStr,
    SourceLocation SL, std::string &BufferDecl, int DistinctionID) {

  std::string PointerName = getStmtSpelling(Arg, AC);
  std::string PointerNameHashStr = getHashAsString(PointerName);
  PointerNameHashStr = (PointerNameHashStr.size() < 4)
                           ? PointerNameHashStr
                           : PointerNameHashStr.substr(0, 3);

  std::string BufferTempName = "buffer_ct" + std::to_string(DistinctionID);
  std::string AllocationTempName =
      "allocation_ct" + std::to_string(DistinctionID);

  // TODO: reinterpret will copy more data
  BufferDecl = getIndent(SL, AC.getSourceManager()).str() + "auto " +
               AllocationTempName +
               " = syclct::memory_manager::get_instance().translate_ptr(" +
               PointerName + ");" + getNL() +
               getIndent(SL, AC.getSourceManager()).str() +
               "cl::sycl::buffer<" + TypeAsStr + ",1> " + BufferTempName +
               " = " + AllocationTempName + ".buffer.reinterpret<" + TypeAsStr +
               ", 1>(cl::sycl::range<1>(" + AllocationTempName +
               ".size/sizeof(" + TypeAsStr + ")));" + getNL();
  return BufferTempName;
}

const clang::VarDecl *
SOLVERFunctionCallRule::getAncestralVarDecl(const clang::CallExpr *CE) {
  auto &Context = syclct::SyclctGlobalInfo::getContext();
  auto Parents = Context.getParents(*CE);
  while (Parents.size() == 1) {
    auto *Parent = Parents[0].get<VarDecl>();
    if (Parent) {
      return Parent;
    }
    else {
      Parents = Context.getParents(Parents[0]);
    }
  }
  return nullptr;
}

REGISTER_RULE(SOLVERFunctionCallRule)

void FunctionCallRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName(
        "cudaGetDeviceCount", "cudaGetDeviceProperties", "cudaDeviceReset",
        "cudaSetDevice", "cudaDeviceGetAttribute", "cudaDeviceGetP2PAttribute",
        "cudaDeviceGetPCIBusId", "cudaGetDevice", "cudaDeviceSetLimit",
        "cudaGetLastError", "cudaPeekAtLastError", "cudaDeviceSynchronize",
        "cudaThreadSynchronize", "cudaGetErrorString", "cudaGetErrorName",
        "cudaDeviceSetCacheConfig", "cudaDeviceGetCacheConfig", "clock",
        "cudaOccupancyMaxPotentialBlockSize", "cudaThreadSetLimit",
        "cudaFuncSetCacheConfig", "cudaThreadExit");
  };

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(functionName())), parentStmt()))
          .bind("FunctionCall"),
      this);
  MF.addMatcher(callExpr(allOf(callee(functionDecl(functionName())),
                               unless(parentStmt())))
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
  std::string Prefix, Suffix;
  if (IsAssigned) {
    Prefix = "(";
    Suffix = ", 0)";
  }

  if (FuncName == "cudaGetDeviceCount") {
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
    }
    std::string ResultVarName = DereferenceArg(CE->getArg(0), *Result.Context);
    emplaceTransformation(
        new InsertBeforeStmt(CE, Prefix + ResultVarName + " = "));
    emplaceTransformation(new ReplaceStmt(
        CE, "syclct::get_device_manager().device_count()" + Suffix));
  } else if (FuncName == "cudaGetDeviceProperties") {
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
    }
    std::string ResultVarName = DereferenceArg(CE->getArg(0), *Result.Context);
    emplaceTransformation(new ReplaceStmt(
        CE->getCallee(), Prefix + "syclct::get_device_manager().get_device"));
    emplaceTransformation(new RemoveArg(CE, 0));
    emplaceTransformation(new InsertAfterStmt(
        CE, ".get_device_info(" + ResultVarName + ")" + Suffix));
  } else if (FuncName == "cudaDeviceReset" || FuncName == "cudaThreadExit") {
    // The functionality of cudaThreadExit() is identical to cudaDeviceReset().
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
    }
    emplaceTransformation(new ReplaceStmt(
        CE, Prefix + "syclct::get_device_manager().current_device().reset()" +
                Suffix));
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
  } else if (FuncName == "cudaDeviceGetPCIBusId") {
    report(CE->getBeginLoc(), Comments::NOTSUPPORTED, "Get PCI BusId");
  } else if (FuncName == "cudaGetDevice") {
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
    auto Loc = CE->getBeginLoc();
    std::string Path = SyclctGlobalInfo::getSourceManager().getFilename(
        SyclctGlobalInfo::getSourceManager().getExpansionLoc(Loc));
    makeCanonical(Path);
    auto IncludeLoc = IncludeLocations[Path];
    if (TimeHeaderFilter.find(Path) == TimeHeaderFilter.end()) {
      TimeHeaderFilter.insert(Path);
      emplaceTransformation(new InsertText(
          IncludeLoc, getNL() +
                          std::string("#include <time.h> // For clock_t, "
                                      "clock and CLOCKS_PER_SEC") +
                          getNL()));
    }
  } else if (FuncName == "cudaDeviceSetLimit" ||
             FuncName == "cudaThreadSetLimit") {
    report(CE->getBeginLoc(), Diagnostics::NOTSUPPORTED, FuncName);
    emplaceTransformation(new ReplaceStmt(CE, false, FuncName, ""));
  } else if (FuncName == "cudaFuncSetCacheConfig") {
    report(CE->getBeginLoc(), Diagnostics::NOTSUPPORTED, FuncName);
  } else if (FuncName == "cudaOccupancyMaxPotentialBlockSize") {
    report(CE->getBeginLoc(), Diagnostics::NOTSUPPORTED, FuncName);
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
      callExpr(allOf(callee(functionDecl(eventAPIName())), parentStmt()))
          .bind("eventAPICall"),
      this);
  MF.addMatcher(callExpr(allOf(callee(functionDecl(eventAPIName())),
                               unless(parentStmt())))
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
    if (IsAssigned)
      ReplStr = "0";
    emplaceTransformation(new ReplaceStmt(CE, false, FuncName, ReplStr));
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
    emplaceTransformation(new ReplaceStmt(CE, false, FuncName, ReplStr));
  } else {
    syclct_unreachable("Unknown function name");
  }
}

void EventAPICallRule::handleEventRecord(const CallExpr *CE,
                                         const MatchFinder::MatchResult &Result,
                                         bool IsAssigned) {
  report(CE->getBeginLoc(), Diagnostics::TIME_MEASUREMENT_FOUND);
  std::ostringstream Repl;

  // Define the helper variable if it is used in the block for first time,
  // otherwise, just use it.
  static std::set<std::pair<const CompoundStmt *, const std::string>> DupFilter;
  const auto *CS = findImmediateBlock(CE);
  auto StmtStr = getStmtSpelling(CE->getArg(0), *Result.Context);
  auto Pair = std::make_pair(CS, StmtStr);

  if (DupFilter.find(Pair) == DupFilter.end()) {
    DupFilter.insert(Pair);
    Repl << "auto ";
  }

  Repl << StmtStr << getCTFixedSuffix() << " = clock()";
  const std::string Name =
      CE->getCalleeDecl()->getAsFunction()->getNameAsString();
  if (IsAssigned) {
    emplaceTransformation(new ReplaceStmt(CE, false, Name, "0"));
    report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_ZERO);
    auto OuterStmt = findNearestNonExprNonDeclAncestorStmt(CE);
    Repl << ", ";
    emplaceTransformation(new InsertBeforeStmt(OuterStmt, std::move(Repl.str()),
                                               /*PairID*/ 0,
                                               /*DoMacroExpansion*/ true));
  } else {
    emplaceTransformation(
        new ReplaceStmt(CE, false, Name, std::move(Repl.str())));
  }
}

void EventAPICallRule::handleEventElapsedTime(
    const CallExpr *CE, const MatchFinder::MatchResult &Result,
    bool IsAssigned) {
  auto StmtStrArg0 = getStmtSpelling(CE->getArg(0), *Result.Context);
  auto StmtStrArg1 = getStmtSpelling(CE->getArg(1), *Result.Context);
  auto StmtStrArg2 = getStmtSpelling(CE->getArg(2), *Result.Context);
  std::ostringstream Repl;
  Repl << "*(" << StmtStrArg0 << ") = (float)(" << StmtStrArg2
       << getCTFixedSuffix() << " - " << StmtStrArg1 << getCTFixedSuffix()
       << ") / CLOCKS_PER_SEC * 1000";
  if (IsAssigned) {
    std::ostringstream Temp;
    Temp << "(" << Repl.str() << ", 0)";
    Repl = std::move(Temp);
    report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
  }
  const std::string Name =
      CE->getCalleeDecl()->getAsFunction()->getNameAsString();
  emplaceTransformation(new ReplaceStmt(CE, false, Name, std::move(Repl.str())));
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
        continue;
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
    const CUDAKernelCallExpr *KCall = nullptr;
    if (auto *Expr = dyn_cast<ExprWithCleanups>(*Iter)) {
      auto *SubExpr = Expr->getSubExpr();
      if (auto *Call = dyn_cast<CUDAKernelCallExpr>(SubExpr))
        KCall = Call;
    } else if (auto *Call = dyn_cast<CUDAKernelCallExpr>(*Iter)) {
      KCall = Call;
    }

    if (KCall) {
      auto KCallLoc = KCall->getBeginLoc().getRawEncoding();
      // Only the kernel calls between begin and end are set to be synced
      if (KCallLoc > RecordBeginLoc && KCallLoc < RecordEndLoc) {
        auto K = SyclctGlobalInfo::getInstance().insertKernelCallExpr(KCall);
        K->setEvent(getStmtSpelling(CE->getArg(2), *Result.Context));
        K->setSync();
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
      callExpr(allOf(callee(functionDecl(streamFunctionName())), parentStmt()))
          .bind("streamAPICall"),
      this);
  MF.addMatcher(callExpr(allOf(callee(functionDecl(streamFunctionName())),
                               unless(parentStmt())))
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
    emplaceTransformation(new ReplaceStmt(CE, false, FuncName, ReplStr));
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
    emplaceTransformation(new ReplaceStmt(CE, false, Name, ReplStr));
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
    emplaceTransformation(new ReplaceStmt(CE, false, Name, ReplStr));
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
    emplaceTransformation(new ReplaceStmt(CE, false, Name, ReplStr));
  } else if (FuncName == "cudaStreamAttachMemAsync" ||
             FuncName == "cudaStreamBeginCapture" ||
             FuncName == "cudaStreamEndCapture" ||
             FuncName == "cudaStreamIsCapturing" ||
             FuncName == "cudaStreamQuery" ||
             FuncName == "cudaStreamWaitEvent") {
    report(CE->getBeginLoc(), Diagnostics::NOTSUPPORTED, FuncName);
    emplaceTransformation(new ReplaceStmt(CE, false, FuncName, ""));
  } else if (FuncName == "cudaStreamAddCallback") {
    auto StmtStr0 = getStmtSpelling(CE->getArg(0), *Result.Context);
    auto StmtStr1 = getStmtSpelling(CE->getArg(1), *Result.Context);
    auto StmtStr2 = getStmtSpelling(CE->getArg(2), *Result.Context);
    std::string ReplStr{"std::async([&]() { "};
    ReplStr += StmtStr0;
    ReplStr += ".wait(); ";
    ReplStr += StmtStr1;
    ReplStr += "(";
    ReplStr += StmtStr0;
    ReplStr += ", 0, ";
    ReplStr += StmtStr2;
    ReplStr += "); ";
    ReplStr += "})";
    if (IsAssigned) {
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
    }
    emplaceTransformation(new ReplaceStmt(CE, false, FuncName, ReplStr));
    insertIncludeFile(CE->getBeginLoc(), FutureHeaderFilter,
                      getNL() + std::string("#include <future>") + getNL());
  } else {
    syclct_unreachable("Unknown function name");
  }
}

REGISTER_RULE(StreamAPICallRule)

static const CXXConstructorDecl *getIfConstructorDecl(const Decl *ND) {
  if (const auto *Tmpl = dyn_cast<FunctionTemplateDecl>(ND))
    ND = Tmpl->getTemplatedDecl();
  return dyn_cast<CXXConstructorDecl>(ND);
}

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
                               .bind("funcDecl")),
               anyOf(callee(functionDecl(hasName("printf")).bind("printf")),
                     anything()))
          .bind("callExpr"),
      this);
}

void DeviceFunctionCallRule::run(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  auto CE = getAssistNodeAsType<CallExpr>(Result, "callExpr");
  auto FD = getAssistNodeAsType<FunctionDecl>(Result, "funcDecl");
  if (CE && FD) {
    auto FuncInfo = DeviceFunctionDecl::LinkRedecls(FD);
    FuncInfo->addCallee(CE);
    if (getAssistNodeAsType<FunctionDecl>(Result, "printf", false)) {
      emplaceTransformation(new ReplaceStmt(
          CE,
          buildString(SyclctGlobalInfo::getStreamName(),
                      " << \"TODO - output needs update\" << cl::sycl::endl")));
      report(CE->getBeginLoc(), Warnings::PRINTF_FUNC_MIGRATION_WARNING);
      FuncInfo->setStream();
    }
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

void MemoryTranslationRule::mallocTranslation(
    const MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr) {
  std::string Name;
  if (ULExpr) {
    Name = ULExpr->getName().getAsString();
  } else {
    Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  }
  if (Name == "cudaMalloc") {
    SyclctGlobalInfo::getInstance().insertCudaMalloc(C);
    emplaceTransformation(
        new ReplaceCalleeName(C, "syclct::sycl_malloc", Name));
  } else if (Name == "cublasAlloc") {
    // TODO: migrate functions when they are in template
    // TODO: migrate functions when they are in macro body
    SyclctGlobalInfo::getInstance().insertCublasAlloc(C);
    auto PtrStr = getStmtSpelling(C->getArg(2), *(Result.Context));
    auto SizeStr = "(" + getStmtSpelling(C->getArg(0), *(Result.Context)) +
                   ")*(" + getStmtSpelling(C->getArg(1), *(Result.Context)) +
                   ")";
    std::string Replacement =
        "syclct::sycl_malloc(" + PtrStr + ", " + SizeStr + ")";
    emplaceTransformation(new ReplaceStmt(C, std::move(Replacement)));
  } else {
    syclct_unreachable("Unknown function name");
  }
}

const ArraySubscriptExpr *
MemoryTranslationRule::getArraySubscriptExpr(const Expr *E) {
  if (const auto MTE = dyn_cast<MaterializeTemporaryExpr>(E)) {
    if (auto TE = MTE->GetTemporaryExpr()) {
      if (auto UO = dyn_cast<UnaryOperator>(TE)) {
        if (auto Arg = dyn_cast<ArraySubscriptExpr>(UO->getSubExpr())) {
          return Arg;
        }
      }
    }
  }
  return nullptr;
}

const Expr *MemoryTranslationRule::getUnaryOperatorExpr(const Expr *E) {
  if (const auto MTE = dyn_cast<MaterializeTemporaryExpr>(E)) {
    if (auto TE = MTE->GetTemporaryExpr()) {
      if (auto UO = dyn_cast<UnaryOperator>(TE)) {
        return UO->getSubExpr();
      }
    }
  }
  return nullptr;
}

void MemoryTranslationRule::replaceMemAPIArg(
    const Expr *E, const ast_matchers::MatchFinder::MatchResult &Result,
    std::string OffsetFromBaseStr) {

  auto ASE = getArraySubscriptExpr(E);
  const clang::Expr *BASE = nullptr;
  if (ASE) {
    BASE = ASE->getBase();
  }

  auto UO = getUnaryOperatorExpr(E);
  std::shared_ptr<clang::syclct::MemVarInfo> VI = nullptr;
  if (BASE &&
      (VI = SyclctGlobalInfo::getInstance().findMemVarInfo(getVarDecl(BASE)))) {
    // Migrate the expr such as "&const_angle[3]" to
    // const_angle.get_ptr() + sizeof(TYPE) * (3)",
    // and "&const_angle[3]" to "const_angle.get_ptr()".
    std::string VarName = VI->getName();
    VarName += ".get_ptr()";

    bool IsOffsetNeeded = true;
    Expr::EvalResult ER;

    if (ASE->getIdx()->EvaluateAsInt(ER, *Result.Context)) {
      auto ExprValue = ER.Val.getAsString(*Result.Context, ASE->getType());
      if (ExprValue == "0") {
        IsOffsetNeeded = false;
      }
    }

    if (IsOffsetNeeded) {
      std::string Type = ASE->getType().getAsString();
      auto StmtStrArg = getStmtSpelling(ASE->getIdx(), *Result.Context);
      std::string Offset = " + sizeof(" + Type + ") * (" + StmtStrArg + ")";
      VarName += Offset;
    }

    if (!OffsetFromBaseStr.empty()) {
      VarName =
          "(void *)((char *)(" + VarName + ") + " + OffsetFromBaseStr + ")";
    }
    emplaceTransformation(
        new ReplaceToken(E->getBeginLoc(), E->getEndLoc(), std::move(VarName)));
  } else if (UO && (VI = SyclctGlobalInfo::getInstance().findMemVarInfo(
                        getVarDecl(UO)))) {
    // Migrate the expr such as "&const_one" to "const_one.get_ptr()".
    std::string VarName = VI->getName();
    VarName += ".get_ptr()";

    if (!OffsetFromBaseStr.empty()) {
      VarName =
          "(void *)((char *)(" + VarName + ") + " + OffsetFromBaseStr + ")";
    }
    emplaceTransformation(
        new ReplaceToken(E->getBeginLoc(), E->getEndLoc(), std::move(VarName)));
  } else if (VI = SyclctGlobalInfo::getInstance().findMemVarInfo(
                 getVarDecl(E))) {
    // Migrate the expr such as "const_one" to "const_one.get_ptr()".
    std::string VarName = VI->getName();
    VarName += ".get_ptr()";

    if (!OffsetFromBaseStr.empty()) {
      VarName =
          "(void *)((char *)(" + VarName + ") + " + OffsetFromBaseStr + ")";
    }
    emplaceTransformation(
        new ReplaceToken(E->getBeginLoc(), E->getEndLoc(), std::move(VarName)));
  } else {
    // Normal situation.
    insertAroundStmt(E, "(void*)(", ")");
  }
}

void MemoryTranslationRule::memcpyTranslation(
    const MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr) {
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

  std::string Name;
  if (ULExpr) {
    Name = ULExpr->getName().getAsString();
  } else {
    Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  }

  std::string ReplaceStr;
  if (Name == "cudaMemcpy") {
    ReplaceStr = "syclct::dpct_memcpy";
  } else {
    ReplaceStr = "syclct::async_dpct_memcpy";
  }

  if (ULExpr) {
    emplaceTransformation(new ReplaceToken(
        ULExpr->getBeginLoc(), ULExpr->getEndLoc(), std::move(ReplaceStr)));
  } else {
    emplaceTransformation(
        new ReplaceCalleeName(C, std::move(ReplaceStr), Name));
  }

  replaceMemAPIArg(C->getArg(0), Result);
  replaceMemAPIArg(C->getArg(1), Result);

  emplaceTransformation(
      new ReplaceStmt(C->getArg(3), std::move(DirectionName)));

  // cudaMemcpyAsync
  if (C->getNumArgs() == 5)
    handleAsync(C, 4, Result);
}

void MemoryTranslationRule::memcpyToAndFromSymbolTranslation(
    const MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr) {

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

  std::string Name;
  if (ULExpr) {
    Name = ULExpr->getName().getAsString();
  } else {
    Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  }

  std::string ReplaceStr;
  if (Name == "cudaMemcpyToSymbol" || Name == "cudaMemcpyFromSymbol") {
    ReplaceStr = "syclct::dpct_memcpy";
  } else {
    ReplaceStr = "syclct::async_dpct_memcpy";
  }

  if (ULExpr) {
    emplaceTransformation(new ReplaceToken(
        ULExpr->getBeginLoc(), ULExpr->getEndLoc(), std::move(ReplaceStr)));
  } else {
    emplaceTransformation(
        new ReplaceCalleeName(C, std::move(ReplaceStr), Name));
  }

  std::string OffsetFromBaseStr =
      getStmtSpelling(C->getArg(3), *Result.Context);

  if ((Name == "cudaMemcpyToSymbol" || Name == "cudaMemcpyToSymbolAsync") &&
      OffsetFromBaseStr != "0") {
    replaceMemAPIArg(C->getArg(0), Result, OffsetFromBaseStr);
  } else {
    replaceMemAPIArg(C->getArg(0), Result);
  }

  if ((Name == "cudaMemcpyFromSymbol" || Name == "cudaMemcpyFromSymbolAsync") &&
      OffsetFromBaseStr != "0") {
    replaceMemAPIArg(C->getArg(1), Result, OffsetFromBaseStr);
  } else {
    replaceMemAPIArg(C->getArg(1), Result);
  }

  // Remove C->getArg(3)
  const auto &SM = *Result.SourceManager;
  const Expr *ArgBefore = C->getArg(2);
  auto Begin = ArgBefore->getEndLoc();
  Begin = Lexer::getLocForEndOfToken(Begin, 0, SM, LangOptions());
  auto End = C->getArg(3)->getEndLoc();
  End = Lexer::getLocForEndOfToken(End, 0, SM, LangOptions());
  auto Length = SM.getFileOffset(End) - SM.getFileOffset(Begin);
  if (Length > 0) {
    emplaceTransformation(new ReplaceText(Begin, Length, ""));
  }

  emplaceTransformation(
      new ReplaceStmt(C->getArg(4), std::move(DirectionName)));

  // cudaMemcpyToSymbolAsync
  if (C->getNumArgs() == 6)
    handleAsync(C, 5, Result);
}

void MemoryTranslationRule::freeTranslation(
    const MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr) {

  std::string Name;
  if (ULExpr) {
    Name = ULExpr->getName().getAsString();
  } else {
    Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  }

  emplaceTransformation(new ReplaceCalleeName(C, "syclct::sycl_free", Name));
}

void MemoryTranslationRule::memsetTranslation(
    const MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr) {

  std::string Name;
  if (ULExpr) {
    Name = ULExpr->getName().getAsString();
  } else {
    Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  }

  std::string ReplaceStr;
  if (Name == "cudaMemsetAsync") {
    ReplaceStr = "syclct::async_dpct_memset";
  } else {
    ReplaceStr = "syclct::dpct_memset";
  }

  emplaceTransformation(new ReplaceCalleeName(C, std::move(ReplaceStr), Name));

  replaceMemAPIArg(C->getArg(0), Result);

  // cudaMemsetAsync
  if (C->getNumArgs() == 4)
    handleAsync(C, 3, Result);
}

// Memory migration rules live here.
void MemoryTranslationRule::registerMatcher(MatchFinder &MF) {
  auto memoryAPI = [&]() {
    return hasAnyName("cudaMalloc", "cudaMemcpy", "cudaMemcpyAsync",
                      "cudaMemcpyToSymbol", "cudaMemcpyToSymbolAsync",
                      "cudaMemcpyFromSymbol", "cudaMemcpyFromSymbolAsync",
                      "cudaFree", "cudaMemset", "cudaMemsetAsync", "cublasFree",
                      "cublasAlloc", "cudaGetSymbolAddress");
  };

  MF.addMatcher(callExpr(allOf(callee(functionDecl(memoryAPI())), parentStmt()))
                    .bind("call"),
                this);

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(memoryAPI())), unless(parentStmt())))
          .bind("callUsed"),
      this);

  MF.addMatcher(
      unresolvedLookupExpr(
          hasAnyDeclaration(namedDecl(memoryAPI())),
          hasParent(callExpr(unless(parentStmt())).bind("callExprUsed")))
          .bind("unresolvedCallUsed"),
      this);

  MF.addMatcher(
      unresolvedLookupExpr(hasAnyDeclaration(namedDecl(memoryAPI())),
                           hasParent(callExpr(parentStmt()).bind("callExpr")))
          .bind("unresolvedCall"),
      this);
}

void MemoryTranslationRule::run(const MatchFinder::MatchResult &Result) {
  auto TranslateCallExpr = [&](const CallExpr *C, const bool IsAssigned,
                               const UnresolvedLookupExpr *ULExpr = NULL) {
    if (!C)
      return;

    if (IsAssigned) {
      report(C->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP);
      insertAroundStmt(C, "(", ", 0)");
    }

    std::string Name;
    if (ULExpr && C) {
      Name = ULExpr->getName().getAsString();
    } else {
      Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
    }

    assert(TranslationDispatcher.find(Name) != TranslationDispatcher.end());
    TranslationDispatcher.at(Name)(Result, C, ULExpr);
  };

  TranslateCallExpr(getNodeAsType<CallExpr>(Result, "call"),
                    /* IsAssigned */ false);
  TranslateCallExpr(getNodeAsType<CallExpr>(Result, "callUsed"),
                    /* IsAssigned */ true);

  TranslateCallExpr(
      getNodeAsType<CallExpr>(Result, "callExprUsed"),
      /* IsAssigned */ true,
      getNodeAsType<UnresolvedLookupExpr>(Result, "unresolvedCallUsed"));
  TranslateCallExpr(
      getNodeAsType<CallExpr>(Result, "callExpr"),
      /* IsAssigned */ false,
      getNodeAsType<UnresolvedLookupExpr>(Result, "unresolvedCall"));
}

void MemoryTranslationRule::getSymbolAddressTranslation(
    const ast_matchers::MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr) {
  // Here only handle ordinary variable name reference, for accessing the
  // address of something residing on the device directly from host side should
  // not be possible.
  std::string Replacement;
  auto StmtStrArg0 = getStmtSpelling(C->getArg(0), *(Result.Context));
  auto StmtStrArg1 = getStmtSpelling(C->getArg(1), *(Result.Context));
  Replacement = "*(" + StmtStrArg0 + ")" + " = " + StmtStrArg1 + ".get_ptr()";
  emplaceTransformation(new ReplaceStmt(C, std::move(Replacement)));
}

MemoryTranslationRule::MemoryTranslationRule() {
  SetRuleProperty(ApplyToCudaFile | ApplyToCppFile);
  TranslationDispatcher["cudaMalloc"] = std::bind(
      &MemoryTranslationRule::mallocTranslation, this, std::placeholders::_1,
      std::placeholders::_2, std::placeholders::_3);
  TranslationDispatcher["cublasAlloc"] = std::bind(
      &MemoryTranslationRule::mallocTranslation, this, std::placeholders::_1,
      std::placeholders::_2, std::placeholders::_3);
  TranslationDispatcher["cudaMemcpy"] = std::bind(
      &MemoryTranslationRule::memcpyTranslation, this, std::placeholders::_1,
      std::placeholders::_2, std::placeholders::_3);
  TranslationDispatcher["cudaMemcpyAsync"] = std::bind(
      &MemoryTranslationRule::memcpyTranslation, this, std::placeholders::_1,
      std::placeholders::_2, std::placeholders::_3);
  TranslationDispatcher["cudaMemcpyToSymbol"] = std::bind(
      &MemoryTranslationRule::memcpyToAndFromSymbolTranslation, this,
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
  TranslationDispatcher["cudaMemcpyToSymbolAsync"] = std::bind(
      &MemoryTranslationRule::memcpyToAndFromSymbolTranslation, this,
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
  TranslationDispatcher["cudaMemcpyFromSymbol"] = std::bind(
      &MemoryTranslationRule::memcpyToAndFromSymbolTranslation, this,
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
  TranslationDispatcher["cudaMemcpyFromSymbolAsync"] = std::bind(
      &MemoryTranslationRule::memcpyToAndFromSymbolTranslation, this,
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
  TranslationDispatcher["cudaFree"] = std::bind(
      &MemoryTranslationRule::freeTranslation, this, std::placeholders::_1,
      std::placeholders::_2, std::placeholders::_3);
  TranslationDispatcher["cublasFree"] = std::bind(
      &MemoryTranslationRule::freeTranslation, this, std::placeholders::_1,
      std::placeholders::_2, std::placeholders::_3);
  TranslationDispatcher["cudaMemset"] = std::bind(
      &MemoryTranslationRule::memsetTranslation, this, std::placeholders::_1,
      std::placeholders::_2, std::placeholders::_3);
  TranslationDispatcher["cudaMemsetAsync"] = std::bind(
      &MemoryTranslationRule::memsetTranslation, this, std::placeholders::_1,
      std::placeholders::_2, std::placeholders::_3);
  TranslationDispatcher["cudaGetSymbolAddress"] = std::bind(
      &MemoryTranslationRule::getSymbolAddressTranslation, this,
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
}

void MemoryTranslationRule::handleAsync(
    const CallExpr *C, unsigned i, const MatchFinder::MatchResult &Result) {
  const Expr *Stream = C->getArg(i);
  if (Stream) {
    auto StreamStr = getStmtSpelling(Stream, *Result.Context);
    // Remove the default stream argument "0"
    if (StreamStr == "0") {
      // Remove preceding semicolon and spaces
      if (i) {
        const auto &SM = *Result.SourceManager;
        const Expr *ArgBefore = C->getArg(i - 1);
        auto Begin = ArgBefore->getEndLoc();
        Begin = Lexer::getLocForEndOfToken(Begin, 0, SM, LangOptions());
        auto End = Stream->getBeginLoc();
        auto Length = SM.getFileOffset(End) - SM.getFileOffset(Begin);
        emplaceTransformation(new ReplaceText(Begin, Length, ""));
      }
      emplaceTransformation(new ReplaceStmt(Stream, ""));
    }
  }
}

REGISTER_RULE(MemoryTranslationRule)

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
  std::vector<std::string> MathFunctions = {
#define ENTRY_RENAMED(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_EMULATED(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_OPERATOR(APINAME, OPKIND) APINAME,
#define ENTRY_TYPECAST(APINAME) APINAME,
#define ENTRY_UNSUPPORTED(APINAME) APINAME,
#include "APINamesMath.inc"
#undef ENTRY_RENAMED
#undef ENTRY_EMULATED
#undef ENTRY_OPERATOR
#undef ENTRY_TYPECAST
#undef ENTRY_UNSUPPORTED
  };

  MF.addMatcher(
      callExpr(callee(functionDecl(
                   internal::Matcher<NamedDecl>(
                       new internal::HasNameMatcher(MathFunctions)),
                   anyOf(unless(hasDeclContext(namespaceDecl(anything()))),
                         hasDeclContext(namespaceDecl(hasName("std")))))),
               unless(hasAncestor(
                   cxxConstructExpr(hasType(typedefDecl(hasName("dim3")))))))
          .bind("math"),
      this);
}

void MathFunctionsRule::run(const MatchFinder::MatchResult &Result) {
  if (auto CE = getNodeAsType<CallExpr>(Result, "math")) {
    ExprAnalysis EA(CE);
    EA.analyze();
    emplaceTransformation(EA.getReplacement());
    if (auto TM = getCmathHeaderInsertTextForCallExpr(CE, Result.SourceManager))
      emplaceTransformation(TM);
  }
}

REGISTER_RULE(MathFunctionsRule)

void WarpFunctionsRule::registerMatcher(MatchFinder &MF) {
  std::vector<std::string> WarpFunctions = {
#define ENTRY_WARP(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#include "APINamesWarp.inc"
#undef ENTRY_WARP
  };

  MF.addMatcher(callExpr(callee(functionDecl(internal::Matcher<NamedDecl>(
                             new internal::HasNameMatcher(WarpFunctions)))),
                         hasAncestor(functionDecl().bind("ancestor")))
                    .bind("warp"),
                this);
}

void WarpFunctionsRule::run(const MatchFinder::MatchResult &Result) {
  if (auto FD = getAssistNodeAsType<FunctionDecl>(Result, "ancestor"))
    DeviceFunctionDecl::LinkRedecls(FD)->setItem();

  if (auto CE = getNodeAsType<CallExpr>(Result, "warp")) {
    ExprAnalysis EA(CE);
    EA.analyze();
    emplaceTransformation(EA.getReplacement());
  }
}
REGISTER_RULE(WarpFunctionsRule)

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

  if (!Namespace.empty() && Namespace == "thrust") {
    APIName = Namespace + "::" + APIName;
  }

  SrcAPIStaticsMap[GetFunctionSignature(C->getCalleeDecl()->getAsFunction())]++;

  if (!TranslationStatistics::IsTranslated(APIName)) {

    const SourceManager &SM = (*Result.Context).getSourceManager();
    const SourceLocation FileLoc = SM.getFileLoc(C->getBeginLoc());

    std::string SLStr = FileLoc.printToString(SM);

    std::size_t PosCol = SLStr.rfind(':');
    std::size_t PosRow = SLStr.rfind(':', PosCol - 1);
    std::string FileName = SLStr.substr(0, PosRow);
    LOCStaticsMap[FileName][2]++;
    report(C->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, APIName.c_str());
  }
}

REGISTER_RULE(RecognizeAPINameRule)

const BinaryOperator *TextureRule::getParentAsAssignedBO(const Expr *E,
                                                         ASTContext &Context) {
  auto Parents = Context.getParents(*E);
  if (Parents.size() > 0)
    return getAssignedBO(Parents[0].get<Expr>(), Context);
  return nullptr;
}

// Return the binary operator if E is the lhs of an assign experssion, otherwise
// nullptr.
const BinaryOperator *TextureRule::getAssignedBO(const Expr *E,
                                                 ASTContext &Context) {
  if (dyn_cast<MemberExpr>(E)) {
    // Continue finding parents when E is MemberExpr.
    return getParentAsAssignedBO(E, Context);
  } else if (auto ICE = dyn_cast<ImplicitCastExpr>(E)) {
    // Stop finding parents and return nullptr when E is ImplicitCastExpr,
    // except for ArrayToPointerDecay cast.
    if (ICE->getCastKind() == CK_ArrayToPointerDecay) {
      return getParentAsAssignedBO(E, Context);
    }
  } else if (auto ASE = dyn_cast<ArraySubscriptExpr>(E)) {
    // Continue finding parents when E is ArraySubscriptExpr, and remove
    // subscript operator anyway for texture object's member.
    emplaceTransformation(new ReplaceToken(
        Lexer::getLocForEndOfToken(ASE->getLHS()->getEndLoc(), 0,
                                   Context.getSourceManager(),
                                   Context.getLangOpts()),
        ASE->getRBracketLoc(), ""));
    return getParentAsAssignedBO(E, Context);
  } else if (auto BO = dyn_cast<BinaryOperator>(E)) {
    // If E is BinaryOperator, return E only when it is assign expression,
    // otherwise return nullptr.
    if (BO->getOpcode() == BO_Assign)
      return BO;
  }
  return nullptr;
}

void TextureRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      declRefExpr(
          hasDeclaration(
              varDecl(
                  hasType(templateSpecializationType(hasDeclaration(
                      classTemplateSpecializationDecl(hasName("texture"))))))
                  .bind("texDecl")),
          // Match texture object's declaration
          anyOf(hasAncestor(functionDecl(anyOf(hasAttr(attr::CUDADevice),
                                               hasAttr(attr::CUDAGlobal)))
                                .bind("texFunc")),
                // Match the __globla__/__device__ functions inside which
                // texture object is referenced
                hasAncestor(memberExpr().bind("texMember")),
                // Match the MemberExpr if member of texture is used
                anything()) // Make this matcher available whether it has
                            // ancestors as before
          )
          .bind("tex"),
      this);
  MF.addMatcher(
      varDecl(hasType(pointsTo(namedDecl(hasName("cudaArray"))))).bind("array"),
      this);
  MF.addMatcher(varDecl(hasType(recordDecl(hasName("cudaChannelFormatDesc"))))
                    .bind("chnDecl"),
                this);
  MF.addMatcher(callExpr(callee(functionDecl(hasAnyName(
                             "cudaCreateChannelDesc", "cudaUnbindTexture",
                             "cudaFreeArray", "cudaMallocArray",
                             "cudaMemcpyToArray", "cudaBindTextureToArray",
                             "tex1D", "tex2D", "tex3D", "tex1Dfetch"))))
                    .bind("call"),
                this);
}

void TextureRule::run(const MatchFinder::MatchResult &Result) {
  if (auto VD = getAssistNodeAsType<VarDecl>(Result, "texDecl")) {
    auto Tex = SyclctGlobalInfo::getInstance().insertTextureInfo(VD);
    emplaceTransformation(new ReplaceVarDecl(VD, Tex->getDeclReplacement()));
    if (auto FD = getAssistNodeAsType<FunctionDecl>(Result, "texFunc")) {
      DeviceFunctionDecl::LinkRedecls(FD)->addTexture(Tex);
    }
    if (auto ME = getNodeAsType<MemberExpr>(Result, "texMember")) {
      auto Field = MapNames::findReplacedName(
          TextureMemberNames, ME->getMemberNameInfo().getAsString());
      if (auto BO = getAssignedBO(ME, *Result.Context)) {
        // e.g.: t.filterMode = cudaFilterModeLinear =>
        // t.set_filter_mode(cl::sycl::filtering_mode::linear)
        /// Add rename field with "set_" prefix.
        emplaceTransformation(new RenameFieldInMemberExpr(ME, "set_" + Field));
        /// Remove operator "=".
        emplaceTransformation(new ReplaceToken(
            Lexer::getLocForEndOfToken(BO->getLHS()->getEndLoc(), 0,
                                       *Result.SourceManager,
                                       Result.Context->getLangOpts()),
            BO->getRHS()->getBeginLoc().getLocWithOffset(-1), "("));
        /// Insert ")" at the end of the expression.
        emplaceTransformation(new InsertAfterStmt(BO, ")"));
        if (auto DRE = dyn_cast<DeclRefExpr>(BO->getRHS()->IgnoreImpCasts())) {
          /// Replace rhs of BinaryOperator if it is a EnumConstantDecl
          if (auto Enum = dyn_cast<EnumConstantDecl>(DRE->getDecl())) {
            emplaceTransformation(new ReplaceStmt(
                BO->getRHS(),
                MapNames::findReplacedName(EnumConstantRule::EnumNamesMap,
                                           Enum->getName())));
          }
        }
      } else {
        emplaceTransformation(
            new RenameFieldInMemberExpr(ME, buildString("get_", Field, "()")));
      }
    }
  } else if (auto VD = getNodeAsType<VarDecl>(Result, "array")) {
    std::string ReplType = "syclct::syclct_array";
    if (VD->getType()->getTypeClass() == Type::Pointer)
      ReplType += " ";
    emplaceTransformation(new ReplaceTypeInDecl(VD, std::move(ReplType)));
  } else if (auto VD = getNodeAsType<VarDecl>(Result, "chnDecl")) {
    const std::string &ReplType = MapNames::findReplacedName(
        MapNames::TypeNamesMap, VD->getType().getUnqualifiedType().getAsString(
                                    Result.Context->getLangOpts()));
    emplaceTransformation(new ReplaceTypeInDecl(VD, std::string(ReplType)));
  } else if (auto CE = getNodeAsType<CallExpr>(Result, "call")) {
    ExprAnalysis A;
    A.analyze(CE);
    emplaceTransformation(A.getReplacement());
  }
}

REGISTER_RULE(TextureRule)

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
