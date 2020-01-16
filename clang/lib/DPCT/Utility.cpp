//===--- Utility.cpp -----------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2020 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//

#include "Utility.h"

#include "AnalysisInfo.h"
#include "Debug.h"
#include "SaveNewFiles.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <algorithm>

using namespace llvm;
using namespace clang;
using namespace std;

namespace path = llvm::sys::path;
namespace fs = llvm::sys::fs;

bool makeCanonical(SmallVectorImpl<char> &Path) {
  if (fs::make_absolute(Path) != std::error_code()) {
    llvm::errs() << "Could not get absolute path from '" << Path << "'\n ";
    return false;
  }
  path::remove_dots(Path, /* remove_dot_dot= */ true);
  return true;
}

bool makeCanonical(string &PathPar) {
  SmallString<256> Path = StringRef(PathPar);
  if (!makeCanonical(Path))
    return false;
  PathPar.assign(begin(Path), end(Path));
  return true;
}

bool isCanonical(StringRef Path) {
  bool HasNoDots = all_of(path::begin(Path), path::end(Path),
                          [](StringRef e) { return e != "." && e != ".."; });
  return HasNoDots && path::is_absolute(Path);
}

const char *getNL(void) {
#if defined(__linux__)
  return "\n";
#elif defined(_WIN64)
  return "\r\n";
#else
#error Only support windows and Linux.
#endif
}

const char *getNL(SourceLocation Loc, const SourceManager &SM) {
  auto LocInfo = SM.getDecomposedLoc(Loc);
  auto Buffer = SM.getBufferData(LocInfo.first);
  Buffer = Buffer.data() + LocInfo.second;
  // Search for both to avoid searching till end of file.
  auto pos = Buffer.find_first_of("\r\n");
  if (pos == StringRef::npos || Buffer[pos] == '\n')
    return "\n";
  else
    return "\r\n";
}

StringRef getIndent(SourceLocation Loc, const SourceManager &SM) {
  auto LocInfo = SM.getDecomposedLoc(Loc);
  auto Buffer = SM.getBufferData(LocInfo.first);
  // Find start of indentation.
  auto begin = Buffer.find_last_of('\n', LocInfo.second);
  if (begin != StringRef::npos) {
    ++begin;
  } else {
    // We're at the beginning of the file.
    begin = 0;
  }
  auto end = Buffer.find_if([](char c) { return !isspace(c); }, begin);
  return Buffer.substr(begin, end - begin);
}

// Get textual representation of the Stmt.
std::string getStmtSpelling(const Stmt *S, const ASTContext &Context) {
  std::string Str;
  if(!S)
    return Str;
  auto &SM = Context.getSourceManager();
  SourceLocation BeginLoc, EndLoc;
  if (SM.isMacroArgExpansion(S->getBeginLoc())) {
    BeginLoc = SM.getImmediateSpellingLoc(S->getBeginLoc());
    EndLoc = SM.getImmediateSpellingLoc(S->getEndLoc());
    if (EndLoc.isMacroID()) {
      // if the immediate spelling location of
      // a macro arg is another macro, get the expansion loc
      EndLoc = SM.getExpansionLoc(EndLoc);
    }
    if (BeginLoc.isMacroID()) {
      // if the immediate spelling location of
      // a macro arg is another macro, get the expansion loc
      BeginLoc = SM.getExpansionLoc(BeginLoc);
    }
  } else {
    BeginLoc = SM.getExpansionLoc(S->getBeginLoc());
    EndLoc = SM.getExpansionLoc(S->getEndLoc());
  }

  int Length = SM.getFileOffset(EndLoc) - SM.getFileOffset(BeginLoc) +
               Lexer::MeasureTokenLength(EndLoc, SM, Context.getLangOpts());
  Str = std::string(SM.getCharacterData(BeginLoc), Length);
  return Str;
}

std::string getStmtExpansion(const Stmt *S, const ASTContext &Context) {
  const SourceManager &SM = Context.getSourceManager();
  SourceLocation Begin(S->getBeginLoc()), _End(S->getEndLoc());
  SourceLocation End(Lexer::getLocForEndOfToken(_End, 0, SM, LangOptions()));
  if (Begin.isMacroID())
    Begin = SM.getExpansionLoc(Begin);
  if (End.isMacroID())
    End = SM.getExpansionLoc(End);
  return std::string(SM.getCharacterData(Begin),
                     SM.getCharacterData(End) - SM.getCharacterData(Begin));
}

SourceProcessType GetSourceFileType(llvm::StringRef SourcePath) {
  SmallString<256> FilePath = SourcePath;
  auto Extension = path::extension(FilePath);

  if (Extension == ".cu") {
    return TypeCudaSource;
  } else if (Extension == ".cuh") {
    return TypeCudaHeader;
  } else if (Extension == ".cpp" || Extension == ".cxx" || Extension == ".cc" ||
             Extension == ".c" || Extension == ".C") {
    return TypeCppSource;
  } else if (Extension == ".hpp" || Extension == ".hxx" || Extension == ".h" ||
             Extension == ".hh" || Extension == ".inl"  || Extension == ".inc" ||
             Extension == ".INL" || Extension == ".INC" ||
             Extension == ".TPP" ||Extension == ".tpp") {
    return TypeCppHeader;
  } else {
    std::string ErrMsg =
        "[ERROR] Not support \"" + Extension.str() + "\" file type!\n";
    dpct::PrintMsg(ErrMsg);
    std::exit(MigrationErrorNotSupportFileType);
  }
}

std::vector<std::string>
ruleTopoSort(std::vector<std::vector<std::string>> &TableRules) {
  std::vector<std::string> Vec;

  std::vector<std::list<int>> AdjacencyList;
  std::vector<int> InDegree;
  std::stack<int> Stack;
  std::vector<std::string> RuleNames;

  int n = TableRules.size();
  AdjacencyList.assign(n, std::list<int>());
  InDegree.assign(n, 0);

  for (int i = 0; i < n; i++) {
    RuleNames.push_back(TableRules[i].at(0));
  }

  for (int i = 0; i < n; i++) {
    for (std::vector<std::string>::iterator it = TableRules[i].begin() + 1;
         it != TableRules[i].end(); ++it) {

      // if detect rule depend on itself,  then just ignore
      if (*it == *TableRules[i].begin()) {
        continue;
      }

      std::vector<std::string>::iterator index =
          find(RuleNames.begin(), RuleNames.end(), *it);
      if (index != RuleNames.end()) {
        AdjacencyList[i].push_back(index - RuleNames.begin());
        InDegree[index - RuleNames.begin()]++;
      }
    }
  }

  for (int i = 0; i < n; i++)
    if (InDegree[i] == 0)
      Stack.push(i);

  while (!Stack.empty()) {
    int v = Stack.top();
    Stack.pop();
    InDegree[v] = -1;

    for (std::list<int>::iterator it = AdjacencyList[v].begin();
         it != AdjacencyList[v].end(); it++) {
      InDegree[*it]--;
      if (InDegree[*it] == 0)
        Stack.push(*it);
    }
    AdjacencyList[v].clear();
    Vec.push_back(RuleNames[v]);
  }
  if (Vec.size() != InDegree.size()) {
    std::cout << "Error: Two rules have dependency on each other！\n";
    dpct::DebugInfo::ShowStatus(MigrationError);
    exit(MigrationError);
  }

  return Vec;
}

const std::string SpacesForStatement = "        "; // Eight spaces
const std::string SpacesForArg = "        ";       // Eight spaces

const std::string &getFmtEndStatement(void) {
  const static std::string EndStatement = ";\n";
  return EndStatement;
}

const std::string &getFmtStatementIndent(std::string &BaseIndent) {
  const static std::string FmtStatementIndent = BaseIndent + SpacesForStatement;
  return FmtStatementIndent;
}

const std::string &getFmtEndArg(void) {
  const static std::string EndArg = ",\n";
  return EndArg;
}

const std::string &getFmtArgIndent(std::string &BaseIndent) {
  const static std::string FmtArgIndent = BaseIndent + SpacesForArg;
  return FmtArgIndent;
}

std::vector<std::string> split(const std::string &Str, char Delim) {
  std::vector<std::string> V;
  std::stringstream S(Str);
  std::string Token;
  while (std::getline(S, Token, Delim))
    V.push_back(Token);

  return V;
}

/// Find the innermost (closest) block (CompoundStmt) where S is located
const clang::CompoundStmt *findImmediateBlock(const clang::Stmt *S) {
  if (!S)
    return nullptr;

  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*S);
  while (Parents.size() == 1) {
    auto *Parent = Parents[0].get<Stmt>();
    if (Parent) {
      if (Parent->getStmtClass() == Stmt::StmtClass::CompoundStmtClass)
        return dyn_cast<CompoundStmt>(Parent);
      Parents = Context.getParents(*Parent);
    } else {
      Parents = Context.getParents(Parents[0]);
    }
  }

  return nullptr;
}

// A worklist-based BFS algorithm to find the innermost (closest) block
// where D is located
const clang::CompoundStmt *findImmediateBlock(const ValueDecl *D) {
  if (!D)
    return nullptr;

  // CS points to the CompoundStmt that is the body of the belonging function
  const CompoundStmt *CS = nullptr;
  if (D->getDeclContext()->getDeclKind() == Decl::Kind::Block) {
    auto BD = static_cast<const BlockDecl *>(D->getDeclContext());
    CS = BD->getCompoundBody();
  } else if (D->getLexicalDeclContext()->getDeclKind() ==
             Decl::Kind::Function) {
    auto BD = static_cast<const FunctionDecl *>(D->getDeclContext());
    CS = dyn_cast<CompoundStmt>(BD->getBody());
  }

  // Worklist
  std::deque<const CompoundStmt *> WL;
  WL.push_back(CS);

  while (!WL.empty()) {
    const CompoundStmt *CS = WL.front();
    WL.pop_front();
    for (auto Iter = CS->body_begin(); Iter != CS->body_end(); ++Iter) {
      // For a DeclStmt, check if TypeName and ArgName match
      if ((*Iter)->getStmtClass() == Stmt::StmtClass::DeclStmtClass) {
        DeclStmt *DS = dyn_cast<DeclStmt>(*Iter);
        for (auto It = DS->decl_begin(); It != DS->decl_end(); ++It) {
          VarDecl *VD = dyn_cast<VarDecl>(*It);
          if (VD == D)
            return CS;
        }
      }
      // Add nested CompoundStmt to the worklist for later search, BFS
      else if ((*Iter)->getStmtClass() == Stmt::StmtClass::CompoundStmtClass) {
        const CompoundStmt *CS = dyn_cast<CompoundStmt>(*Iter);
        WL.push_back(CS);
      }
    }
  }

  return nullptr;
}

const clang::FunctionDecl *getImmediateOuterFuncDecl(const clang::Stmt *S) {
  if (!S)
    return nullptr;

  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*S);
  while (Parents.size() == 1) {
    if (auto *Parent = Parents[0].get<Decl>())
      if (auto FD = dyn_cast<clang::FunctionDecl>(Parent))
        return FD;

    Parents = Context.getParents(Parents[0]);
  }

  return nullptr;
}

bool callingFuncHasDeviceAttr(const CallExpr *CE) {
  auto FD = getImmediateOuterFuncDecl(CE);
  return FD && FD->hasAttr<CUDADeviceAttr>();
}

// Determine if a Stmt and a ValueDecl are in the same scope
bool isInSameScope(const Stmt *S, const ValueDecl *D) {
  if (!S || !D)
    return false;

  // Find the innermost block of D and S
  const auto *CS1 = findImmediateBlock(D);
  const auto *CS2 = findImmediateBlock(S);

  if (!CS1 || !CS2)
    return false;

  return CS1 == CS2;
}

// Iteratively get the inner ValueDecl of a potetionally nested expression
// with implicit casts
const DeclRefExpr *getInnerValueDecl(const Expr *Arg) {
  auto DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreImpCasts());
  while (!DRE) {
    if (auto UO = dyn_cast<UnaryOperator>(Arg->IgnoreImpCasts()))
      Arg = UO->getSubExpr();
    else
      return nullptr;
    DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreImpCasts());
  }
  return DRE;
}

// Check if a string starts with the prefix
bool startsWith(const std::string &Str, const std::string &Prefix) {
  return Prefix.size() <= Str.size() &&
         std::equal(Prefix.begin(), Prefix.end(), Str.begin());
}

bool startsWith(const std::string &Str, char C) {
  return Str.size() && Str[0] == C;
}

// Check if a string ends with the suffix
bool endsWith(const std::string &Str, const std::string &Suffix) {
  return Suffix.size() <= Str.size() &&
         std::equal(Suffix.rbegin(), Suffix.rend(), Str.rbegin());
}

bool endsWith(const std::string &Str, char C) {
  return Str.size() && Str[Str.size() - 1] == C;
}

const clang::Stmt *getParentStmt(const clang::Stmt *S) {
  if (!S)
    return nullptr;

  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*S);
  assert(Parents.size() == 1);
  if (Parents.size() == 1)
    return Parents[0].get<Stmt>();

  return nullptr;
}

// Determine if S is a single line statement inside
// a if/while/do while/for statement
bool IsSingleLineStatement(const clang::Stmt *S) {
  auto ParentStmt = getParentStmt(S);
  if (!ParentStmt)
    return false;

  auto ParentStmtClass = ParentStmt->getStmtClass();
  return ParentStmtClass == Stmt::StmtClass::IfStmtClass ||
         ParentStmtClass == Stmt::StmtClass::WhileStmtClass ||
         ParentStmtClass == Stmt::StmtClass::DoStmtClass ||
         ParentStmtClass == Stmt::StmtClass::ForStmtClass;
}

// Find the nearest non-Expr non-Decl ancestor node of Expr E
// Assumes: E != nullptr
const ast_type_traits::DynTypedNode
findNearestNonExprNonDeclAncestorNode(const clang::Expr *E) {
  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto ParentNodes = Context.getParents(*E);
  ast_type_traits::DynTypedNode LastNode, ParentNode;
  while (!ParentNodes.empty()) {
    ParentNode = ParentNodes[0];
    if (!ParentNode.get<Expr>() && !ParentNode.get<Decl>() &&
        !ParentNode.getSourceRange().getBegin().isMacroID()) {
      break;
    }
    LastNode = ParentNode;
    ParentNodes = Context.getParents(LastNode);
  }
  return LastNode;
}

// Find the nearest non-Expr non-Decl ancestor statement of Expr E
// Assumes: E != nullptr
const clang::Stmt *findNearestNonExprNonDeclAncestorStmt(const clang::Expr *E) {
  return findNearestNonExprNonDeclAncestorNode(E).get<Stmt>();
}

SourceRange getScopeInsertRange(const MemberExpr *ME) {
  return getScopeInsertRange(ME, ME->getBeginLoc(), ME->getEndLoc());
}

SourceRange getScopeInsertRange(const Expr *E,
                                const SourceLocation &FuncNameBegin,
                                const SourceLocation &FuncCallEnd) {
  SourceLocation StmtBegin, StmtEndAfterSemi;
  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  auto ParentNode = Context.getParents(*E);
  ast_type_traits::DynTypedNode LastNode;
  SourceLocation StmtEnd;
  if (ParentNode.empty()) {
    StmtBegin = FuncNameBegin;
    StmtEnd = FuncCallEnd;
  } else if (!ParentNode[0].get<Expr>() && !ParentNode[0].get<Decl>()) {
    StmtBegin = FuncNameBegin;
    StmtEnd = FuncCallEnd;
  } else {
    auto AncestorStmt = findNearestNonExprNonDeclAncestorNode(E);
    StmtBegin = AncestorStmt.getSourceRange().getBegin();
    StmtEnd = AncestorStmt.getSourceRange().getEnd();
    if (StmtBegin.isMacroID())
      StmtBegin = SM.getExpansionLoc(StmtBegin);
    if (StmtEnd.isMacroID())
      StmtEnd = SM.getExpansionLoc(StmtEnd);
  }

  Optional<Token> TokSharedPtr;
  TokSharedPtr = Lexer::findNextToken(StmtEnd, SM, LangOptions());
  Token TokSemi = TokSharedPtr.getValue();
  StmtEndAfterSemi = TokSemi.getEndLoc();
  return {StmtBegin, StmtEndAfterSemi};
}

std::string getCanonicalPath(SourceLocation Loc) {
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  std::string Path = SM.getFilename(SM.getExpansionLoc(Loc));
  makeCanonical(Path);
  return Path;
}

bool containOnlyDigits(const std::string &str) {
  return std::all_of(str.begin(), str.end(), ::isdigit);
}

void replaceSubStr(std::string &Str, const std::string &SubStr,
                   const std::string &Repl) {
  auto P = Str.find(SubStr);
  if (P != std::string::npos)
    Str.replace(P, SubStr.size(), Repl);
}
void replaceSubStrAll(std::string &Str, const std::string &SubStr,
                      const std::string &Repl) {
  auto P = Str.find(SubStr);
  while (P != std::string::npos) {
    Str.replace(P, SubStr.size(), Repl);
    P = Str.find(SubStr);
  }
}

/// Get the immediate ancestor with type \tparam T of \param S
template <typename T> const T *getImmediateAncestor(const Stmt *S) {
  if (!S)
    return nullptr;

  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*S);
  while (Parents.size() == 1) {
    if (auto *Parent = Parents[0].get<T>()) {
      return Parent;
    } else {
      Parents = Context.getParents(Parents[0]);
    }
  }

  return nullptr;
}

/// Find the FunctionDecl where \param S is located
const FunctionDecl *getFunctionDecl(const Stmt *S) {
  return getImmediateAncestor<FunctionDecl>(S);
}

/// Get the CallExpr where \param S is referenced
const CallExpr *getCallExpr(const Stmt *S) {
  return getImmediateAncestor<CallExpr>(S);
}

/// Check if \param E is an expr that loads the address of \param DRE,
/// ignoring any casts and parens.
bool isAddressOfExpr(const Expr *E, const DeclRefExpr *DRE) {
  E = E->IgnoreCasts()->IgnoreParens();
  if (auto UO = dyn_cast<UnaryOperator>(E))
    if (UO->getOpcode() == UO_AddrOf)
      if (auto DRE2 = dyn_cast<DeclRefExpr>(UO->getSubExpr()))
        if (DRE->getDecl() == DRE2->getDecl())
          return true;
  return false;
}

/// Check if \param CE allocates memory pointed to by \param Arg
bool isCudaMemoryAllocation(const DeclRefExpr *Arg, const CallExpr *CE) {
  auto FD = CE->getDirectCallee();
  if (!FD)
    return false;
  auto FuncName = FD->getNameAsString();
  if (FuncName == "cudaMalloc" || FuncName == "cudaMallocPitch") {
    if (!CE->getNumArgs())
      return false;
    if (isAddressOfExpr(CE->getArg(0), Arg))
      return true;
  }
  return false;
}

/// This function traverses all the nodes in the AST represented by \param Root
/// in a depth-first manner, until the node \param Sentinal is reached, to check
/// if the pointer \param Arg to a piece of memory is used as lvalue after the
/// most recent memory allocation until \param Sentinal.
///
/// \param Arg: the expr that represents a reference to a declared variable
/// \param Root: the root of an AST
/// \param Sentinal: the sentinal node indicating termination of traversal
/// \param CurrentScope: the current scope of searching
/// \param UsedInScope: the map recording used-as-lavlue status for all scopes
/// \param Done: if current searching should stop or not
///
/// devPtr (T *) can be initialized in the following ways:
///   1. cudaMalloc(&devPtr, size);
///   2. cudaMallocPitch(&devPtr, pitch, width, height);
/// where "&devPtr" can be surrounded by arbitrary number of cast or paren
/// expressions.
/// If a new allocation happens on the memory pointed to by devPtr, \Used is
/// reset to false.
///
/// devPtr (T *) can be used as lvalue in the various ways:
///   1. devPtr = devPtr + 1;
///   2. devPtr = devPtr - 1;
///   3. devPtr += 1;
///   4. devPtr -= 1;
///   5. mod(&devPtr); // void mod(int **);
///   6. mod(devPtr);  // void mod(int *&);
///   ...
/// In a Clang AST, \param Arg is judged of used-as-lvalue when it is not under
/// a LValueToRValue cast node in the AST, which covers all the above cases.
/// Each used-as-lvalue scenario sets \param Used to true.
///
/// If the memory is never seen to be allocated in the traversing process,
/// \param Used is conservatively treated as true.
void findUsedAsLvalue(const DeclRefExpr *Arg, const Stmt *Root,
                      const Stmt *Sentinal,
                      std::vector<const Stmt *> &CurrentScope,
                      std::map<std::vector<const Stmt *>, bool> &UsedInScope,
                      bool &Done) {
  // Done with searching when Sentinal is reached.
  if (!Arg || !Root || !Sentinal)
    return;
  if (Root == Sentinal) {
    Done = true;
    return;
  }

  if (auto DRE = dyn_cast<DeclRefExpr>(Root)) {
    if (DRE->getType()->isPointerType()) {
      if (DRE->getDecl() != Arg->getDecl())
        return;
      if (auto *Parent = getParentStmt(DRE))
        if (auto *ICE = dyn_cast<ImplicitCastExpr>(Parent))
          if (ICE->getCastKind() == CK_LValueToRValue)
            return;
      // Arg is used as lvalue
      UsedInScope[CurrentScope] = true;
    }
  } else if (auto CE = dyn_cast<CallExpr>(Root)) {
    if (isCudaMemoryAllocation(Arg, CE))
      UsedInScope[CurrentScope] = false;
    else
      for (auto It = CE->arg_begin(); !Done && It != CE->arg_end(); ++It)
        findUsedAsLvalue(Arg, *It, Sentinal, CurrentScope, UsedInScope, Done);
  } else if (auto IS = dyn_cast<IfStmt>(Root)) {
    // Condition
    findUsedAsLvalue(Arg, IS->getCond(), Sentinal, CurrentScope, UsedInScope, Done);
    if (Done)
      return;
    bool Used = UsedInScope[CurrentScope];

    // Then branch
    CurrentScope.push_back(IS->getThen());
    UsedInScope[CurrentScope] = Used;
    findUsedAsLvalue(Arg, IS->getThen(), Sentinal, CurrentScope, UsedInScope, Done);
    if (Done)
      return;
    CurrentScope.pop_back();

    // Else branch
    if (auto ElseBranch = IS->getElse()) {
      CurrentScope.push_back(ElseBranch);
      UsedInScope[CurrentScope] = Used;
      findUsedAsLvalue(Arg, ElseBranch, Sentinal, CurrentScope, UsedInScope, Done);
      if (Done)
        return;
      CurrentScope.pop_back();
    }
  } else if (auto WS = dyn_cast<WhileStmt>(Root)) {
    // Condition
    findUsedAsLvalue(Arg, WS->getCond(), Sentinal, CurrentScope, UsedInScope, Done);
    if (Done)
      return;

    // Body
    bool Used = UsedInScope[CurrentScope];
    CurrentScope.push_back(WS->getBody());
    UsedInScope[CurrentScope] = Used;
    findUsedAsLvalue(Arg, WS->getBody(), Sentinal, CurrentScope, UsedInScope, Done);
    if (Done)
      return;
    CurrentScope.pop_back();
  } else if (auto FS = dyn_cast<ForStmt>(Root)) {
    // Initilization
    findUsedAsLvalue(Arg, FS->getInit(), Sentinal, CurrentScope, UsedInScope, Done);
    if (Done)
      return;
    // Condition
    findUsedAsLvalue(Arg, FS->getCond(), Sentinal, CurrentScope, UsedInScope, Done);
    if (Done)
      return;
    // Increment
    findUsedAsLvalue(Arg, FS->getInc(), Sentinal, CurrentScope, UsedInScope, Done);
    if (Done)
      return;

    // Body
    bool Used = UsedInScope[CurrentScope];
    CurrentScope.push_back(FS->getBody());
    UsedInScope[CurrentScope] = Used;
    findUsedAsLvalue(Arg, FS->getBody(), Sentinal, CurrentScope, UsedInScope, Done);
    if (Done)
      return;
    CurrentScope.pop_back();
  } else {
    // Finishes when Sentinal is reached or we're done searching current
    // children nodes
    for (auto It = Root->child_begin(); !Done && It != Root->child_end(); ++It)
      findUsedAsLvalue(Arg, *It, Sentinal, CurrentScope, UsedInScope, Done);
  }
}

/// This function checks if the pointer \param Arg to a piece of memory is used
/// as lvalue after the memory is allocated, until \param S in the same
/// function. If the memory not allocated before \param S in the same function,
/// \param Arg is considered used-as-lvalue before \param S.
bool isArgUsedAsLvalueUntil(const DeclRefExpr *Arg, const Stmt *S) {
  // Global variables are always treated as used-as-lvalue
  if (Arg->getDecl()->isDefinedOutsideFunctionOrMethod())
    return true;

  auto *FD = getFunctionDecl(S);
  if (!FD)
    return true;

  auto *CS = FD->getBody();

  std::vector<const Stmt *> CurrentScope{CS};
  // If \param Arg is used as lvalue before \param S in the scope
  std::map<std::vector<const Stmt *>, bool> UsedInScope;
  UsedInScope[CurrentScope] = true;

  // If we are done with searching (\param S has been reached)
  bool Done = false;

  // Traverse from the function body
  findUsedAsLvalue(Arg, CS, S, CurrentScope, UsedInScope, Done);

  return UsedInScope[CurrentScope];
}

/// This function gets the length from current token begin to next token begin.
/// But if there is any line break before next token, then the length will be
/// from current token begin to the line break.
/// \param CurTok Current token.
/// \param SM SourceManager.
/// \return The result length.
unsigned int getLenToNextTokenBegin(const Token &CurTok, SourceManager &SM) {
  SourceLocation CurTokBegin = CurTok.getLocation();
  unsigned int TokLength = CurTok.getLength();
  const char *C = SM.getCharacterData(CurTokBegin) + TokLength;
  while (C && *C) {
    if (!isspace(*C)) {
      break;
    }
#if defined(__linux__)
    if (*C == '\n') {
      break;
    }
#elif defined(_WIN32)
    if (*C == '\r') {
      break;
    }
#else
#error Only support Windows and Linux.
#endif
    ++C;
  }
  return C - SM.getCharacterData(CurTokBegin);
}

/// This function gets the statement nodes of the initialization, condition or
/// increment parts of the \p Node.
/// \param Node The statement node which is if, for, do, while or switch.
/// \return The result statement nodes vector.
std::vector<const Stmt*> getConditionNode(ast_type_traits::DynTypedNode Node){
  std::vector<const Stmt *> Res;
  if (const IfStmt *CondtionNode = Node.get<IfStmt>()) {
    Res.push_back(CondtionNode->getCond());
    Res.push_back(CondtionNode->getConditionVariableDeclStmt());
  } else if (const ForStmt *CondtionNode = Node.get<ForStmt>()) {
    Res.push_back(CondtionNode->getCond());
    Res.push_back(CondtionNode->getInc());
    Res.push_back(CondtionNode->getInit());
    Res.push_back(CondtionNode->getConditionVariableDeclStmt());
  } else if (const WhileStmt *CondtionNode = Node.get<WhileStmt>()) {
    Res.push_back(CondtionNode->getCond());
    Res.push_back(CondtionNode->getConditionVariableDeclStmt());
  } else if (const DoStmt *CondtionNode = Node.get<DoStmt>()) {
    Res.push_back(CondtionNode->getCond());
  } else if (const SwitchStmt *CondtionNode = Node.get<SwitchStmt>()) {
    Res.push_back(CondtionNode->getCond());
    Res.push_back(CondtionNode->getConditionVariableDeclStmt());
  }
  return Res;
}

/// This function checks whether expression \p E is a child node of the
/// initialization, condition or increment part of for, while, do, if or switch.
/// \param E The expression to be checked.
/// \return The result.
bool isConditionOfFlowControl(const clang::Expr* E) {
  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto ParentNodes = Context.getParents(*E);
  ast_type_traits::DynTypedNode ParentNode;
  std::vector<ast_type_traits::DynTypedNode> AncestorNodes;
  bool FoundStmtHasCondition = false;
  while (!ParentNodes.empty()) {
    ParentNode = ParentNodes[0];
    AncestorNodes.push_back(ParentNode);
    if (ParentNode.get<IfStmt>() || ParentNode.get<ForStmt>() ||
        ParentNode.get<WhileStmt>() || ParentNode.get<DoStmt>() ||
        ParentNode.get<SwitchStmt>()) {
      FoundStmtHasCondition = true;
      break;
    }
    ParentNodes = Context.getParents(ParentNode);
  }
  if (!FoundStmtHasCondition)
    return false;
  auto CondtionNodes =
      getConditionNode(AncestorNodes[AncestorNodes.size() - 1]);

  for (auto CondtionNode : CondtionNodes) {
    if (CondtionNode == nullptr)
      continue;
    for (auto Node : AncestorNodes) {
      if (Node.get<Stmt>() && Node.get<Stmt>() == CondtionNode)
        return true;
    }
    if (E == CondtionNode)
      return true;
  }
  return false;
}

std::string getBufferNameAndDeclStr(const std::string &PointerName,
                                    const ASTContext &AC,
                                    const std::string &TypeAsStr,
                                    SourceLocation SL, std::string &BufferDecl,
                                    int DistinctionID) {
  std::string BufferTempName = "buffer_ct" + std::to_string(DistinctionID);
  std::string AllocationTempName =
      "allocation_ct" + std::to_string(DistinctionID);
  // TODO: reinterpret will copy more data
  BufferDecl = getIndent(SL, AC.getSourceManager()).str() + "auto " +
               AllocationTempName +
               " = dpct::memory_manager::get_instance().translate_ptr(" +
               PointerName + ");" + getNL() +
               getIndent(SL, AC.getSourceManager()).str() +
               "cl::sycl::buffer<" + TypeAsStr + ",1> " + BufferTempName +
               " = " + AllocationTempName + ".buffer.reinterpret<" + TypeAsStr +
               ", 1>(cl::sycl::range<1>(" + AllocationTempName +
               ".size/sizeof(" + TypeAsStr + ")));" + getNL();
  return BufferTempName;
}
std::string getBufferNameAndDeclStr(const Expr *Arg, const ASTContext &AC,
                                    const std::string &TypeAsStr,
                                    SourceLocation SL, std::string &BufferDecl,
                                    int DistinctionID) {
  std::string PointerName = getStmtSpelling(Arg, AC);
  return getBufferNameAndDeclStr(PointerName, AC, TypeAsStr, SL, BufferDecl,
                                 DistinctionID);
}

// Recursively travers the subtree under \p S for all the reference of \p VD,
// store and return matched nodes in \p Result
void VarReferencedInFD(const Stmt *S, const ValueDecl *VD,
                       std::vector<const clang::DeclRefExpr *> &Result) {
  if (!S)
    return;

  if (auto DRF = dyn_cast<DeclRefExpr>(S)) {
    if (DRF->getDecl() == VD) {
      Result.push_back(DRF);
    }
  }
  for (auto It = S->child_begin(); It != S->child_end(); ++It) {
    VarReferencedInFD(*It, VD, Result);
  }
}

// Get the length of spaces until the next new line char, including the length
// of new line chars ('\r' and '\n').
// Return 0 if there is non-space char before the next new line char.
int getLengthOfSpacesToEndl(const char *CharData) {
  if (!CharData)
    return 0;
  int Len = 0;
  while (CharData && *CharData) {
    if (*CharData == '\n')
      return Len + 1;
    if (*CharData == '\r')
      return Len + 2;
    if (isspace(*CharData)) {
      ++CharData;
      ++Len;
    } else {
      return 0;
    }
  }
  return 0;
}
/// Calculate the ranges of the input \p Repls which has NOT set NotFormatFlags.
/// \param Repls Replacements with format flags.
/// \return The result ranges.
std::vector<clang::tooling::Range> calculateRangesWithFormatFlag(
    const clang::tooling::Replacements &Repls) {
  std::vector<bool> NotFormatFlags;
  std::vector<clang::tooling::Range> Ranges;

  int Diff = 0;
  for (auto R : Repls) {
    if (R.getNotFormatFlag())
      NotFormatFlags.push_back(true);
    else
      NotFormatFlags.push_back(false);
    Ranges.emplace_back(/*offset*/ R.getOffset() + Diff,
                                     /*length*/ R.getReplacementText().size());

    Diff = Diff + R.getReplacementText().size() - R.getLength();
  }

  std::vector<clang::tooling::Range> RangesAfterFilter;
  int Size = Ranges.size();
  for (int i = 0; i < Size; ++i) {
    if (!NotFormatFlags[i])
      RangesAfterFilter.push_back(Ranges[i]);
  }
  return RangesAfterFilter;
}
