//===--- CallExprRewriter.h -----------------------------*- C++ -*---===//
//
// Copyright (C) 2019 - 2021 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef CALL_EXPR_REWRITER_H
#define CALL_EXPR_REWRITER_H

#include "Diagnostics.h"

namespace clang {
namespace dpct {

class CallExprRewriter;
class FuncCallExprRewriter;
class MathCallExprRewriter;
class MathFuncNameRewriter;
class MathSimulatedRewriter;
class MathTypeCastRewriter;
class MathBinaryOperatorRewriter;
class MathUnsupportedRewriter;
class WarpFunctionRewriter;
class NoRewriteFuncNameRewriter;
template <class... MsgArgs> class UnsupportFunctionRewriter;

/*
Factory usage example:
using BinaryOperatorExprRewriterFactory =
    CallExprRewriterFactory<BinaryOperatorExprRewriter, BinaryOperatorKind>;
*/
/// Base class in abstract factory pattern
class CallExprRewriterFactoryBase {
public:
  virtual std::shared_ptr<CallExprRewriter> create(const CallExpr *) const = 0;
  virtual ~CallExprRewriterFactoryBase() {}

  static std::unique_ptr<const std::unordered_map<std::string,
                                  std::shared_ptr<CallExprRewriterFactoryBase>>>
      RewriterMap;
  static void initRewriterMap();
};

/// Abstract factory for all rewriter factories
template <class RewriterTy, class... Args>
class CallExprRewriterFactory : public CallExprRewriterFactoryBase {
  std::tuple<std::string, Args...> Initializer;

private:
  template <size_t... Idx>
  inline std::shared_ptr<CallExprRewriter>
  createRewriter(const CallExpr *Call, std::index_sequence<Idx...>) const {
    return std::shared_ptr<RewriterTy>(
        new RewriterTy(Call, std::get<Idx>(Initializer)...));
  }

public:
  CallExprRewriterFactory(StringRef SourceCalleeName, Args... Arguments)
      : Initializer(SourceCalleeName.str(), std::forward<Args>(Arguments)...) {}
  // Create a meaningful rewriter only if the CallExpr is not nullptr
  std::shared_ptr<CallExprRewriter>
  create(const CallExpr *Call) const override {
    if (!Call)
      return std::shared_ptr<CallExprRewriter>();
    return createRewriter(Call,
                          std::index_sequence_for<std::string, Args...>());
  }
};

using FuncCallExprRewriterFactory =
    CallExprRewriterFactory<FuncCallExprRewriter, std::string>;
using MathFuncNameRewriterFactory =
    CallExprRewriterFactory<MathFuncNameRewriter, std::string>;
using NoRewriteFuncNameRewriterFactory =
    CallExprRewriterFactory<NoRewriteFuncNameRewriter, std::string>;
using MathUnsupportedRewriterFactory =
    CallExprRewriterFactory<MathUnsupportedRewriter, std::string>;
using MathSimulatedRewriterFactory =
    CallExprRewriterFactory<MathSimulatedRewriter, std::string>;
using MathTypeCastRewriterFactory =
    CallExprRewriterFactory<MathTypeCastRewriter, std::string>;
using MathBinaryOperatorRewriterFactory =
    CallExprRewriterFactory<MathBinaryOperatorRewriter, BinaryOperatorKind>;
using WarpFunctionRewriterFactory =
    CallExprRewriterFactory<WarpFunctionRewriter, std::string>;
template <class... MsgArgs>
using UnsupportFunctionRewriterFactory =
    CallExprRewriterFactory<UnsupportFunctionRewriter<MsgArgs...>, Diagnostics,
                            MsgArgs...>;

/// Base class for rewriting call expressions
class CallExprRewriter {
protected:
  // Call is guaranteed not to be nullptr
  const CallExpr *Call;
  StringRef SourceCalleeName;

protected:
  // All instances of the subclasses can only be constructed by corresponding
  // factories. As a result, the access modifiers of the constructors are
  // supposed to be protected instead of public.
  CallExprRewriter(const CallExpr *Call, StringRef SourceCalleeName)
      : Call(Call), SourceCalleeName(SourceCalleeName) {}
  bool NoRewrite = false;
public:
  ArgumentAnalysis Analyzer;
  virtual ~CallExprRewriter() {}

  /// This function should be overwrited to implement call expression rewriting.
  virtual Optional<std::string> rewrite() = 0;
  // Emits a warning/error/note and/or comment depending on MsgID. For details
  // see Diagnostics.inc, Diagnostics.h and Diagnostics.cpp
  template <typename IDTy, typename... Ts>
  inline void report(IDTy MsgID, bool UseTextBegin, Ts &&... Vals) {
    TransformSetTy TS;
    auto SL = Call->getBeginLoc();
    auto &SM = DpctGlobalInfo::getSourceManager();
    if (SL.isMacroID() && !SM.isMacroArgExpansion(SL)) {
      auto ItMatch = dpct::DpctGlobalInfo::getMacroTokenToMacroDefineLoc().find(
        getHashStrFromLoc(SM.getImmediateSpellingLoc(SL)));
      if (ItMatch != dpct::DpctGlobalInfo::getMacroTokenToMacroDefineLoc().end()) {
        if (ItMatch->second->IsInRoot) {
          SL = ItMatch->second->NameTokenLoc;
        }
      }
    }
    DiagnosticsUtils::report<IDTy, Ts...>(
      SL, MsgID, DpctGlobalInfo::getCompilerInstance(), &TS,
      UseTextBegin, std::forward<Ts>(Vals)...);
    for (auto &T : TS)
      DpctGlobalInfo::getInstance().addReplacement(
        T->getReplacement(DpctGlobalInfo::getContext()));
  }

  bool isNoRewrite() {
    return NoRewrite;
  }
protected:
  std::vector<std::string> getMigratedArgs();
  std::string getMigratedArg(unsigned Index);

  StringRef getSourceCalleeName() { return SourceCalleeName; }
};

class ConditionalRewriterFactory : public CallExprRewriterFactoryBase {
  std::function<bool(const CallExpr *)> Pred;
  std::shared_ptr<CallExprRewriterFactoryBase> First, Second;

public:
  template <class InputPred>
  ConditionalRewriterFactory(
      InputPred &&P, std::shared_ptr<CallExprRewriterFactoryBase> FirstFactory,
      std::shared_ptr<CallExprRewriterFactoryBase> SecondFactory)
      : Pred(std::forward<InputPred>(P)), First(FirstFactory),
        Second(SecondFactory) {}
  std::shared_ptr<CallExprRewriter> create(const CallExpr *C) const override {
    if (Pred(C))
      return First->create(C);
    else
      return Second->create(C);
  }
};

class AssignableRewriter : public CallExprRewriter {
  std::shared_ptr<CallExprRewriter> Inner;
  bool IsAssigned;

public:
  AssignableRewriter(const CallExpr *C,
                     std::shared_ptr<CallExprRewriter> InnerRewriter)
      : CallExprRewriter(C, ""), Inner(InnerRewriter),
        IsAssigned(isAssigned(C)) {}

  Optional<std::string> rewrite() override {
    Optional<std::string> &&Result = Inner->rewrite();
    if (Result.hasValue() && IsAssigned)
      return "(" + Result.getValue() + ", 0)";
    return Result;
  }
};

class AssignableRewriterFactory : public CallExprRewriterFactoryBase {
  std::shared_ptr<CallExprRewriterFactoryBase> Inner;

public:
  AssignableRewriterFactory(
      std::shared_ptr<CallExprRewriterFactoryBase> InnerFactory)
      : Inner(InnerFactory) {}
  std::shared_ptr<CallExprRewriter> create(const CallExpr *C) const override {
    return std::make_shared<AssignableRewriter>(C, Inner->create(C));
  }
};

/// Base class for rewriting function calls
class FuncCallExprRewriter : public CallExprRewriter {
protected:
  std::string TargetCalleeName;
  std::vector<std::string> RewriteArgList;

protected:
  FuncCallExprRewriter(const CallExpr *Call, StringRef SourceCalleeName,
                       StringRef TargetCalleeName)
      : CallExprRewriter(Call, SourceCalleeName),
        TargetCalleeName(TargetCalleeName) {}

public:
  virtual ~FuncCallExprRewriter() {}

  virtual Optional<std::string> rewrite() override;

  friend FuncCallExprRewriterFactory;

protected:
  template <class... Args> void appendRewriteArg(Args &&... Arguments) {
    RewriteArgList.emplace_back(std::forward<Args...>(Arguments)...);
  }

  // Build string which is used to replace original expession.
  Optional<std::string> buildRewriteString();

  void setTargetCalleeName(const std::string &Str) { TargetCalleeName = Str; }
};

/// Base class for rewriting math function calls
class MathCallExprRewriter : public FuncCallExprRewriter {
public:
  virtual Optional<std::string> rewrite() override;

protected:
  MathCallExprRewriter(const CallExpr *Call, StringRef SourceCalleeName,
                       StringRef TargetCalleeName)
      : FuncCallExprRewriter(Call, SourceCalleeName, TargetCalleeName) {}

  void reportUnsupportedRoundingMode();
};


/// The rewriter for renaming math function calls
class MathFuncNameRewriter : public MathCallExprRewriter {
protected:
  MathFuncNameRewriter(const CallExpr *Call, StringRef SourceCalleeName,
    StringRef TargetCalleeName)
    : MathCallExprRewriter(Call, SourceCalleeName, TargetCalleeName) {}

public:
  virtual Optional<std::string> rewrite() override;

protected:
  std::string getNewFuncName();
  static const std::vector<std::string> SingleFuctions;
  static const std::vector<std::string> DoubleFuctions;
  friend MathFuncNameRewriterFactory;
};

/// The rewriter for renaming math function calls
class NoRewriteFuncNameRewriter : public MathFuncNameRewriter {
protected:
  NoRewriteFuncNameRewriter(const CallExpr *Call, StringRef SourceCalleeName,
    StringRef TargetCalleeName)
    : MathFuncNameRewriter(Call, SourceCalleeName, TargetCalleeName) {
    NoRewrite = true;
  }

public:
  virtual Optional<std::string> rewrite() override;
  friend NoRewriteFuncNameRewriterFactory;
};

/// The rewriter for warning on unsupported math functions
class MathUnsupportedRewriter : public MathCallExprRewriter {
protected:
  using Base = MathCallExprRewriter;
  MathUnsupportedRewriter(const CallExpr *Call, StringRef SourceCalleeName,
                          StringRef TargetCalleeName)
      : Base(Call, SourceCalleeName, TargetCalleeName) {}

  virtual Optional<std::string> rewrite() override;

  friend MathUnsupportedRewriterFactory;
};

/// The rewriter for replacing math function calls with type casting expressions
class MathTypeCastRewriter : public MathCallExprRewriter {
protected:
  using Base = MathCallExprRewriter;
  MathTypeCastRewriter(const CallExpr *Call, StringRef SourceCalleeName,
                       StringRef TargetCalleeName)
      : Base(Call, SourceCalleeName, TargetCalleeName) {}

  virtual Optional<std::string> rewrite() override;

  friend MathTypeCastRewriterFactory;
};

/// The rewriter for replacing math function calls with emulations
class MathSimulatedRewriter : public MathCallExprRewriter {
protected:
  using Base = MathCallExprRewriter;
  MathSimulatedRewriter(const CallExpr *Call, StringRef SourceCalleeName,
                        StringRef TargetCalleeName)
      : Base(Call, SourceCalleeName, TargetCalleeName) {}

  virtual Optional<std::string> rewrite() override;

  friend MathSimulatedRewriterFactory;
};

/// The rewriter for replacing math function calls with binary operator
/// expressions
class MathBinaryOperatorRewriter : public MathCallExprRewriter {
  std::string LHS, RHS;
  BinaryOperatorKind Op;

protected:
  MathBinaryOperatorRewriter(const CallExpr *Call, StringRef SourceCalleeName,
                             BinaryOperatorKind Op)
      : MathCallExprRewriter(Call, SourceCalleeName, ""), Op(Op) {}

public:
  virtual ~MathBinaryOperatorRewriter() {}

  virtual Optional<std::string> rewrite() override;

protected:
  void setLHS(std::string L) { LHS = L; }
  void setRHS(std::string R) { RHS = R; }

  // Build string which is used to replace original expession.
  inline Optional<std::string> buildRewriteString() {
    if (LHS == "")
      return buildString(BinaryOperator::getOpcodeStr(Op), RHS);
    return buildString(LHS, " ", BinaryOperator::getOpcodeStr(Op), " ", RHS);
  }

  friend MathBinaryOperatorRewriterFactory;
};

/// The rewriter for migrating warp functions
class WarpFunctionRewriter : public FuncCallExprRewriter {
private:
  static const std::map<std::string, std::string> WarpFunctionsMap;
  void reportNoMaskWarning() {
    report(Diagnostics::MASK_UNSUPPORTED, false, TargetCalleeName);
  }

protected:
  WarpFunctionRewriter(const CallExpr *Call, StringRef SourceCalleeName,
                       StringRef TargetCalleeName)
      : FuncCallExprRewriter(Call, SourceCalleeName, TargetCalleeName) {}

public:
  virtual Optional<std::string> rewrite() override;

protected:
  std::string getNewFuncName();

  friend WarpFunctionRewriterFactory;
};

template <class StreamT, class T> void print(StreamT &Stream, const T &Val) {
  Val.print(Stream);
}
template <class StreamT> void print(StreamT &Stream, const Expr *E) {
  ExprAnalysis EA;
  print(Stream, EA, E);
}
template <class StreamT> void print(StreamT &Stream, StringRef Str) {
  Stream << Str;
}
template <class StreamT> void print(StreamT &Stream, const std::string &Str) {
  Stream << Str;
}
template <class StreamT>
void print(StreamT &Stream, const TemplateArgumentInfo &Arg) {
  print(Stream, Arg.getString());
}
template <class StreamT>
void print(StreamT &Stream, ExprAnalysis &EA, const Expr *E) {
  EA.analyze(E);
  Stream << EA.getRewritePrefix() << EA.getReplacedString()
         << EA.getRewritePostfix();
}

template <class StreamT>
void print(StreamT &Stream, ArgumentAnalysis &AA, std::pair<const CallExpr*, const Expr*> P) {
  AA.setCallSpelling(P.first);
  AA.analyze(P.second);
  Stream << AA.getRewritePrefix() << AA.getRewriteString()
    << AA.getRewritePostfix();
}

template <class StreamT>
void printWithParens(StreamT &Stream, ExprAnalysis &EA, const Expr *E) {
  std::unique_ptr<ParensPrinter<StreamT>> Paren;
  E = E->IgnoreImplicitAsWritten();
  if (needExtraParens(E))
    Paren = std::make_unique<ParensPrinter<StreamT>>(Stream);
  print(Stream, EA, E);
}
template <class StreamT> void printWithParens(StreamT &Stream, const Expr *E) {
  ExprAnalysis EA;
  printWithParens(Stream, EA, E);
}

template <class StreamT>
void printWithParens(StreamT &Stream, ArgumentAnalysis &AA, std::pair<const CallExpr*, const Expr*> P) {
  std::unique_ptr<ParensPrinter<StreamT>> Paren;
  P.second = P.second->IgnoreImplicitAsWritten();
  if (needExtraParens(P.second))
    Paren = std::make_unique<ParensPrinter<StreamT>>(Stream);
  print(Stream, AA, P);
}

template <class StreamT> void printWithParens(StreamT &Stream, std::pair<const CallExpr*, const Expr*> P) {
  ArgumentAnalysis AA;
  printWithParens(Stream, AA, P);
}

template <class StreamT> void printMemberOp(StreamT &Stream, bool IsArrow) {
  if (IsArrow)
    Stream << "->";
  else
    Stream << ".";
}

class DerefExpr {
  bool AddrOfRemoved = false, NeedParens = false;
  const Expr *E = nullptr;

  template <class StreamT>
  void print(StreamT &Stream, ExprAnalysis &EA, bool IgnoreDerefOp) const {
    std::unique_ptr<ParensPrinter<StreamT>> Parens;
    if (!AddrOfRemoved && !IgnoreDerefOp)
      Stream << "*";

    printWithParens(Stream, EA, E);
  }

  DerefExpr() = default;

public:
  template <class StreamT>
  void printArg(StreamT &Stream, ArgumentAnalysis &A) const {
    print(Stream, A, false);
  }
  template <class StreamT> void printMemberBase(StreamT &Stream) const {
    ExprAnalysis EA;
    print(Stream, EA, true);
    printMemberOp(Stream, !AddrOfRemoved);
  }

  template <class StreamT> void print(StreamT &Stream) const {
    ExprAnalysis EA;
    print(Stream, EA, false);
  }

  static DerefExpr create(const Expr *E);
};

class RenameWithSuffix {
  StringRef OriginalName, SuffixStr;

public:
  RenameWithSuffix(StringRef Original, StringRef Suffix)
      : OriginalName(Original), SuffixStr(Suffix) {}
  template <class StreamT> void print(StreamT &Stream) const {
    Stream << OriginalName;
    if (!SuffixStr.empty())
      Stream << "_" << SuffixStr;
  }
};

template <bool HasPrefixArg, class... ArgsT> class ArgsPrinter;
template <bool HasPrefixArg> class ArgsPrinter<HasPrefixArg> {
  mutable ArgumentAnalysis A;

public:
  template <class StreamT> void print(StreamT &) const {}
  template <class StreamT>
  void printArg(std::false_type, StreamT &Stream, const Expr *E) const {
    dpct::print(Stream, A, E);
  }

  template <class StreamT>
  void printArg(std::false_type, StreamT &Stream,
                std::pair<const CallExpr *, const Expr *> P) const {
    dpct::print(Stream, A, P);
  }

  template <class StreamT>
  void printArg(std::false_type, StreamT &Stream, DerefExpr Arg) const {
    Arg.printArg(Stream, A);
  }
  template <class StreamT, class ArgT>
  void printArg(std::false_type, StreamT &Stream, const ArgT &Arg) const {
    dpct::print(Stream, Arg);
  }
  template <class StreamT, class ArgT>
  void printArg(std::false_type, StreamT &Stream,
                const std::vector<ArgT> &Vec) const {
    if (Vec.empty())
      return;
    auto Itr = Vec.begin();
    printArg(std::false_type(), Stream, *Itr);
    while (++Itr != Vec.end()) {
      printArg(std::true_type(), Stream, *Itr);
    }
  }
  template <class StreamT, class ArgT>
  void printArg(std::true_type, StreamT &Stream,
                const std::vector<ArgT> &Vec) const {
    for (auto &Arg : Vec) {
      printArg(std::true_type(), Stream, Arg);
    }
  }
  template <class StreamT, class ArgT>
  void printArg(std::true_type, StreamT &Stream, ArgT &&Arg) const {
    Stream << ", ";
    printArg(std::false_type(), Stream, std::forward<ArgT>(Arg));
  }

  ArgsPrinter() = default;
  ArgsPrinter(const ArgsPrinter &) {}
};
template <bool HasPrefixArg, class FirstArgT, class... RestArgsT>
class ArgsPrinter<HasPrefixArg, FirstArgT, RestArgsT...>
    : public ArgsPrinter<true, RestArgsT...> {
  using Base = ArgsPrinter<true, RestArgsT...>;
  FirstArgT First;

public:
  template <class InputFirstArgT, class... InputRestArgsT>
  ArgsPrinter(InputFirstArgT &&FirstArg, InputRestArgsT &&... RestArgs)
      : Base(std::forward<InputRestArgsT>(RestArgs)...),
        First(std::forward<InputFirstArgT>(FirstArg)) {}
  template <class StreamT> void print(StreamT &Stream) const {
    Base::printArg(std::integral_constant<bool, HasPrefixArg>(), Stream, First);
    Base::print(Stream);
  }
};

template <class StreamT>
void printBase(StreamT &Stream, std::pair<const CallExpr*, const Expr*> P, bool IsArrow) {
  printWithParens(Stream, P);
  printMemberOp(Stream, IsArrow);
}

template <class StreamT>
void printBase(StreamT &Stream, const Expr *E, bool IsArrow) {
  printWithParens(Stream, E);
  printMemberOp(Stream, IsArrow);
}
template <class StreamT>
void printBase(StreamT &Stream, const DerefExpr &D, bool) {
  D.printMemberBase(Stream);
}
template <class StreamT, class T>
void printBase(StreamT &Stream, const T &Val, bool IsArrow) {
  print(Stream, Val);
  printMemberOp(Stream, IsArrow);
}

template <class CalleeT, class... CallArgsT> class CallExprPrinter {
  CalleeT Callee;
  ArgsPrinter<false, CallArgsT...> Args;

public:
  CallExprPrinter(CalleeT Callee, CallArgsT &&... Args)
      : Callee(Callee), Args(std::forward<CallArgsT>(Args)...) {}
  template <class StreamT> void print(StreamT &Stream) const {
    dpct::print(Stream, Callee);
    ParensPrinter<StreamT> Parens(Stream);
    Args.print(Stream);
  }
};

class TemplatedCallee {
  StringRef CalleeName;
  ArgsPrinter<false, std::vector<TemplateArgumentInfo>> TemplateArgs;

public:
  TemplatedCallee(StringRef Callee, std::vector<TemplateArgumentInfo> &&Args)
      : CalleeName(Callee), TemplateArgs(std::move(Args)) {}
  template <class StreamT> void print(StreamT &Stream) const {
    dpct::print(Stream, CalleeName);
    Stream << "<";
    TemplateArgs.print(Stream);
    Stream << ">";
  }
};

template <class BaseT, class MemberT> class MemberExprPrinter {
  BaseT Base;
  bool IsArrow;
  MemberT MemberName;

public:
  MemberExprPrinter(BaseT Base, bool IsArrow, MemberT MemberName)
      : Base(Base), IsArrow(IsArrow), MemberName(MemberName) {}

  template <class StreamT> void print(StreamT &Stream) const {
    printBase(Stream, Base, IsArrow);
    dpct::print(Stream, MemberName);
  }
};

template <class BaseT, class MemberT, class... CallArgsT>
class MemberCallPrinter
    : public CallExprPrinter<MemberExprPrinter<BaseT, MemberT>, CallArgsT...> {
public:
  MemberCallPrinter(BaseT Base, bool IsArrow, MemberT MemberName,
                    CallArgsT &&... Args)
      : CallExprPrinter<MemberExprPrinter<BaseT, MemberT>, CallArgsT...>(
            MemberExprPrinter<BaseT, MemberT>(std::move(Base), IsArrow,
                                              std::move(MemberName)),
            std::forward<CallArgsT>(Args)...) {}
};

template <class LValueT, class RValueT> class AssignExprPrinter {
  LValueT LVal;
  RValueT RVal;

public:
  AssignExprPrinter(LValueT &&L, RValueT &&R)
      : LVal(std::forward<LValueT>(L)), RVal(std::forward<RValueT>(R)) {}
  template <class StreamT> void print(StreamT &Stream) const {
    dpct::print(Stream, LVal);
    Stream << " = ";
    dpct::print(Stream, RVal);
  }
};

template <class ArgT> class DeleterCallExprRewriter : public CallExprRewriter {
  ArgT Arg;

public:
  DeleterCallExprRewriter(const CallExpr *C, StringRef Source,
                          std::function<ArgT(const CallExpr *)> ArgCreator)
      : CallExprRewriter(C, Source), Arg(ArgCreator(C)) {}
  Optional<std::string> rewrite() override {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    OS << "delete ";
    printWithParens(OS, Arg);
    return Result;
  }
};

template <class... ArgsT>
class NewExprPrinter : CallExprPrinter<StringRef, ArgsT...> {
  using Base = CallExprPrinter<StringRef, ArgsT...>;

public:
  NewExprPrinter(StringRef TypeName, ArgsT &&... Args)
      : Base(TypeName, std::forward<ArgsT>(Args)...) {}
  template <class StreamT> void print(StreamT &Stream) const {
    Stream << "new ";
    Base::print(Stream);
  }
};

template <class FirstPrinter, class... RestPrinter>
class MultiStmtsPrinter : MultiStmtsPrinter<RestPrinter...> {
  using Base = MultiStmtsPrinter<RestPrinter...>;
  FirstPrinter First;

public:
  MultiStmtsPrinter(SourceLocation BeginLoc, SourceManager &SM,
                    FirstPrinter &&First, RestPrinter &&... Rest)
      : Base(BeginLoc, SM, std::move(Rest)...), First(std::move(First)) {}
  template <class StreamT> void print(StreamT &Stream) const {
    Base::printStmt(Stream, First);
    Base::print(Stream);
  }
};

template <class LastPrinter> class MultiStmtsPrinter<LastPrinter> {
  LastPrinter Last;
  StringRef Indent;
  StringRef NL;

protected:
  template <class StreamT, class PrinterT>
  void printStmt(StreamT &Stream, const PrinterT &Printer) const {
    dpct::print(Stream, Printer);
    Stream << ";" << NL << Indent;
  }

public:
  MultiStmtsPrinter(SourceLocation BeginLoc, SourceManager &SM,
                    LastPrinter &&Last)
      : Last(std::move(Last)), Indent(getIndent(BeginLoc, SM)),
        NL(getNL(BeginLoc, SM)) {}
  template <class StreamT> void print(StreamT &Stream) const {
    dpct::print(Stream, Last);
  }
};

template <class FirstPrinter, class... RestPrinter>
class CommaExprPrinter : CommaExprPrinter<RestPrinter...> {
  using Base = CommaExprPrinter<RestPrinter...>;
  FirstPrinter First;

public:
  CommaExprPrinter(FirstPrinter &&First, RestPrinter &&... Rest)
      : Base(std::move(Rest)...), First(std::move(First)) {}
  template <class StreamT> void print(StreamT &Stream) const {
    dpct::print(Stream, First);
    Stream << ", ";
    Base::print(Stream);
  }
};

template <class LastPrinter> class CommaExprPrinter<LastPrinter> {
  LastPrinter Last;

public:
  CommaExprPrinter(LastPrinter &&Last) : Last(std::move(Last)) {}
  template <class StreamT> void print(StreamT &Stream) const {
    dpct::print(Stream, Last);
  }
};

template <class Printer>
class PrinterRewriter : Printer, public CallExprRewriter {
public:
  template <class... ArgsT>
  PrinterRewriter(const CallExpr *C, StringRef Source, ArgsT &&... Args)
      : Printer(std::forward<ArgsT>(Args)...), CallExprRewriter(C, Source) {}
  template <class... ArgsT>
  PrinterRewriter(const CallExpr *C, StringRef Source,
                  const std::function<ArgsT(const CallExpr *)> &... ArgCreators)
      : PrinterRewriter(C, Source, ArgCreators(C)...) {}
  Optional<std::string> rewrite() override {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    Printer::print(OS);
    return OS.str();
  }
};

template <class... StmtPrinters>
class PrinterRewriter<MultiStmtsPrinter<StmtPrinters...>>
    : MultiStmtsPrinter<StmtPrinters...>, public CallExprRewriter {
  using Base = MultiStmtsPrinter<StmtPrinters...>;

public:
  PrinterRewriter(const CallExpr *C, StringRef Source,
                  StmtPrinters &&... Printers)
      : Base(C->getBeginLoc(), DpctGlobalInfo::getSourceManager(),
             std::move(Printers)...),
        CallExprRewriter(C, Source) {}
  PrinterRewriter(
      const CallExpr *C, StringRef Source,
      const std::function<StmtPrinters(const CallExpr *)> &... PrinterCreators)
      : PrinterRewriter(C, Source, PrinterCreators(C)...) {}
  Optional<std::string> rewrite() override {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    Base::print(OS);
    return OS.str();
  }
};

template <class... ArgsT>
class TemplatedCallExprRewriter
    : public PrinterRewriter<CallExprPrinter<TemplatedCallee, ArgsT...>> {
public:
  TemplatedCallExprRewriter(
      const CallExpr *C, StringRef Source,
      const std::function<TemplatedCallee(const CallExpr *)> &CalleeCreator,
      const std::function<ArgsT(const CallExpr *)> &... ArgsCreator)
      : PrinterRewriter<CallExprPrinter<TemplatedCallee, ArgsT...>>(
            C, Source, CalleeCreator(C), ArgsCreator(C)...) {}
};

template <class BaseT, class... ArgsT>
class MemberCallExprRewriter
    : public PrinterRewriter<MemberCallPrinter<BaseT, StringRef, ArgsT...>> {
public:
  MemberCallExprRewriter(
      const CallExpr *C, StringRef Source,
      const std::function<BaseT(const CallExpr *)> &BaseCreator, bool IsArrow,
      StringRef Member,
      const std::function<ArgsT(const CallExpr *)> &... ArgsCreator)
      : PrinterRewriter<MemberCallPrinter<BaseT, StringRef, ArgsT...>>(
            C, Source, BaseCreator(C), IsArrow, Member, ArgsCreator(C)...) {}
};

template <class LValueT, class RValueT>
class AssignExprRewriter
    : public PrinterRewriter<AssignExprPrinter<LValueT, RValueT>> {
public:
  AssignExprRewriter(const CallExpr *C, StringRef Source,
                     const std::function<LValueT(const CallExpr *)> &LCreator,
                     const std::function<RValueT(const CallExpr *)> &RCreator)
      : PrinterRewriter<AssignExprPrinter<LValueT, RValueT>>(
            C, Source, LCreator(C), RCreator(C)) {}
};

template <class... MsgArgs>
class UnsupportFunctionRewriter : public CallExprRewriter {
  template <class T>
  std::string getMsgArg(const std::function<T(const CallExpr *)> &Func,
                        const CallExpr *C) {
    return getMsgArg(Func(C), C);
  }
  template <class T>
  static std::string getMsgArg(const T &InputArg, const CallExpr *) {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    print(OS, InputArg);
    return OS.str();
  }

public:
  UnsupportFunctionRewriter(const CallExpr *CE, StringRef CalleeName,
                            Diagnostics MsgID, const MsgArgs &... Args)
      : CallExprRewriter(CE, CalleeName) {
    report(MsgID, false, getMsgArg(Args, CE)...);
  }

  Optional<std::string> rewrite() override { return Optional<std::string>(); }

  friend UnsupportFunctionRewriterFactory<MsgArgs...>;
};

} // namespace dpct
} // namespace clang

#endif // !__CALL_EXPR_REWRITER_H__
