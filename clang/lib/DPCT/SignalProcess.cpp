//===--- SignalProcess.cpp ------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2020 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "SignalProcess.h"
#include "SaveNewFiles.h"

#include "clang/Basic/LangOptions.h"
#include <setjmp.h>

extern void PrintReportOnFault(std::string &FaultMsg);
#if defined(__linux__) || defined(_WIN64)
#include <signal.h>
#endif

#if defined(__linux__) || defined(_WIN64)
static const std::string SigDescription(const int &Signo) {
  switch (Signo) {
  case SIGABRT:
    return "SIGABRT"; // Abnormal termination
  case SIGSEGV:
    return "SIGSEGV"; // Illegal storage access
  case SIGILL:
    return "SIGILL"; // Illegal instruction
  case SIGINT:
    return "SIGINT"; // CTRL+C signal
  case SIGTERM:
    return "SIGTERM"; // Termination request
  case SIGFPE:
    return "SIGFPE"; // Floating-point error
  default:
    return "UNKNOWN EXCEPTION";
  }
}
#endif
extern jmp_buf ProcessingEnterCP;
extern bool NoStopOnErrFlag;
extern int FatalErrorCnt;

#if defined(_WIN64)
void FaultHandler(int Signo) {
  if( NoStopOnErrFlag && (Signo == SIGILL || Signo == SIGSEGV || Signo == SIGFPE ) ) {
      std::string FaultMsg = "Error: Meet signal:" + SigDescription(Signo) +
                             "\nIntel(R) DPC++ Compatibility Tool trys to "
                             "skip further process of current file\n";
      PrintReportOnFault(FaultMsg);
      FatalErrorCnt++;
      longjmp(ProcessingEnterCP, 1);
  }
  std::string FaultMsg = "\nMeet signal:" + SigDescription(Signo) +
                         "\nIntel(R) DPC++ Compatibility Tool tries to give "
                         "analysis reports and terminates...\n";
  PrintReportOnFault(FaultMsg);
  exit(MigrationError);
}

static void SetHandler(void (*Handler)(int)) {
  signal(SIGABRT, Handler);
  signal(SIGSEGV, Handler);
  signal(SIGILL, Handler);
  signal(SIGINT, Handler);
  signal(SIGTERM, Handler);
  signal(SIGFPE, Handler);
}
#endif

#if defined(__linux__)
static void FaultHandler(int Signo, siginfo_t *Info, void *Extra) {

  if( NoStopOnErrFlag && (Signo == SIGILL || Signo == SIGSEGV || Signo == SIGFPE ) ) {
      std::string FaultMsg = "Error: Meet signal:" + SigDescription(Signo) +
                             "\nIntel(R) DPC++ Compatibility Tool trys to "
                             "skip further process of current file\n";
      PrintReportOnFault(FaultMsg);
      FatalErrorCnt++;
      longjmp(ProcessingEnterCP, 1);
  }
  std::string FaultMsg = "\nMeet signal:" + SigDescription(Signo) +
                         "\nIntel(R) DPC++ Compatibility Tool trys to give "
                         "analysis reports and terminates...\n";
  PrintReportOnFault(FaultMsg);
  exit(MigrationError);
}

static void SetHandler(void (*handler)(int, siginfo_t *, void *)) {
  struct sigaction action;
  action.sa_flags = SA_SIGINFO;
  action.sa_sigaction = handler;

  if (sigaction(SIGSEGV, &action, NULL) == -1) {
    llvm::errs() << "SIGSEGV: sigaction installation failure\n";
    exit(-1);
  }

  if (sigaction(SIGABRT, &action, NULL) == -1) {
    llvm::errs() << "SIGABRT: sigaction installation failure\n";
    exit(-1);
  }

  if (sigaction(SIGTERM, &action, NULL) == -1) {
    llvm::errs() << "SIGTERM: sigaction installation failure\n";
    exit(-1);
  }

  if (sigaction(SIGFPE, &action, NULL) == -1) {
    llvm::errs() << "SIGFPE: sigaction installation failure\n";
    exit(-1);
  }

  if (sigaction(SIGSTKFLT, &action, NULL) == -1) {
    llvm::errs() << "SIGSTKFLT: sigaction installation failure\n";
    exit(-1);
  }
  if (sigaction(SIGPIPE, &action, NULL) == -1) {
    llvm::errs() << "SIGPIPE: sigaction installation failure\n";
    exit(-1);
  }
}
#endif

#if defined(_WIN64) || defined(__linux__)
void InstallSignalHandle(void) { SetHandler(FaultHandler); }
#endif
