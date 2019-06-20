//===--- Diagnostics.h ---------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef SYCLCT_AST_DIAGNOSTICS_H
#define SYCLCT_AST_DIAGNOSTICS_H

#include "Debug.h"
#include "SaveNewFiles.h"
#include "TextModification.h"

#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/Support/FormatVariadic.h"

#include <assert.h>
#include <unordered_map>

extern llvm::cl::opt<std::string> SuppressWarnings;
extern llvm::cl::opt<std::string> OutputFile;
extern llvm::cl::opt<OutputVerbosityLev> OutputVerbosity;
extern bool SuppressWarningsAllFlag;

namespace clang {
namespace syclct {

struct DiagnosticsMessage;

extern std::unordered_map<int, DiagnosticsMessage> DiagnosticIDTable;
extern std::unordered_map<int, DiagnosticsMessage> CommentIDTable;

struct DiagnosticsMessage {
  int ID;
  int Category;
  const char *Msg;
  DiagnosticsMessage() = default;
  DiagnosticsMessage(std::unordered_map<int, DiagnosticsMessage> &Table, int ID,
                     int Category, const char *Msg)
      : ID(ID), Category(Category), Msg(Msg) {
    assert(Table.find(ID) == Table.end() && "[SYCLCT Internal error] Two "
                                            "messages with the same ID "
                                            "are being registered");
    Table[ID] = *this;
  }
};

#define DEF_NOTE(NAME, ID, MSG) NAME = ID,
#define DEF_ERROR(NAME, ID, MSG) NAME = ID,
#define DEF_WARNING(NAME, ID, MSG) NAME = ID,
#define DEF_COMMENT(NAME, ID, MSG)
enum class Diagnostics {
#include "Diagnostics.inc"
#undef DEF_NOTE
#undef DEF_ERROR
#undef DEF_WARNING
#undef DEF_COMMENT
  END
};

#define DEF_NOTE(NAME, ID, MSG)
#define DEF_ERROR(NAME, ID, MSG)
#define DEF_WARNING(NAME, ID, MSG)
#define DEF_COMMENT(NAME, ID, MSG) NAME = ID,
enum class Comments {
#include "Diagnostics.inc"
#undef DEF_NOTE
#undef DEF_ERROR
#undef DEF_WARNING
#undef DEF_COMMENT
  END
};

#define DEF_NOTE(NAME, ID, MSG)
#define DEF_ERROR(NAME, ID, MSG)
#define DEF_WARNING(NAME, ID, MSG) NAME = ID,
#define DEF_COMMENT(NAME, ID, MSG)
enum class Warnings {
  BEGIN = 1000,
#include "Diagnostics.inc"
#undef DEF_NOTE
#undef DEF_ERROR
#undef DEF_WARNING
#undef DEF_COMMENT
  END
};

namespace DiagnosticsUtils {

extern unsigned int UniqueID;

template <typename... Ts> static void applyReport(DiagnosticBuilder &B) {}

template <typename FTy, typename... Ts>
static void applyReport(DiagnosticBuilder &B, const FTy &F,
                        const Ts &... Rest) {
  B << F;
  applyReport<Ts...>(B, Rest...);
}

static inline std::string getMessagePrefix(int ID) {
  return "SYCLCT" + std::to_string(ID) + ":" + std::to_string(UniqueID) + ": ";
}

template <typename... Ts>
void reportWarning(SourceLocation SL, const DiagnosticsMessage &Msg,
                   const CompilerInstance &CI, Ts &&... Vals) {

  DiagnosticsEngine &DiagEngine = CI.getDiagnostics();

  std::string Message = getMessagePrefix(Msg.ID) + Msg.Msg;

  if (!OutputFile.empty()) {
    //  Redirects warning message to output file if the option "-output-file" is
    //  set
    const SourceManager &SM = CI.getSourceManager();
    int LineNum = SM.getSpellingLineNumber(SL);
    const std::pair<FileID, unsigned> DecomposedLocation =
        SM.getDecomposedLoc(SL);

    FileID FID = DecomposedLocation.first;
    unsigned *LineCache =
        SM.getSLocEntry(FID).getFile().getContentCache()->SourceLineCache;
    const char *Buffer = SM.getBuffer(FID)->getBufferStart();
    std::string LineOriCode(Buffer + LineCache[LineNum - 1],
                            Buffer + LineCache[LineNum]);

    const SourceLocation FileLoc = SM.getFileLoc(SL);
    std::string File = FileLoc.printToString(SM);
    Message = File + " warning: " + Message + "\n" + LineOriCode;
    SyclctTerm() << Message;
  }

  if (OutputVerbosity != silent) {
    unsigned ID = DiagEngine.getDiagnosticIDs()->getCustomDiagID(
        (DiagnosticIDs::Level)Msg.Category, Message);
    auto B = DiagEngine.Report(SL, ID);
    applyReport<Ts...>(B, Vals...);
  }
}

static inline SourceLocation getStartOfLine(SourceLocation Loc,
                                            const SourceManager &SM,
                                            const LangOptions &LangOpts) {
  auto LocInfo = SM.getDecomposedLoc(SM.getExpansionLoc(Loc));
  auto Buffer = SM.getBufferData(LocInfo.first);
  auto NLPos = Buffer.find_last_of('\n', LocInfo.second);
  if (NLPos == StringRef::npos) {
    NLPos = 0;
  } else {
    NLPos++;
  }
  return SM.getExpansionLoc(Loc).getLocWithOffset(NLPos - LocInfo.second);
}

template <typename... Ts>
TextModification *
insertCommentPrevLine(SourceLocation SL, const DiagnosticsMessage &Msg,
                      const CompilerInstance &CI, Ts &&... Vals) {

  auto StartLoc = getStartOfLine(SL, CI.getSourceManager(), LangOptions());
  auto Formatted = llvm::formatv(Msg.Msg, std::forward<Ts>(Vals)...);
  std::string Str;
  llvm::raw_string_ostream OS(Str);
  OS << getMessagePrefix(Msg.ID);
  OS << Formatted;
  return new InsertComment(StartLoc, OS.str());
}

// Emits a warning/error/note and/or comment depending on MsgID. For details
template <typename IDTy, typename... Ts>
void report(SourceLocation SL, IDTy MsgID, const CompilerInstance &CI,
            TransformSetTy *TS, Ts &&... Vals) {
  if (!SuppressWarningsAllFlag) {
    static bool WarningInitialized = false;
    static std::set<int> WarningIDs;
    if (!WarningInitialized) {
      // Separate string into list by comma
      if (SuppressWarnings != "") {
        auto WarningStrs = split(SuppressWarnings, ',');
        for (const auto &Str : WarningStrs) {
          auto Range = split(Str, '-');
          if (Range.size() == 1) {
            WarningIDs.insert(std::stoi(Str));
          } else if (Range.size() == 2) {
            size_t RangeBegin = std::stoi(Range[0]);
            size_t RangeEnd = std::stoi(Range[1]);
            if (RangeBegin < (size_t)Warnings::BEGIN)
              RangeBegin = (size_t)Warnings::BEGIN;
            if (RangeEnd >= (size_t)Warnings::END)
              RangeEnd = (size_t)Warnings::END - 1;
            if (RangeBegin > RangeEnd)
              continue;
            for (auto I = RangeBegin; I <= RangeEnd; ++I)
              WarningIDs.insert(I);
          }
        }
      }
      WarningInitialized = true;
    }

    // Only report warnings that are not suppressed
    if (WarningIDs.find((int)MsgID) == WarningIDs.end() &&
        DiagnosticIDTable.find((int)MsgID) != DiagnosticIDTable.end())
      reportWarning(SL, DiagnosticIDTable[(int)MsgID], CI,
                    std::forward<Ts>(Vals)...);
  }
  if (TS && CommentIDTable.find((int)MsgID) != CommentIDTable.end()) {
    TS->emplace_back(insertCommentPrevLine(SL, CommentIDTable[(int)MsgID], CI,
                                           std::forward<Ts>(Vals)...));
  }
  UniqueID++;
}
} // namespace DiagnosticsUtils
} // namespace syclct
} // namespace clang
#endif // SYCLCT_AST_DIAGNOSTICS_H
