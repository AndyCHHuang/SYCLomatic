//===--- SaveNewFiles.cpp --------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2020 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "AnalysisInfo.h"
#include "SaveNewFiles.h"
#include "Debug.h"
#include "ExternalReplacement.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <algorithm>

#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Refactoring.h"

#include "llvm/Support/raw_os_ostream.h"

#include "Utility.h"
#include "llvm/Support/raw_os_ostream.h"
#include <cassert>
#include <fstream>
using namespace clang::dpct;
using namespace llvm;
namespace path = llvm::sys::path;
namespace fs = llvm::sys::fs;

namespace clang {
namespace tooling {
std::string getFormatSearchPath();
}
}

static bool formatFile(StringRef FileName,
                       const std::vector<clang::tooling::Range> &Ranges,
                       clang::SourceManager &SM) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> ErrorOrMemoryBuffer =
      MemoryBuffer::getFileAsStream(FileName);
  if (std::error_code EC = ErrorOrMemoryBuffer.getError()) {
    return false;
  }

  std::unique_ptr<llvm::MemoryBuffer> FileBuffer =
      std::move(ErrorOrMemoryBuffer.get());
  if (FileBuffer->getBufferSize() == 0)
    return false;

  clang::format::FormattingAttemptStatus Status;
  StringRef StyleStr = "file"; // DPCTFormatStyle::custom
  if (clang::dpct::DpctGlobalInfo::getFormatStyle() ==
      DPCTFormatStyle::google) {
    StyleStr = "google";
  } else if (clang::dpct::DpctGlobalInfo::getFormatStyle() ==
             DPCTFormatStyle::llvm) {
    StyleStr = "llvm";
  }
  std::string StyleSearchPath = clang::tooling::getFormatSearchPath().empty()
                                    ? clang::dpct::DpctGlobalInfo::getInRoot()
                                    : clang::tooling::getFormatSearchPath();
  llvm::Expected<clang::format::FormatStyle> StyleOrErr =
      clang::format::getStyle(StyleStr, StyleSearchPath, "llvm",
                              FileBuffer->getBuffer());
  clang::format::FormatStyle Style;
  if (!StyleOrErr) {
    PrintMsg(llvm::toString(StyleOrErr.takeError()) + "\n");
    PrintMsg("Using LLVM style as fallback formatting style.\n");
    clang::format::FormatStyle FallbackStyle = clang::format::getNoStyle();
    getPredefinedStyle("llvm", clang::format::FormatStyle::LanguageKind::LK_Cpp,
                       &FallbackStyle);
    Style = FallbackStyle;
  } else {
    Style = StyleOrErr.get();
  }
  if (clang::dpct::DpctGlobalInfo::getFormatRange() ==
      clang::format::FormatRange::migrated) {
    Style.AllowShortFunctionsOnASingleLine =
        clang::format::FormatStyle::SFS_None;
    if (clang::dpct::DpctGlobalInfo::getGuessIndentWidthMatcherFlag())
      Style.IndentWidth = clang::dpct::DpctGlobalInfo::getIndentWidth();
  }

  clang::Rewriter Rewrite(SM, clang::LangOptions());

  if (DpctGlobalInfo::getFormatRange() == clang::format::FormatRange::all) {
    std::vector<clang::tooling::Range> AllLineRanges;
    AllLineRanges.push_back(clang::tooling::Range(
        /*Offest*/ 0, /*Length*/ FileBuffer.get()->getBufferSize()));
    clang::tooling::Replacements Replaces = clang::format::sortIncludes(
        Style, FileBuffer->getBuffer(), AllLineRanges, FileName);
    auto ChangedCode =
        clang::tooling::applyAllReplacements(FileBuffer->getBuffer(), Replaces);
    if (!ChangedCode) {
      PrintMsg(llvm::toString(ChangedCode.takeError()) + "\n");
      return false;
    }
    AllLineRanges = clang::tooling::calculateRangesAfterReplacements(
        Replaces, AllLineRanges);
    clang::tooling::Replacements FormatChanges = reformat(
        Style, FileBuffer->getBuffer(), AllLineRanges, FileName, &Status);
    Replaces = Replaces.merge(FormatChanges);
    clang::tooling::applyAllReplacements(Replaces, Rewrite);
  } else {
    // only format migrated lines
    clang::tooling::Replacements FormatChanges =
        reformat(Style, FileBuffer->getBuffer(), Ranges, FileName, &Status);
    clang::tooling::applyAllReplacements(FormatChanges, Rewrite);
  }
  Rewrite.overwriteChangedFiles();
  return true;
}

// TODO: it's global variable,  refine in future.
std::map<std::string, bool> IncludeFileMap;

static void rewriteDir(SmallString<512> &FilePath, const StringRef InRoot,
                       const StringRef OutRoot) {
  assert(isCanonical(InRoot) && "InRoot must be a canonical path.");
  assert(isCanonical(FilePath) && "FilePath must be a canonical path.");

  SmallString<512> InRootAbs;
  SmallString<512> OutRootAbs;
  SmallString<512> FilePathAbs;
  std::error_code EC;
  bool InRootAbsValid = true;
  EC = llvm::sys::fs::real_path(InRoot, InRootAbs);
  if ((bool)EC) {
    InRootAbsValid = false;
  }
  bool OutRootAbsValid = true;
  EC = llvm::sys::fs::real_path(OutRoot, OutRootAbs);
  if ((bool)EC) {
    OutRootAbsValid = false;
  }
  bool FilePathAbsValid = true;
  EC = llvm::sys::fs::real_path(FilePath, FilePathAbs);
  if ((bool)EC) {
    FilePathAbsValid = false;
  }

#if defined(_WIN64)
  std::string LocalFilePath = StringRef(FilePath).lower();
  std::string LocalInRoot =
      InRootAbsValid ? InRootAbs.str().lower() : InRoot.lower();
  std::string LocalOutRoot =
      OutRootAbsValid ? OutRootAbs.str().lower() : OutRoot.lower();
#elif defined(__linux__)
  std::string LocalFilePath =
      FilePathAbsValid ? FilePathAbs.c_str() : StringRef(FilePath);
  std::string LocalInRoot = InRootAbsValid ? InRootAbs.c_str() : InRoot;
  std::string LocalOutRoot = OutRootAbsValid ? OutRootAbs.c_str() : OutRoot;
#else
#error Only support windows and Linux.
#endif

  auto PathDiff = mismatch(path::begin(LocalFilePath), path::end(LocalFilePath),
                           path::begin(LocalInRoot));
  SmallString<512> NewFilePath = StringRef(LocalOutRoot);
  path::append(NewFilePath, PathDiff.first, path::end(LocalFilePath));
  FilePath = NewFilePath;
}

static void rewriteFileName(SmallString<512> &FilePath) {
  SourceProcessType FileType = GetSourceFileType(FilePath.str());

  if (FileType & TypeCudaSource) {
    path::replace_extension(FilePath, "dp.cpp");
  } else if (FileType & TypeCppSource) {
    auto Extension = path::extension(FilePath);
    path::replace_extension(FilePath, Extension + ".dp.cpp");
  } else if (FileType & TypeCudaHeader) {
    path::replace_extension(FilePath, "dp.hpp");
  }
}

/// Apply all generated replacements, and immediately save the results to files
/// in output directory.
///
/// \returns 0 upon success. Non-zero upon failure.
/// Prerequisite: InRoot and OutRoot are both absolute paths
int saveNewFiles(clang::tooling::RefactoringTool &Tool, StringRef InRoot,
                 StringRef OutRoot) {
  assert(isCanonical(InRoot) && "InRoot must be a canonical path.");
  using namespace clang;
  ProcessStatus status = MigrationSucceeded;
  // Set up Rewriter.
  LangOptions DefaultLangOptions;
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticPrinter DiagnosticPrinter(llvm::errs(), &*DiagOpts);
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), &*DiagOpts,
      &DiagnosticPrinter, false);
  SourceManager Sources(Diagnostics, Tool.getFiles());
  Rewriter Rewrite(Sources, DefaultLangOptions);

  SmallString<512> OutPath;

  bool AppliedAll = true;
  if (Tool.getReplacements().empty()) {
    // There are no rules applying on the *.cpp files,
    // dpct just do nothing with them.
    status = MigrationNoCodeChangeHappen;
  } else {
    std::unordered_map<std::string, std::vector<clang::tooling::Range>>
        FileRangesMap;
    // There are matching rules for *.cpp files ,*.cu files, also header files
    // included, migrate these files into *.dp.cpp files.
    for (auto &Entry : groupReplacementsByFile(
             Rewrite.getSourceMgr().getFileManager(), Tool.getReplacements())) {

      OutPath = StringRef(Entry.first);
      makeCanonical(OutPath);
      auto Find = IncludeFileMap.find(OutPath.c_str());
      if (Find != IncludeFileMap.end()) {
        IncludeFileMap[OutPath.c_str()] = true;
      }

      // This operation won't fail; it already succeeded once during argument
      // validation.
      makeCanonical(OutPath);
      rewriteDir(OutPath, InRoot, OutRoot);

      rewriteFileName(OutPath);

      // for headfile, as it can be included from differnt file, it need
      // merge the migration triggered by each including.
      if (OutPath.back() == 'h') {
        // note the replacement of Entry.second are updated by this call.
        mergeExternalReps(std::string(OutPath.str()), Entry.second);
      }

      std::vector<clang::tooling::Range> Ranges;
      Ranges = calculateRangesWithFormatFlag(Entry.second);
      FileRangesMap.insert(std::make_pair(OutPath.str(), Ranges));

      std::error_code EC;
      EC = fs::create_directories(path::parent_path(OutPath));
      if ((bool)EC) {
        std::string ErrMsg =
            "[ERROR] Create file : " + std::string(OutPath.str()) +
            " fail: " + EC.message() + "\n";
        status = MigrationSaveOutFail;
        PrintMsg(ErrMsg);
        return status;
      }
      // std::ios::binary prevents ofstream::operator<< from converting \n to
      // \r\n on windows.
      std::ofstream File(OutPath.str(), std::ios::binary);
      llvm::raw_os_ostream Stream(File);
      if (!File) {
        std::string ErrMsg =
            "[ERROR] Create file: " + std::string(OutPath.str()) + " fail.\n";
        PrintMsg(ErrMsg);
        status = MigrationSaveOutFail;
        return status;
      }

      AppliedAll =
          tooling::applyAllReplacements(Entry.second, Rewrite) || AppliedAll;
      Rewrite
          .getEditBuffer(Sources.getOrCreateFileID(
              Tool.getFiles().getFile(Entry.first).get(),
              clang::SrcMgr::C_User /*normal user code*/))
          .write(Stream);
    }
    if (DpctGlobalInfo::getFormatRange() != clang::format::FormatRange::none) {
      clang::format::setFormatRangeGetterHandler(
          clang::dpct::DpctGlobalInfo::getFormatRange);
      bool FormatResult = true;
      for (auto Iter : FileRangesMap) {
        FormatResult =
            formatFile(Iter.first, Iter.second, Sources) && FormatResult;
      }
      if (!FormatResult) {
        PrintMsg("[Warning] Error happened while formatting. Generating "
                 "unformatted code.\n");
      }
    }
  }

  // The necessary header files which have no no replacements will be copied to
  // "-out-root" directory
  for (const auto &Entry : IncludeFileMap) {
    SmallString<512> FilePath = StringRef(Entry.first);
    if (!Entry.second) {
      makeCanonical(FilePath);
      rewriteDir(FilePath, InRoot, OutRoot);
      if (fs::exists(FilePath)) {
        // A header file with this name already exists.
        llvm::errs() << "File '" << FilePath
                     << "' already exists; skipping it.\n";
        continue;
      }

      std::error_code EC;
      EC = fs::create_directories(path::parent_path(FilePath));
      if ((bool)EC) {
        std::string ErrMsg =
            "[ERROR] Create file: " + std::string(FilePath.str()) +
            " fail: " + EC.message() + "\n";
        status = MigrationSaveOutFail;
        PrintMsg(ErrMsg);
        return status;
      }
      // std::ios::binary prevents ofstream::operator<< from converting \n to
      // \r\n on windows.
      std::ofstream File(FilePath.str(), std::ios::binary);

      if (!File) {
        std::string ErrMsg =
            "[ERROR] Create file: " + std::string(FilePath.str()) +
            " failed.\n";
        status = MigrationSaveOutFail;
        PrintMsg(ErrMsg);
        return status;
      }

      llvm::raw_os_ostream Stream(File);

      Rewrite
          .getEditBuffer(Sources.getOrCreateFileID(
              Tool.getFiles().getFile(Entry.first).get(),
              clang::SrcMgr::C_User /*normal user code*/))
          .write(Stream);
    }
  }

  if (!AppliedAll) {
    llvm::errs() << "Skipped some replacements.\n";
    status = MigrationSkipped;
  }
  return status;
}
