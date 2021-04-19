//===--- ValidateArguments.cpp -------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2021 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//

#include "ValidateArguments.h"
#include "Debug.h"
#include "Utility.h"
#include "SaveNewFiles.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <cassert>

using namespace llvm;
using namespace std;
namespace path = llvm::sys::path;
namespace fs = llvm::sys::fs;

// Set OutRoot to the current working directory.
static bool getDefaultOutRoot(std::string &OutRootPar) {
  SmallString<256> OutRoot;
  if (fs::current_path(OutRoot) != std::error_code()) {
    llvm::errs() << "Could not get current path.\n";
    return false;
  }
  OutRoot.append("/dpct_output");
  if (fs::is_directory(OutRoot)) {
    std::error_code EC;
    fs::directory_iterator Iter(OutRoot, EC);
    if ((bool)EC) {
      llvm::errs() << "Could not access output directory.\n";
      return false;
    }
    fs::directory_iterator End;
    if (Iter != End) {
      llvm::errs() << "dpct_output directory is not empty. Please use option"
                      " \"--out-root\" to set output directory.\n";
      return false;
    } else {
      clang::dpct::PrintMsg(
          "The directory \"dpct_output\" is used as \"out-root\"\n");
    }
  } else {
    std::error_code EC = fs::create_directory(OutRoot, false);
    if ((bool)EC) {
      llvm::errs() << "Could not create dpct_output directory.\n";
      return false;
    }
    clang::dpct::PrintMsg(
        "The directory \"dpct_output\" is used as \"out-root\"\n");
  }
  OutRootPar.assign(begin(OutRoot), end(OutRoot));
  return true;
}

// If input source files exist in the commandline,
// set InRoot to the directory of the first input source file.
// If input source file does not exist,
// set InRoot to ".".
static bool getDefaultInRoot(std::string &InRootPar,
                             const vector<string> &SourceFiles) {
  if (SourceFiles.size() == 0) {
    InRootPar = ".";
    return true;
  }

  SmallString<256> InRoot = StringRef(SourceFiles.front());
  // Remove the last component from path.
  path::remove_filename(InRoot);
  if (!makeCanonical(InRoot))
    return false;

  InRootPar.assign(begin(InRoot), end(InRoot));
  return true;
}

bool makeInRootCanonicalOrSetDefaults(
    string &InRoot, const std::vector<std::string> SourceFiles) {
  if (InRoot.empty()) {
    if (!getDefaultInRoot(InRoot, SourceFiles))
      return false;
  } else if (!makeCanonical(InRoot)) {
    return false;
  }
  if (fs::get_file_type(InRoot) != fs::file_type::directory_file) {
    llvm::errs() << "Error: '" << InRoot << "' is not a directory.\n";
    return false;
  }

  SmallString<512> InRootAbs;
  std::error_code EC = llvm::sys::fs::real_path(InRoot, InRootAbs);
  if ((bool)EC) {
    clang::dpct::DebugInfo::ShowStatus(MigrationErrorInvalidInRootPath);
    dpctExit(MigrationErrorInvalidInRootPath);
  }
  InRoot = InRootAbs.str().str();
  return true;
}

bool makeOutRootCanonicalOrSetDefaults(string &OutRoot) {
  if (OutRoot.empty()) {
    if (!getDefaultOutRoot(OutRoot))
      return false;
  } else if (!makeCanonical(OutRoot)) {
    return false;
  }
  return true;
}

// Make sure all files have an extension and are under InRoot.
//
// TODO: Produce diagnostics with llvm machinery
int validatePaths(const std::string &InRoot,
                   const std::vector<std::string> &SourceFiles) {
  assert(isCanonical(InRoot) && "InRoot must be a canonical path.");
  int Ok = 0;
  for (const auto &FilePath : SourceFiles) {
    auto AbsPath = FilePath;
    if (!makeCanonical(AbsPath)) {
      Ok = -1;
      continue;
    }

    if (!isChildPath(InRoot, AbsPath)) {
      Ok = -1;
      llvm::errs() << "Error: File '" << AbsPath
                   << "' is not under the specified input root directory '"
                   << InRoot << "'\n";
    }

    if (!path::has_extension(AbsPath)) {
      Ok = -2;
      llvm::errs() << "Error: File '" << AbsPath
                   << "' does not have an extension.\n";
    }
  }

  return Ok;
}

int checkSDKPathOrIncludePath(const std::string &Path, std::string &RealPath) {
  if (Path.empty()) {
    return 1;
  }
  SmallString<512> AbsPath;
  auto EC = llvm::sys::fs::real_path(Path, AbsPath);
  if ((bool)EC) {
    return -1;
  }

#if defined(_WIN32)
  RealPath = AbsPath.str().lower();
  if (RealPath.size() >= 3 && RealPath.substr(0, 3) == "unc") {
    RealPath = "\\" + RealPath.substr(3);
  }
#elif defined(__linux__)
  RealPath = AbsPath.c_str();
#else
#error Only support windows and Linux.
#endif
  return 0;
}

bool checkReportArgs(ReportTypeEnum &RType, ReportFormatEnum &RFormat,
                     std::string &RFile, bool &ROnly, bool &GenReport,
                     std::string &DVerbose) {
  bool Success = true;
  if (ROnly || !RFile.empty() || !DVerbose.empty() ||
      RType != ReportTypeEnum::notsettype ||
      RFormat != ReportFormatEnum::notsetformat) {
    GenReport = true;
    // check user provided value and give default value if required.
    if (RType == ReportTypeEnum::notsettype) {
      RType = ReportTypeEnum::stats;
    }
    // check the report format value
    if (RFormat == ReportFormatEnum::notsetformat) {
      RFormat = ReportFormatEnum::csv;
    } else if (!(RFormat == ReportFormatEnum::csv ||
                 RFormat == ReportFormatEnum::formatted)) {
      llvm::errs() << "error value provided in option: --report-format, use "
                      "[csv|formatted].\n\n";
      Success = false;
    }
    // check the report file value.
    if (RFile.empty()) {
      RFile = "stdout";
    }
    #ifdef DPCT_DEBUG_BUILD
    // check the report diags content value.
    if (DVerbose.empty()) {
      clang::dpct::VerboseLevel = clang::dpct::VerboseLow;
    } else if (DVerbose == "pass") {
      clang::dpct::VerboseLevel = clang::dpct::VerboseLow;
    } else if (DVerbose == "transformation") {
      clang::dpct::VerboseLevel = clang::dpct::VerboseHigh;
    } else {
      Success = false;
      llvm::errs()
          << "error value provided in option: -report-diags-content, use "
             "[pass|transformation].\n\n";
    }
    #endif
  }

  return Success;
}

void validateCustomHelperFileNameArg(HelperFilesCustomizationLevel Level,
                                     std::string &Name, const std::string &OutRoot) {
  if ((Level == HelperFilesCustomizationLevel::none) && (Name != "dpct")) {
    clang::dpct::PrintMsg("Warning: Ignored \"--custom-helper-file\", since "
                          "\"--use-custom-helper-file\" is not specified or "
                          "has \"none\" value.\n");
    Name = "dpct";
    return;
  }

  std::string FileName = Name + ".hpp";
  if (FileName.size() >= MAX_NAME_LEN) {
    clang::dpct::DebugInfo::ShowStatus(
        MigrationErrorCustomHelperFileNameTooLong);
    dpctExit(MigrationErrorCustomHelperFileNameTooLong);
  }

  std::string FilePath = OutRoot + "/include/" + Name + "/" + FileName;
  if (FilePath.size() >= MAX_PATH_LEN) {
    clang::dpct::DebugInfo::ShowStatus(
        MigrationErrorCustomHelperFileNamePathTooLong, FilePath);
    dpctExit(MigrationErrorCustomHelperFileNamePathTooLong);
  }

  for (size_t Idx = 0, End = Name.size(); Idx < End; Idx++) {
    if ((!isdigit(Name[Idx])) && (!isalpha(Name[Idx])) && (Name[Idx] != '_')) {
      clang::dpct::DebugInfo::ShowStatus(
          MigrationErrorCustomHelperFileNameContainInvalidChar);
      dpctExit(MigrationErrorCustomHelperFileNameContainInvalidChar);
    }
  }
}
