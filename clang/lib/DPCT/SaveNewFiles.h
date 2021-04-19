//===--- SaveNewFiles.h --------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2021 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//

#ifndef DPCT_SAVE_NEW_FILES_H
#define DPCT_SAVE_NEW_FILES_H

#include "ValidateArguments.h"
#include "llvm/Support/Error.h"
#include <map>

#define DiagRef \
"See Diagnostics Reference to resolve warnings and complete the migration:\n"\
"https://software.intel.com/content/www/us/en/develop/documentation/"\
"intel-dpcpp-compatibility-tool-user-guide/top/diagnostics-reference.html\n"

namespace llvm {
class StringRef;
}

namespace clang {
namespace tooling {
class RefactoringTool;
}
} // namespace clang

/// ProcessStatus defines various statuses of dpct workflow
enum ProcessStatus {
  MigrationSucceeded = 0,
  MigrationNoCodeChangeHappen = 1,
  MigrationSkipped = 2,
  MigrationError = -1,
  MigrationSaveOutFail = -2, /*eg. have no write permission*/
  MigrationErrorRunFromSDKFolder = -3,
  MigrationErrorInRootContainCTTool = -4,
  MigrationErrorInvalidCudaIncludePath = -5,
  MigrationErrorInvalidInRootOrOutRoot = -6,
  MigrationErrorInvalidInRootPath = -7,
  MigrationErrorInvalidFilePath = -8,
  MigrationErrorInvalidReportArgs = -9,
  VcxprojPaserFileNotExist = -11,
  VcxprojPaserCreateCompilationDBFail = -12, /*eg. hav no write permission*/
  MigrationErrorInvalidInstallPath = -13,
  MigrationErrorPathTooLong = -14,
  MigrationErrorInvalidWarningID = -15,
  MigrationOptionParsingError = -16,
  MigrationErrorFileParseError = -17,
  MigrationErrorShowHelp = -18,
  MigrationErrorCannotFindDatabase = -19,
  MigrationErrorCannotParseDatabase = -20,
  MigrationErrorNoExplicitInRoot = -21,
  MigrationSKIPForMissingCompileCommand = -22,
  MigrationErrorSpecialCharacter = -23,
  MigrationErrorPrefixTooLong = -25,
  MigrationErrorNoFileTypeAvail = -27,
  MigrationErrorInRootContainSDKFolder = -28,
  MigrationErrorCannotAccessDirInDatabase = -29,
  MigrationErrorInconsistentFileInDatabase = -30,
  MigrationErrorCudaVersionUnsupported = -31,
  MigrationErrorSupportedCudaVersionNotAvailable = -32,
  MigrationErrorInvalidExplicitNamespace = -33,
  MigrationErrorCustomHelperFileNameContainInvalidChar = -34,
  MigrationErrorCustomHelperFileNameTooLong = -35,
  MigrationErrorCustomHelperFileNamePathTooLong = -36,
};

/// Apply all generated replacements, and immediately save the results to
/// files in output directory.
///
/// \returns 0 upon success. Non-zero upon failure.
/// Prerequisite: InRoot and OutRoot are both absolute paths
int saveNewFiles(clang::tooling::RefactoringTool &Tool, llvm::StringRef InRoot,
                 llvm::StringRef OutRoot);

void loadYAMLIntoFileInfo(std::string Path);

// std::string:  source file name including path.
// bool: false: the source file has no replacement.
//       true:  the source file has replacement.
extern std::map<std::string, bool> IncludeFileMap;

// This function is registered by SetFileProcessHandle() called by runDPCT() in
// DPCT.cpp, and called in Tooling.cpp::DoFileProcessHandle(). It traverses all
// the files in directory \pInRoot, collecting *.cu files not
// processed by the the first loop of calling proccessFiles() in
// Tooling.cpp::ClangTool::run()) into \pFilesNotProcessed, and copies the rest
// files to the output directory.
void processAllFiles(llvm::StringRef InRoot, llvm::StringRef OutRoot,
                     std::vector<std::string> &FilesNotProcessed);

#endif // DPCT_SAVE_NEW_FILES_H
