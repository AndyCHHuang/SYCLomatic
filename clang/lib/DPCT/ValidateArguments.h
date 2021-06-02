//===--- ValidateArguments.h ---------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2021 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//

#ifndef DPCT_VALIDATE_ARGUMENTS_H
#define DPCT_VALIDATE_ARGUMENTS_H

#include <string>
#include <vector>

#if defined(_WIN32)
#define MAX_PATH_LEN _MAX_PATH
#define MAX_NAME_LEN _MAX_FNAME
#else
#define MAX_PATH_LEN PATH_MAX
#define MAX_NAME_LEN NAME_MAX
#endif

/// The enum that specifies the level of Unified Shared Memory, only
/// two levels are supported currrently.
/// none:       uses helper functions from DPCT header files for memory
///             management migration
/// restricted: uses API from DPC++ Explicit and Restricted Unified
///             Shared Memory extension for memory management migration
enum class UsmLevel { none, restricted };
/// OutputVerbosityLev defines various verbosity levels for dpct reports
enum class OutputVerbosityLev { silent, normal, detailed, diagnostics };
enum class DPCTFormatStyle { llvm, google, custom };
enum class ReportFormatEnum { notsetformat, csv, formatted };
enum class HelperFilesCustomizationLevel { none, file, all, api };
enum class ReportTypeEnum { notsettype, apis, stats, all, diags };
enum class AssumedNDRangeDimEnum : unsigned int { dim1 = 1, dim3 = 3 };
enum class ExplicitNamespace { none, cl, sycl, sycl_math, dpct };
enum class DPCPPExtensions { submit_barrier };
bool makeInRootCanonicalOrSetDefaults(
    std::string &InRoot, const std::vector<std::string> SourceFiles);
bool makeOutRootCanonicalOrSetDefaults(std::string &OutRoot);

/// Make sure files passed to Intel(R) DPC++ Compatibility Tool are under the
/// input root directory and have an extension.
/// return value:
/// 0: success (InRoot and SourceFiles are valid)
/// -1: fail for InRoot not valid or there is file SourceFiles not in InRoot
/// -2: fail for there is file in SourceFiles without extension
int validatePaths(const std::string &InRoot,
                  const std::vector<std::string> &SourceFiles);
bool checkReportArgs(ReportTypeEnum &RType, ReportFormatEnum &RFormat,
                     std::string &RFile, bool &ROnly, bool &GenReport,
                     std::string &DVerbose);

/// Retrun value:
///  0: Path is valid
///  1: Path is empty, option SDK include path is not used
/// -1: Path is invaild
int checkSDKPathOrIncludePath(const std::string &Path, std::string &RealPath);

void validateCustomHelperFileNameArg(HelperFilesCustomizationLevel Level,
                                     std::string &Name,
                                     const std::string &OutRoot);
#endif // DPCT_VALIDATE_ARGUMENTS_H
