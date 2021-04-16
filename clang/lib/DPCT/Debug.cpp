//===--- Debug.cpp--------- --------------------------------*- C++ -*---===//
////
//// Copyright (C) 2018 - 2021 Intel Corporation. All rights reserved.
////
//// The information and source code contained herein is the exclusive
//// property of Intel Corporation and may not be disclosed, examined
//// or reproduced in whole or in part without explicit written authorization
//// from the company.
////
////===-----------------------------------------------------------------===//
#include "Debug.h"
#include "ASTTraversal.h"
#include "SaveNewFiles.h"

#include <numeric>
#include <unordered_set>

#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace dpct {

// std::string -> file name
// std::array<unsigned int, 3> =>
// array[0]: count LOC(Lines Of Code) to API
// array[1]: count LOC(Lines Of Code) to SYCL
// array[2]: count API not support
std::unordered_map<std::string, std::array<unsigned int, 3>> LOCStaticsMap;

// std::string -> APIName ,types information
// unsigned int -> Times met
std::map<std::string, unsigned int> SrcAPIStaticsMap;

int VerboseLevel = NonVerbose;

void DebugInfo::printMigrationRules(
    const std::vector<std::unique_ptr<ASTTraversal>> &TRs) {
#ifdef DPCT_DEBUG_BUILD
  auto print = [&]() {
    DpctDiags() << "Migration Rules:\n";

    constexpr char Indent[] = "  ";
    if (TRs.empty()) {
      DpctDiags() << Indent << "None\n";
      return;
    }

    size_t NumRules = 0;
    for (auto &TR : TRs) {
      if (auto I = dyn_cast<MigrationRule>(&*TR)) {
        DpctDiags() << Indent << I->getName() << "\n";
        ++NumRules;
      }
    }
    DpctDiags() << "# of MigrationRules: " << NumRules << "\n";
  };

  if (VerboseLevel > NonVerbose) {
    print();
  }

#endif // DPCT_DEBUG_BUILD
}

#ifdef DPCT_DEBUG_BUILD
// Start of debug build
static void printMatchedRulesDebugImpl(
    const std::vector<std::unique_ptr<ASTTraversal>> &MatchedRules) {
  // Verbose level lower than "High" doesn't show migration rules' information
  if (VerboseLevel < VerboseHigh) {
    return;
  }

  for (auto &MR : MatchedRules) {
    if (auto TR = dyn_cast<MigrationRule>(&*MR)) {
      TR->print(DpctDiags());
    }
  }

  for (auto &MR : MatchedRules) {
    if (auto TR = dyn_cast<MigrationRule>(&*MR)) {
      TR->printStatistics(DpctDiags());
    }
  }
}

static void printReplacementsDebugImpl(const TransformSetTy &TS,
                                       ASTContext &Context) {
  // Verbos level lower than "High" doesn't show detailed replacements'
  // information
  if (VerboseLevel < VerboseHigh) {
    return;
  }

  for (auto &TM : TS) {
    TM->print(DpctDiags(), Context);
  }

  std::unordered_map<int, size_t> NameCountMap;
  for (auto &TM : TS) {
    ++(NameCountMap.insert(std::make_pair((int)TM->getID(), 0)).first->second);
  }

  if (NameCountMap.empty())
    return;

  const size_t NumRepls =
      std::accumulate(NameCountMap.begin(), NameCountMap.end(), 0,
                      [](const size_t &a, const std::pair<int, size_t> &obj) {
                        return a + obj.second;
                      });
  for (const auto &Pair : NameCountMap) {
    auto &ID = Pair.first;
    auto &Numbers = Pair.second;
    DpctDiags() << "# of replacement <" << TextModification::TMNameMap.at((int)ID)
                << ">: " << Numbers << " (" << Numbers << "/" << NumRepls
                << ")\n";
  }
}

// End of debug build
#endif

void DebugInfo::printMatchedRules(
    const std::vector<std::unique_ptr<ASTTraversal>> &MatchedRules) {
#ifdef DPCT_DEBUG_BUILD // Debug build
  printMatchedRulesDebugImpl(MatchedRules);
#endif
}

void DebugInfo::printReplacements(const TransformSetTy &TS,
                                  ASTContext &Context) {
#ifdef DPCT_DEBUG_BUILD // Debug build
  printReplacementsDebugImpl(TS, Context);
#endif
}

// Log buffer, default size 4096, when running out of memory, dynamic memory
// allocation is handled by SmallVector internally.
static llvm::SmallVector<char, /* default buffer size */ 4096> DpctLogBuffer;
static llvm::raw_svector_ostream DpctLogStream(DpctLogBuffer);
static llvm::SmallVector<char, /* default buffer size */ 4096> DpctStatsBuffer;
static llvm::raw_svector_ostream DpctStatsStream(DpctStatsBuffer);
static llvm::SmallVector<char, /* default buffer size */ 4096> DpctDiagsBuffer;
static llvm::raw_svector_ostream DpctDiagsStream(DpctDiagsBuffer);

static llvm::SmallVector<char, /* default buffer size */ 4096> DpctTermBuffer;
static llvm::raw_svector_ostream DpctTermStream(DpctTermBuffer);

llvm::raw_ostream &DpctLog() { return DpctLogStream; }
llvm::raw_ostream &DpctStats() { return DpctStatsStream; }
llvm::raw_ostream &DpctDiags() { return DpctDiagsStream; }
llvm::raw_ostream &DpctTerm() { return DpctTermStream; }
std::string getDpctStatsStr() { return DpctStatsStream.str().str(); }
std::string getDpctDiagsStr() { return DpctDiagsStream.str().str(); }
std::string getDpctTermStr() { return DpctTermStream.str().str(); }

void DebugInfo::ShowStatus(int Status, std::string Message) {

  std::string StatusString;
  switch (Status) {
  case MigrationSucceeded:
    StatusString = "Migration process completed";
    break;
  case MigrationNoCodeChangeHappen:
    StatusString = "Migration not necessary; no CUDA code detected";
    break;
  case MigrationSkipped:
    StatusString = "Some migration rules were skipped";
    break;
  case MigrationError:
    StatusString = "An error has occurred during migration";
    break;
  case MigrationSaveOutFail:
    StatusString = "Error: Unable to save the output to the specified directory";
    break;
  case MigrationErrorInvalidCudaIncludePath:
    StatusString = "Error: Path for CUDA header files specified by "
                   "--cuda-include-path is invalid.";
    break;
  case MigrationErrorCudaVersionUnsupported:
    StatusString = "Error: The version of CUDA header files specified by "
                   "--cuda-include-path is not supported. See Release Notes "
                   "for supported versions.";
    break;
  case MigrationErrorSupportedCudaVersionNotAvailable:
    StatusString = "Error: Intel(R) DPC++ Compatibility Tool was not able to "
                   "detect path for CUDA header files. Use --cuda-include-path "
                   "to specify the correct path to the header files.";
    break;
  case MigrationErrorInvalidInRootOrOutRoot:
    StatusString = "Error: The path for --in-root or --out-root is not valid";
    break;
  case MigrationErrorInvalidInRootPath:
    StatusString = "Error: The path for --in-root is not valid";
    break;
  case MigrationErrorInvalidReportArgs:
    StatusString = "Error: The value(s) provided for report option(s) is incorrect.";
    break;
  case MigrationErrorInvalidWarningID:
    StatusString = "Error: Invalid warning ID or range; "
                   "valid warning IDs range from " +
                   std::to_string((size_t)Warnings::BEGIN) + " to " +
                   std::to_string((size_t)Warnings::END - 1);
    break;
  case MigrationOptionParsingError:
    StatusString = "Option parsing error,"
                   " run 'dpct --help' to see supported options and values";
    break;
  case MigrationErrorPathTooLong:
#if defined(_WIN32)
    StatusString = "Error: Path is too long; should be less than _MAX_PATH (" +
                   std::to_string(_MAX_PATH) + ")";
#else
    StatusString = "Error: Path is too long; should be less than PATH_MAX (" +
                   std::to_string(PATH_MAX) + ")";
#endif
    break;
  case MigrationErrorFileParseError:
    StatusString = "Error: Cannot parse input file(s)";
    break;
  case MigrationErrorCannotFindDatabase:
    StatusString = "Error: Cannot find compilation database";
    break;
  case MigrationErrorCannotParseDatabase:
    StatusString = "Error: Cannot parse compilation database";
    break;
  case MigrationErrorNoExplicitInRoot:
    StatusString =
        "Error: The option --process-all requires that the --in-root be "
        "specified explicitly. Use the --in-root option to specify the "
        "directory to be migrated.";
    break;
  case MigrationErrorSpecialCharacter:
    StatusString = "Error: Prefix contains special characters;"
                   " only alphabetical characters, digits and underscore "
                   "character are allowed";
    break;
  case MigrationErrorPrefixTooLong:
    StatusString = "Error: Prefix is too long; should be less than 128 characters";
    break;
  case MigrationErrorNoFileTypeAvail:
    StatusString = "Error: File Type not available for input file";
    break;
  case MigrationErrorInRootContainCTTool:
    StatusString =
        "Error: Input folder is the parent of, or the same folder as, the "
        "installation directory of the Intel(R) DPC++ Compatibility Tool";
    break;
  case MigrationErrorRunFromSDKFolder:
    StatusString = "Error: Input folder specified by --in-root option is "
                   "in the CUDA_PATH folder";
    break;
  case MigrationErrorInRootContainSDKFolder:
    StatusString = "Error: Input folder is the parent of, or the same folder "
                   "as, the CUDA_PATH folder";
    break;
  case MigrationErrorCannotAccessDirInDatabase:
    StatusString = "Error: Cannot access directory \"" + Message +
                   "\" from the compilation database, check if the directory "
                   "exists and can be accessed by the tool.";
          break;
  case MigrationErrorInconsistentFileInDatabase:
    StatusString = "Error: The file name(s) in the \"command\" and \"file\" "
                   "fields of the compilation database are inconsistent:\n" +
                   Message;
    break;
  case MigrationErrorCustomHelperFileNameContainInvalidChar:
    StatusString =
        "Error: Custom helper header file name is invalid. The name can only "
        "contain digits(0-9), underscore(_) or letters(a-zA-Z).";
    break;
  case MigrationErrorCustomHelperFileNameTooLong:
    StatusString =
        "Error: Custom helper header file name is too long.";
    break;
  case MigrationErrorCustomHelperFileNamePathTooLong:
    StatusString = "Error: The path resulted from --out-root and --custom-helper-file "
                   "option values: \"" + Message + "\" is too long.";
    break;
  default:
    DpctLog() << "Unknown error\n";
    dpctExit(-1);
  }

  if (Status != 0) {
    DpctLog() << "dpct exited with code: " << Status << " (" << StatusString
              << ")\n";
  }

  llvm::dbgs() << DpctLogStream.str() << "\n";
  return;
}
// Currently, set IsPrintOnNormal false only at the place where messages about
// start and end of file parsing are produced,
//.i.e in the place "lib/Tooling:int ClangTool::run(ToolAction *Action)".
void PrintMsg(const std::string &Msg, bool IsPrintOnNormal) {
  if (!OutputFile.empty()) {
    //  Redirects stdout/stderr output to <file>
    DpctTerm() << Msg;
  }

  switch (OutputVerbosity) {
  case OutputVerbosityLev::detailed:
  case OutputVerbosityLev::diagnostics:
    llvm::outs() << Msg;
    break;
  case OutputVerbosityLev::normal:
    if (IsPrintOnNormal) {
      llvm::outs() << Msg;
    }
    break;
  case OutputVerbosityLev::silent:
  default:
    break;
  }
}

} // namespace dpct
} // namespace clang
