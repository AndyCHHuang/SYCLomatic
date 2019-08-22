//===--- DPCT.cpp -------------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2019 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "clang/DPCT/DPCT.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"

#include "ASTTraversal.h"
#include "AnalysisInfo.h"
#include "Config.h"
#include "Debug.h"
#include "SaveNewFiles.h"
#include "Utility.h"
#include "ValidateArguments.h"
#include "GAnalytics.h"
#include <string>

#include "ToolChains/Cuda.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Options.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>

#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "SignalProcess.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::dpct;
using namespace clang::tooling;

using namespace llvm::cl;

const char *const CtHelpMessage =
    "\n"
    "<source0> ... Paths of input source files. These paths are\n"
    "\tlooked up in the compilation database. If the path of a source file is\n"
    "\tabsolute, it must exist in the CMake source tree. If the path is\n"
    "\trelative, the current working directory must exist in the CMake\n"
    "\tsource tree and the path must be a subdirectory of the current\n"
    "\tworking directory. \"./\" prefixes in a relative path will be\n"
    "\tautomatically removed.  The remainder of a relative path must be a\n"
    "\tsuffix of a path in the compilation database.\n"
    "\n";

const char *const CtHelpHint =
    "  Warning: Please specify file(s) to be migrated.\n"
    "  Get help on oneAPI DPC++ Compatibility Tool, run: dpct --help\n"
    "\n";

static OptionCategory DPCTCat("oneAPI DPC++ Compatibility Tool");
static extrahelp CommonHelp(CtHelpMessage);
static opt<std::string> Passes(
    "passes",
    desc("Comma separated list of migration passes, which will be applied.\n"
         "Only the specified passes are applied."),
    value_desc("FunctionAttrsRule,..."), cat(DPCTCat), llvm::cl::Hidden);
static opt<std::string> InRoot(
    "in-root", desc("Directory path for root of source tree to be migrated.\n"
                    "Only files under this root will be migrated."),
    value_desc("/path/to/input/root/"), cat(DPCTCat), llvm::cl::Optional);
static opt<std::string>
    OutRoot("out-root", desc("Directory path for root of generated files.\n"
                             "Directory will be created if it doesn't exist."),
            value_desc("/path/to/output/root/"), cat(DPCTCat),
            llvm::cl::Optional);

static opt<std::string> SDKIncludePath(
    "cuda-include-path",
    desc("Directory path of CUDA header files.\n"
         "If this option is set, option \"--cuda-path\" will be ignored."),
    value_desc("/path/to/CUDA/include/"), cat(DPCTCat), llvm::cl::Optional);

static opt<std::string> ReportType(
    "report-type",
    desc("Comma separated list of report types.\n"
         "\"apis\": Information about API signatures that need migration and\n"
         "  the number of times they were encountered.\n"
         "  The report file name will have \".apis\" suffix added.\n"
         "\"stats\": High level migration statistics;  Lines "
         " Of Code (LOC) \n  migrated to DPC++, LOC migrated to Compatibility "
         "API,\n  LOC not needing migration,  LOC needing migration, but not "
         "migrated.\n"
         "  The report file name will have \".stats\" suffix added.\n"
         "\"all\": Generates all of the above reports.\n"
         "Default is \"stats\"."),
    value_desc("[all|apis|stats]"), cat(DPCTCat), llvm::cl::Optional);

static opt<std::string>
    ReportFormat("report-format",
                 desc("Format of reports:\n\"csv\": Output is lines of comma "
                      "separated values.\n"
                      "  Report file name extension will be \".csv\".\n"
                      "\"formatted\": Output is formatted to be easier to read "
                      "by human eyes.\n"
                      "  Report file name extension will be \".log\".\n"
                      "Default is \"csv\".\n"),
                 value_desc("[csv|formatted]"), cat(DPCTCat),
                 llvm::cl::Optional);

static opt<std::string> ReportFilePrefix(
    "report-file-prefix",
    desc("Prefix for the report file names.\nThe full file name will have a "
         "suffix "
         "derived from the report-type\nand an extension derived from the "
         "report-format.\n"
         "For example: <prefix>.apis.csv or <prefix>.stats.log.\n"
         "If this option is not specified, the report will go "
         "to stdout.\nThe report files are created in the "
         "directory, specified by -out-root.\nDefault is stdout."),
    value_desc("prefix"), cat(DPCTCat), llvm::cl::Optional);
bool ReportOnlyFlag = false;
static opt<bool, true> ReportOnly(
    "report-only",
    llvm::cl::desc("Only reports are generated.  No DPC++ code is generated.\n"
                   "Default is to generate both reports and DPC++ code."),
    cat(DPCTCat), llvm::cl::location(ReportOnlyFlag));

bool KeepOriginalCodeFlag = false;

static opt<bool, true>
    ShowOrigCode("keep-original-code",
                 llvm::cl::desc("Keep original code in comments of generated "
                                "DPC++ files.\nDefault: off"),
                 cat(DPCTCat), llvm::cl::location(KeepOriginalCodeFlag));

static opt<std::string>
    DiagsContent("report-diags-content",
                 desc("Diagnostics verbosity level. \"pass\": Basic migration "
                      "pass information. "
                      "\"transformation\": Detailed migration pass "
                      "transformation information."),
                 value_desc("[pass|transformation]"), cat(DPCTCat),
                 llvm::cl::Optional, llvm::cl::Hidden);

static std::string WarningDesc("Comma separated list of warnings to "
                               " suppress.\nValid warning ids range from " +
                               std::to_string((size_t)Warnings::BEGIN) +
                               " to " +
                               std::to_string((size_t)Warnings::END - 1));
opt<std::string> SuppressWarnings("suppress-warnings", desc(WarningDesc),
                                  value_desc("WarningID,..."), cat(DPCTCat));

bool SuppressWarningsAllFlag = false;
static std::string WarningAllDesc("Suppress all warnings");
opt<bool, true> SuppressWarningsAll("suppress-warnings-all",
                                    desc(WarningAllDesc), cat(DPCTCat),
                                    location(SuppressWarningsAllFlag));

bool NoStopOnErrFlag = false;

static opt<bool, true>
    NoStopOnErr("no-stop-on-err",
                llvm::cl::desc("Continue migration and report generation after "
                               "possible errors.\nDefault: off"),
                cat(DPCTCat), llvm::cl::location(NoStopOnErrFlag));

opt<OutputVerbosityLev> OutputVerbosity(
    "output-verbosity", llvm::cl::desc("Set the output verbosity level:"),
    llvm::cl::values(
        clEnumVal(silent, "Only messages from clang"),
        clEnumVal(normal,
                  "Only warnings, errors, notes from both clang and dpct"),
        clEnumVal(detailed,
                  "Normal + messages about start and end of file parsing"),
        clEnumVal(diagnostics,
                  "Everything, as now - which includes "
                  "information about conflicts,\n\t\t\t\t\tseg faults, "
                  "etc.... This one is default.")),
    llvm::cl::init(diagnostics), cat(DPCTCat), llvm::cl::Optional);

opt<std::string> OutputFile(
    "output-file", desc("redirects stdout/stderr output to <file> in the\n"
                        "output diretory specified by '-out-root' option."),
    value_desc("output file name"), cat(DPCTCat), llvm::cl::Optional);

std::string CudaPath;          // Global value for the CUDA install path.
std::string DpctInstallPath; // Installation directory for this tool

class DPCTConsumer : public ASTConsumer {
public:
  DPCTConsumer(ReplTy &R, CompilerInstance &CI, StringRef InFile)
      : ATM(CI, InRoot), Repl(R), PP(CI.getPreprocessor()), CI(CI) {
    int RequiredRType = 0;
    SourceProcessType FileType = GetSourceFileType(InFile);

    if (FileType & (TypeCudaSource | TypeCudaHeader)) {
      RequiredRType = ApplyToCudaFile;
    } else if (FileType & (TypeCppSource | TypeCppHeader)) {
      RequiredRType = ApplyToCppFile;
    }

    if (Passes != "") {
      // Separate string into list by comma
      auto Names = split(Passes, ',');

      std::vector<std::vector<std::string>> Rules;

      for (auto const &Name : Names) {
        auto *ID = ASTTraversalMetaInfo::getID(Name);
        auto MapEntry = ASTTraversalMetaInfo::getConstructorTable()[ID];
        auto RuleObj = (TranslationRule *)MapEntry();
        CommonRuleProperty RuleProperty = RuleObj->GetRuleProperty();
        auto RType = RuleProperty.RType;
        auto RulesDependon = RuleProperty.RulesDependon;

        // Add rules should be run on the source file
        if (RType & RequiredRType) {
          std::vector<std::string> Vec;
          Vec.push_back(Name);
          for (auto const &RuleName : RulesDependon) {
            Vec.push_back(RuleName);
          }
          Rules.push_back(Vec);
        }
      }

      std::vector<std::string> SortedRules = ruleTopoSort(Rules);
      for (std::vector<std::string>::reverse_iterator it = SortedRules.rbegin();
           it != SortedRules.rend(); it++) {
        auto *RuleID = ASTTraversalMetaInfo::getID(*it);
        if (!RuleID) {
          const std::string ErrMsg = "[ERROR] Rule\"" + *it + "\" not found\n";
          PrintMsg(ErrMsg);
          llvm_unreachable(ErrMsg.c_str());
        }
        ATM.emplaceTranslationRule(RuleID);
      }

    } else {
      ATM.emplaceAllRules(RequiredRType);
    }
  }

  void HandleTranslationUnit(ASTContext &Context) override {
    // The migration process is separated into two stages:
    // 1) Analysis of AST and identification of applicable migration rules
    // 2) Generation of actual textual Replacements
    // Such separation makes it possible to post-process the list of identified
    // migration rules before applying them.
    ATM.matchAST(Context, TransformSet, SSM);

    auto &Global = DpctGlobalInfo::getInstance();
    for (const auto &I : TransformSet) {
      auto Repl = I->getReplacement(Context);
      Global.addReplacement(Repl);

      // TODO: Need to print debug info here
    }
  }

  void Initialize(ASTContext &Context) override {
    // Set Context for build information
    DpctGlobalInfo::setCompilerInstance(CI);

    PP.addPPCallbacks(llvm::make_unique<IncludesCallbacks>(
        TransformSet, Context.getSourceManager(), ATM));
  }

  ~DPCTConsumer() {
    // Clean EmittedTransformations for input file migrated.
    ASTTraversalMetaInfo::getEmittedTransformations().clear();
  }

private:
  ASTTraversalManager ATM;
  TransformSetTy TransformSet;
  StmtStringMap SSM;
  ReplTy &Repl;
  Preprocessor &PP;
  CompilerInstance &CI;
};

class DPCTAction : public ASTFrontendAction {
  ReplTy &Repl;

public:
  DPCTAction(ReplTy &R) : Repl(R) {}

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    return llvm::make_unique<DPCTConsumer>(Repl, CI, InFile);
  }

  bool usesPreprocessorOnly() const override { return false; }
};

// Object of this class will be handed to RefactoringTool::run and will create
// the Action.
class DPCTActionFactory : public FrontendActionFactory {
  ReplTy &Repl;

public:
  DPCTActionFactory(ReplTy &R) : Repl(R) {}
  FrontendAction *create() override { return new DPCTAction{Repl}; }
};

std::string getCudaInstallPath(int argc, const char **argv) {
  std::vector<const char *> Argv;
  Argv.reserve(argc);
  // do not copy "--" so the driver sees a possible --cuda-path option
  std::copy_if(argv, argv + argc, back_inserter(Argv),
               [](const char *s) { return std::strcmp(s, "--"); });
  // Remove the redundant prefix "--extra-arg=" so that
  // CudaInstallationDetector can find correct path.
  for (unsigned int i = 0; i < Argv.size(); i++) {
    if (strncmp(argv[i], "--extra-arg=--cuda-path", 23) == 0) {
      Argv[i] = argv[i] + 12;
    }
  }

  // Output parameters to indicate errors in parsing. Not checked here,
  // OptParser will handle errors.
  unsigned MissingArgIndex, MissingArgCount;
  std::unique_ptr<llvm::opt::OptTable> Opts = driver::createDriverOptTable();
  llvm::opt::InputArgList ParsedArgs =
      Opts->ParseArgs(Argv, MissingArgIndex, MissingArgCount);

  // Create minimalist CudaInstallationDetector and return the InstallPath.
  DiagnosticsEngine E(nullptr, nullptr, nullptr, false);
  driver::Driver Driver("", llvm::sys::getDefaultTargetTriple(), E, nullptr);
  driver::CudaInstallationDetector SDKDetector(
      Driver, llvm::Triple(Driver.getTargetTriple()), ParsedArgs);

  std::string Path = SDKDetector.getInstallPath();
  if (!SDKDetector.isValid()) {
      std::string ErrMsg = "[ERROR] Not found valid SDK path\n";
    PrintMsg(ErrMsg);
    exit(MigrationErrorInvalidSDKPath);
  }

  makeCanonical(Path);
  return Path;
}

std::string getInstallPath(clang::tooling::ClangTool &Tool,
                           const char *invokeCommand) {
  SmallString<512> InstalledPath(invokeCommand);

  // Do a PATH lookup, if there are no directory components.
  if (llvm::sys::path::filename(InstalledPath) == InstalledPath) {
    if (llvm::ErrorOr<std::string> Tmp = llvm::sys::findProgramByName(
            llvm::sys::path::filename(InstalledPath.str()))) {
      InstalledPath = *Tmp;
    }
  }

  makeCanonical(InstalledPath);
  StringRef InstalledPathParent(llvm::sys::path::parent_path(InstalledPath));
  // Move up to parent directory of bin directory
  StringRef InstallPath = llvm::sys::path::parent_path(InstalledPathParent);
  return InstallPath.str();
}

// To validate the root path of the project to be migrated.
void ValidateInputDirectory(clang::tooling::RefactoringTool &Tool,
                            std::string &InRoot) {

  if (isChildPath(CudaPath, InRoot) || isSamePath(CudaPath, InRoot)) {
    std::string ErrMsg =
        "[ERROR] Input root specified by \"-in-root\" option \"" + InRoot +
        "\" is in CUDA_PATH folder \"" + CudaPath + "\"\n";
    PrintMsg(ErrMsg);
    exit(MigrationErrorRunFromSDKFolder);
  }

  if (isChildPath(InRoot, DpctInstallPath) ||
      isSamePath(InRoot, DpctInstallPath)) {
    std::string ErrMsg = "[ERROR] Input folder \"" + InRoot +
                         "\" is the parent or the same as the folder where "
                         "oneAPI DPC++ Compatibility Tool is installed \"" +
                         DpctInstallPath + "\"\n";
    PrintMsg(ErrMsg);
    exit(MigrationErrorInRootContainCTTool);
  }
}

unsigned int GetLinesNumber(clang::tooling::RefactoringTool &Tool,
                            StringRef Path) {
  // Set up Rewriter and to get source manager.
  LangOptions DefaultLangOptions;
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticPrinter DiagnosticPrinter(llvm::errs(), &*DiagOpts);
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), &*DiagOpts,
      &DiagnosticPrinter, false);
  SourceManager Sources(Diagnostics, Tool.getFiles());
  Rewriter Rewrite(Sources, DefaultLangOptions);
  SourceManager &SM = Rewrite.getSourceMgr();

  const FileEntry *Entry = SM.getFileManager().getFile(Path);
  if (!Entry) {
    std::string ErrMsg = "FilePath Invalide...\n";
    PrintMsg(ErrMsg);
    llvm_unreachable(ErrMsg.c_str());
  }

  FileID FID = SM.getOrCreateFileID(Entry, SrcMgr::C_User);

  SourceLocation EndOfFile = SM.getLocForEndOfFile(FID);
  unsigned int LineNumber = SM.getSpellingLineNumber(EndOfFile, nullptr);
  return LineNumber;
}

static void printMetrics(clang::tooling::RefactoringTool &Tool) {

  for (const auto &Elem : LOCStaticsMap) {
    unsigned TotalLines = GetLinesNumber(Tool, Elem.first);
    unsigned TransToAPI = Elem.second[0];
    unsigned TransToSYCL = Elem.second[1];
    unsigned NotTrans = TotalLines - TransToSYCL - TransToAPI;
    unsigned NotSupport = Elem.second[2];

    DpctStats() << "\n";
    DpctStats()
        << "File name, LOC migrated to SYCL, LOC migrated to Compatibility "
           "API, LOC not needed to migrate, LOC not able to migrate";
    DpctStats() << "\n";
    DpctStats() << Elem.first + ", " + std::to_string(TransToSYCL) + ", " +
                         std::to_string(TransToAPI) + ", " +
                         std::to_string(NotTrans) + ", " +
                         std::to_string(NotSupport);
    DpctStats() << "\n";
  }
}

static void saveApisReport(void) {
  if (ReportFilePrefix == "stdout") {
    std::string buf;
    llvm::raw_string_ostream OS(buf);
    OS << "------------------APIS report--------------------\n";
    OS << "API name\t\t\t\tFrequency";
    OS << "\n";

    for (const auto &Elem : SrcAPIStaticsMap) {
      std::string APIName = Elem.first;
      unsigned int Count = Elem.second;
      OS << llvm::format("%-30s%16u\n", APIName.c_str(), Count);
    }
    OS << "-------------------------------------------------\n";
    PrintMsg(OS.str());
  } else {
    std::string RFile = OutRoot + "/" + ReportFilePrefix +
                        (ReportFormat == "csv" ? ".apis.csv" : ".apis.log");
    llvm::sys::fs::create_directories(llvm::sys::path::parent_path(RFile));
    // std::ios::binary prevents ofstream::operator<< from converting \n to \r\n
    // on windows.
    std::ofstream File(RFile, std::ios::binary);

    std::string Str;
    llvm::raw_string_ostream Title(Str);
    Title << (ReportFormat == "csv" ? "API name, Frequency"
                                    : "API name\t\t\t\tFrequency");

    File << Title.str() << std::endl;
    for (const auto &Elem : SrcAPIStaticsMap) {
      std::string APIName = Elem.first;
      unsigned int Count = Elem.second;
      if (ReportFormat == "csv") {
        File << APIName << "," << std::to_string(Count) << std::endl;
      } else {
        std::string Str;
        llvm::raw_string_ostream OS(Str);
        OS << llvm::format("%-30s%16u\n", APIName.c_str(), Count);
        File << OS.str();
      }
    }
  }
}

static void saveStatsReport(clang::tooling::RefactoringTool &Tool,
                            double Duration) {

  printMetrics(Tool);
  DpctStats() << "\nTotal migration time: " + std::to_string(Duration) +
                       " ms\n";
  if (ReportFilePrefix == "stdout") {
    std::string buf;
    llvm::raw_string_ostream OS(buf);
    OS << "----------Stats report---------------\n";
    OS << getDpctStatsStr() << "\n";
    OS << "-------------------------------------\n";
    PrintMsg(OS.str());
  } else {
    std::string RFile = OutRoot + "/" + ReportFilePrefix +
                        (ReportFormat == "csv" ? ".stats.csv" : ".stats.log");
    llvm::sys::fs::create_directories(llvm::sys::path::parent_path(RFile));
    // std::ios::binary prevents ofstream::operator<< from converting \n to \r\n
    // on windows.
    std::ofstream File(RFile, std::ios::binary);
    File << getDpctStatsStr() << "\n";
  }
}

static void saveDiagsReport() {

  // DpctDiags() << "\n";
  if (ReportFilePrefix == "stdout") {
    std::string buf;
    llvm::raw_string_ostream OS(buf);
    OS << "--------Diags message----------------\n";
    OS << getDpctDiagsStr() << "\n";
    OS << "-------------------------------------\n";
    PrintMsg(OS.str());
  } else {
    std::string RFile = OutRoot + "/" + ReportFilePrefix + ".diags.log";
    llvm::sys::fs::create_directories(llvm::sys::path::parent_path(RFile));
    // std::ios::binary prevents ofstream::operator<< from converting \n to \r\n
    // on windows.
    std::ofstream File(RFile, std::ios::binary);
    File << getDpctDiagsStr() << "\n";
  }
}

std::string printCTVersion() {

  std::string buf;
  llvm::raw_string_ostream OS(buf);

  OS << "\noneAPI DPC++ Compatibility Tool Version: " << DPCT_VERSION_MAJOR << "."
     << DPCT_VERSION_MINOR << "-" << DPCT_VERSION_PATCH << " codebase:";
  // getClangRepositoryPath() export the machine name of repo in release build.
  // so skip the repo name.
  std::string Path = "";
  std::string Revision = getClangRevision();
  if (!Path.empty() || !Revision.empty()) {
    OS << '(';
    if (!Path.empty())
      OS << Path;
    if (!Revision.empty()) {
      if (!Path.empty())
        OS << ' ';
      OS << Revision;
    }
    OS << ')';
  }

  OS << "\n";
  return OS.str();
}

static void DumpOutputFile(void) {
  // Redirect stdout/stderr output to <file> if option "-output-file" is set
  if (!OutputFile.empty()) {
    std::string FilePath = OutRoot + "/" + OutputFile;
    llvm::sys::fs::create_directories(llvm::sys::path::parent_path(FilePath));
    // std::ios::binary prevents ofstream::operator<< from converting \n to \r\n
    // on windows.
    std::ofstream File(FilePath, std::ios::binary);
    File << getDpctTermStr() << "\n";
  }
}

void PrintReportOnFault(std::string &FaultMsg) {
  PrintMsg(FaultMsg);
  saveApisReport();
  saveDiagsReport();

  std::string FileApis = OutRoot + "/" + ReportFilePrefix +
                         (ReportFormat == "csv" ? ".apis.csv" : ".apis.log");
  std::string FileDiags = OutRoot + "/" + ReportFilePrefix + ".diags.log";

  std::ofstream File;
  File.open(FileApis, std::ios::app);
  if (File) {
    File << FaultMsg;
    File.close();
  }

  File.open(FileDiags, std::ios::app);
  if (File) {
    File << FaultMsg;
    File.close();
  }

  DumpOutputFile();
}

int run(int argc, const char **argv) {

  if (argc < 2) {
    std::cout << CtHelpHint;
    return MigrationNoCodeChangeHappen;
  }
  GAnalytics("");
#if defined(__linux__) || defined(_WIN64)
  InstallSignalHandle();
#endif

#if defined(_WIN64)
  // To support wildcard "*" in source file name in windows.
  llvm::InitLLVM X(argc, argv);
#endif

  // Set hangle for libclangTooling to proccess message for dpct
  clang::tooling::SetPrintHandler(PrintMsg);

  // CommonOptionsParser will adjust argc to the index of "--"
  int OriginalArgc = argc;
  llvm::cl::SetVersionPrinter(
      [](llvm::raw_ostream &OS) { OS << printCTVersion() << "\n"; });
  CommonOptionsParser OptParser(argc, argv, DPCTCat);
  clock_t StartTime = clock();
  if (!makeCanonicalOrSetDefaults(InRoot, OutRoot,
                                  OptParser.getSourcePathList()))
    exit(-1);

  if (!validatePaths(InRoot, OptParser.getSourcePathList()))
    exit(-1);

  int Res = checkSDKIncludePath(SDKIncludePath, RealSDKIncludePath);
  if (Res == -1) {
    exit(-1);
  } else if (Res == 0) {
    IsSetSDKIncludeOption = true;
  }

  bool GenReport = false;
  if (checkReportArgs(ReportType, ReportFormat, ReportFilePrefix,
                      ReportOnlyFlag, GenReport, DiagsContent) == false)
    exit(-1);

  if (GenReport) {
    std::string buf;
    llvm::raw_string_ostream OS(buf);
    OS << "Generate report: "
       << "report-type:" << ReportType << ", report-format:" << ReportFormat
       << ", report-file-prefix:" << ReportFilePrefix << "\n";

    PrintMsg(OS.str());
  }

  CudaPath = getCudaInstallPath(OriginalArgc, argv);
  DPCT_DEBUG_WITH_TYPE(
      "CudaPath", DpctLog() << "Cuda Path found: " << CudaPath << "\n");

  RefactoringTool Tool(OptParser.getCompilations(),
                       OptParser.getSourcePathList());
  DpctInstallPath = getInstallPath(Tool, argv[0]);

  ValidateInputDirectory(Tool, InRoot);
  // Made "-- -x cuda --cuda-host-only -nocudalib" option set by default, .i.e
  // commandline "dpct -in-root ./ -out-root ./ ./topologyQuery.cu  --  -x
  // cuda --cuda-host-only -nocudalib -I../common/inc" became "dpct -in-root
  // ./ -out-root ./ ./topologyQuery.cu  -- -I../common/inc"
  Tool.appendArgumentsAdjuster(
      getInsertArgumentAdjuster("-nocudalib", ArgumentInsertPosition::BEGIN));

  Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
      "--cuda-host-only", ArgumentInsertPosition::BEGIN));

  Tool.appendArgumentsAdjuster(
      getInsertArgumentAdjuster("cuda", ArgumentInsertPosition::BEGIN));

  Tool.appendArgumentsAdjuster(
      getInsertArgumentAdjuster("-x", ArgumentInsertPosition::BEGIN));

#ifdef _WIN32
  Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
      "-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP",
      ArgumentInsertPosition::BEGIN));
  Tool.appendArgumentsAdjuster(
      getInsertArgumentAdjuster("-fms-compatibility-version=19.00.24215.1",
                                ArgumentInsertPosition::BEGIN));
#endif
  DpctGlobalInfo::setInRoot(InRoot);
  DpctGlobalInfo::setCudaPath(CudaPath);
  DpctGlobalInfo::setKeepOriginCode(KeepOriginalCodeFlag);

  DPCTActionFactory Factory(Tool.getReplacements());
  if (int RunResult = Tool.run(&Factory) && !NoStopOnErrFlag) {
    DebugInfo::ShowStatus(RunResult);
    DumpOutputFile();
    return RunResult;
  }

  auto &Global = DpctGlobalInfo::getInstance();
  Global.buildReplacements();
  Global.emplaceReplacements(Tool.getReplacements());

  if (GenReport) {
    // report: apis, stats, diags
    if (ReportType.find("all") != std::string::npos ||
        ReportType.find("apis") != std::string::npos)
      saveApisReport();

    if (ReportType.find("all") != std::string::npos ||
        ReportType.find("stats") != std::string::npos) {
      clock_t EndTime = clock();
      double Duration = (double)(EndTime - StartTime) / (CLOCKS_PER_SEC / 1000);
      saveStatsReport(Tool, Duration);
    }
    // all doesn't include diags.
    if (ReportType.find("diags") != std::string::npos) {
      saveDiagsReport();
    }
    if (ReportOnlyFlag) {
      DumpOutputFile();
      return MigrationSucceeded;
    }
  }
  // if run was successful
  int Status = saveNewFiles(Tool, InRoot, OutRoot);
  DebugInfo::ShowStatus(Status);

  DumpOutputFile();
  return Status;
}
