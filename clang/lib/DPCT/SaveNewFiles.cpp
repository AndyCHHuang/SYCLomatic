//===--- SaveNewFiles.cpp --------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2021 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "SaveNewFiles.h"
#include "AnalysisInfo.h"
#include "Debug.h"
#include "ExternalReplacement.h"
#include "Checkpoint.h"

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
} // namespace clang

extern int FatalErrorCnt;
extern std::map<std::string, uint64_t> ErrorCnt;

static bool formatFile(StringRef FileName,
                       const std::vector<clang::tooling::Range> &Ranges,
                       clang::tooling::Replacements &FormatChanges) {
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
  clang::format::FormatStyle Style = DpctGlobalInfo::getCodeFormatStyle();

  if (clang::format::BlockLevelFormatFlag) {
    if (clang::dpct::DpctGlobalInfo::getFormatRange() ==
        clang::format::FormatRange::migrated) {
      Style.IndentWidth = clang::dpct::DpctGlobalInfo::getKCIndentWidth();
    }
  } else {
    if (clang::dpct::DpctGlobalInfo::getFormatRange() ==
            clang::format::FormatRange::migrated &&
        clang::dpct::DpctGlobalInfo::getGuessIndentWidthMatcherFlag()) {
      Style.IndentWidth = clang::dpct::DpctGlobalInfo::getIndentWidth();
    }
  }

  // Here need new SourceManager. Because SourceManager caches the file buffer,
  // if we use a common SourceManager, the second time format will still act on
  // the fisrt input (the original output of dpct without format), then the
  // result is wrong.
  clang::LangOptions DefaultLangOptions;
  IntrusiveRefCntPtr<clang::DiagnosticOptions> DiagOpts =
      new clang::DiagnosticOptions();
  clang::TextDiagnosticPrinter DiagnosticPrinter(llvm::errs(), &*DiagOpts);
  clang::DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<clang::DiagnosticIDs>(new clang::DiagnosticIDs()),
      &*DiagOpts,
      &DiagnosticPrinter, false);

  clang::FileSystemOptions FSO;
  FSO.WorkingDir = ".";
  clang::FileManager FM(FSO, nullptr);
  clang::SourceManager SM(Diagnostics, FM, false);
  clang::Rewriter Rewrite(SM, clang::LangOptions());
  if (DpctGlobalInfo::getFormatRange() == clang::format::FormatRange::all) {
    std::vector<clang::tooling::Range> AllLineRanges;
    AllLineRanges.push_back(clang::tooling::Range(
        /*Offest*/ 0, /*Length*/ FileBuffer.get()->getBufferSize()));
    FormatChanges = reformat(Style, FileBuffer->getBuffer(), AllLineRanges,
                             FileName, &Status);
  } else {
    // only format migrated lines
    FormatChanges =
        reformat(Style, FileBuffer->getBuffer(), Ranges, FileName, &Status);
  }

  clang::tooling::applyAllReplacements(FormatChanges, Rewrite);
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
      FilePathAbsValid ? FilePathAbs.c_str() : StringRef(FilePath).str();
  std::string LocalInRoot = InRootAbsValid ? InRootAbs.c_str() : InRoot.str();
  std::string LocalOutRoot = OutRootAbsValid ? OutRootAbs.c_str() : OutRoot.str();
#else
#error Only support windows and Linux.
#endif

  auto PathDiff =
      std::mismatch(path::begin(LocalFilePath), path::end(LocalFilePath),
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

void processAllFiles(StringRef InRoot, StringRef OutRoot,
                     std::vector<std::string> &FilesNotProcessed) {
  std::error_code EC;
  for (fs::recursive_directory_iterator Iter(Twine(InRoot), EC), End;
       Iter != End; Iter.increment(EC)) {
    if ((bool)EC) {
      std::string ErrMsg =
          "[ERROR] Access : " + std::string(InRoot.str()) +
          " fail: " + EC.message() + "\n";
      PrintMsg(ErrMsg);
    }

    auto FilePath = Iter->path();

    // Skip output directory if it is in the in-root directory.
    if(isChildOrSamePath(OutRoot.str(), FilePath))
      continue;

    bool IsHidden = false;
    for (path::const_iterator PI = path::begin(FilePath),
                              PE = path::end(FilePath);
         PI != PE; ++PI) {
      StringRef Comp = *PI;
      if (Comp.startswith(".")) {
        IsHidden = true;
        break;
      }
    }

    // Skip hiddlen folder or file whose name begins with ".".
    if (IsHidden) {
      continue;
    }

    if (Iter->type() == fs::file_type::regular_file) {
      SmallString<512> OutputFile = llvm::StringRef(FilePath);
      rewriteDir(OutputFile, InRoot, OutRoot);
      if (IncludeFileMap.find(FilePath) != IncludeFileMap.end()) {
        // Skip the files processed by the the first loop of
        // calling proccessFiles() in Tooling.cpp::ClangTool::run().
        continue;
      } else {
        if (GetSourceFileType(FilePath) & TypeCudaSource) {
          // Only migrates isolated CUDA source files.
          FilesNotProcessed.push_back(FilePath);
        } else {
          // Copy the rest files to the output directory.
          std::ifstream In(FilePath);

          auto Parent = path::parent_path(OutputFile);
          std::error_code EC;
          EC = fs::create_directories(Parent);
          if ((bool)EC) {
            std::string ErrMsg =
                "[ERROR] Create Directory : " + std::string(Parent.str()) +
                " fail: " + EC.message() + "\n";
            PrintMsg(ErrMsg);
          }

          std::ofstream Out(OutputFile.c_str());
          if (Out.fail()) {
            std::string ErrMsg =
                "[ERROR] Create file : " + std::string(OutputFile.c_str()) +
                " failure!\n";
            PrintMsg(ErrMsg);
          }
          Out << In.rdbuf();
          Out.close();
          In.close();
        }
      }

    } else if (Iter->type() == fs::file_type::directory_file) {
      const auto Path = Iter->path();
      SmallString<512> OutDirectory = llvm::StringRef(Path);
      rewriteDir(OutDirectory, InRoot, OutRoot);

      if (fs::exists(OutDirectory))
        continue;

      std::error_code EC;
      EC = fs::create_directories(OutDirectory);
      if ((bool)EC) {
        std::string ErrMsg =
            "[ERROR] Create Directory : " + std::string(OutDirectory.str()) +
            " fail: " + EC.message() + "\n";
        PrintMsg(ErrMsg);
      }
    }
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
  volatile ProcessStatus status = MigrationSucceeded;
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

  // The variable defined here is assist to merge history records.
  std::unordered_map<std::string /*FileName*/,
                     bool /*false:Not processed in current migration*/>
      MainSrcFileMap;

  std::string YamlFile =
      OutRoot.str() + "/" + DpctGlobalInfo::getYamlFileName();
  std::string SrcFile = "MainSrcFiles_placehold";
  auto PreTU = std::make_shared<clang::tooling::TranslationUnitReplacements>();

  if (llvm::sys::fs::exists(YamlFile)) {
    if (loadFromYaml(YamlFile, *PreTU, true) == 0) {
      for (const auto &Repl : PreTU->Replacements) {
        auto &FileRelpsMap = DpctGlobalInfo::getFileRelpsMap();
        FileRelpsMap[Repl.getFilePath().str()].push_back(Repl);
      }
      for (const auto &FileDigest : PreTU->MainSourceFilesDigest) {
        auto &DigestMap = DpctGlobalInfo::getDigestMap();
        DigestMap[FileDigest.first] = FileDigest.second;

        // Mark all the main src files loaded from yaml file are not processed
        // in current migration.
        MainSrcFileMap[FileDigest.first] = false;
      }
    }
  }

  bool AppliedAll = true;
  std::vector<clang::tooling::Replacement> MainSrcFilesRepls;
  std::vector<std::pair<std::string, std::string>> MainSrcFilesDigest;

  if (Tool.getReplacements().empty()) {
    // There are no rules applying on the *.cpp files,
    // dpct just do nothing with them.
    status = MigrationNoCodeChangeHappen;
  } else {
    std::unordered_map<std::string, std::vector<clang::tooling::Range>>
        FileRangesMap;
    std::unordered_map<std::string, std::vector<clang::tooling::Range>>
        FileBlockLevelFormatRangesMap;
    // There are matching rules for *.cpp files ,*.cu files, also header files
    // included, migrate these files into *.dp.cpp files.
    auto GroupResult = groupReplacementsByFile(
        Rewrite.getSourceMgr().getFileManager(), Tool.getReplacements());
    for (auto &Entry : GroupResult) {
      OutPath = StringRef(
          DpctGlobalInfo::removeSymlinks(Rewrite.getSourceMgr().getFileManager(), Entry.first));
      makeCanonical(OutPath);
      bool HasRealReplacements = true;
      auto Repls = Entry.second;

      if (Repls.size() == 1) {
        auto Repl = *Repls.begin();
        if(Repl.getLength() == 0 && Repl.getReplacementText().empty())
          HasRealReplacements = false;
      }
      auto Find = IncludeFileMap.find(OutPath.c_str());
      if (HasRealReplacements && Find != IncludeFileMap.end()) {
        IncludeFileMap[OutPath.c_str()] = true;
      }

      // This operation won't fail; it already succeeded once during argument
      // validation.
      makeCanonical(OutPath);
      rewriteFileName(OutPath);
      rewriteDir(OutPath, InRoot, OutRoot);

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
      std::ofstream File(OutPath.str().str(), std::ios::binary);
      llvm::raw_os_ostream Stream(File);
      if (!File) {
        std::string ErrMsg =
            "[ERROR] Create file: " + std::string(OutPath.str()) + " fail.\n";
        PrintMsg(ErrMsg);
        status = MigrationSaveOutFail;
        return status;
      }

      // For headfile, as it can be included from differnt file, it need
      // merge the migration triggered by each including.
      // For mainfile, as it can be compiled or preprocessed with different
      // macro defined, it aslo need merge the migration triggered by each command.
      SourceProcessType FileType = GetSourceFileType(Entry.first);
      if (FileType & (TypeCppHeader | TypeCudaHeader)) {
        mergeExternalReps(Entry.first, OutPath.str().str(), Entry.second);
      } else {

        auto Hash = llvm::sys::fs::md5_contents(Entry.first);
        MainSrcFilesDigest.push_back(
        std::make_pair(Entry.first, Hash->digest().c_str()));

        bool IsMainSrcFileChanged = false;
        std::string FilePath = Entry.first;

        auto &DigestMap = DpctGlobalInfo::getDigestMap();
        auto DigestIter = DigestMap.find(Entry.first);
        if (DigestIter != DigestMap.end()) {
          auto Digest = llvm::sys::fs::md5_contents(Entry.first);
          if (DigestIter->second != Digest->digest().c_str())
            IsMainSrcFileChanged = true;
        }

        auto &FileRelpsMap = dpct::DpctGlobalInfo::getFileRelpsMap();
        auto Iter = FileRelpsMap.find(Entry.first);
        if (Iter != FileRelpsMap.end() && !IsMainSrcFileChanged ) {
          const auto &PreRepls = Iter->second;
          mergeAndUniqueReps(Entry.second, PreRepls);
        }

        // Mark current migrating main src file processed.
        MainSrcFileMap[Entry.first] = true;

        for (const auto &Repl : Entry.second) {
          MainSrcFilesRepls.push_back(Repl);
        }
      }

      std::vector<clang::tooling::Range> Ranges;
      Ranges = calculateRangesWithFormatFlag(Entry.second);
      FileRangesMap.insert(std::make_pair(OutPath.str().str(), Ranges));

      std::vector<clang::tooling::Range> BlockLevelFormatRanges;
      BlockLevelFormatRanges =
          calculateRangesWithBlockLevelFormatFlag(Entry.second);
      FileBlockLevelFormatRangesMap.insert(
          std::make_pair(OutPath.str().str(), BlockLevelFormatRanges));

      AppliedAll =
          tooling::applyAllReplacements(Entry.second, Rewrite) || AppliedAll;
      Rewrite
          .getEditBuffer(Sources.getOrCreateFileID(
              Tool.getFiles().getFile(Entry.first).get(),
              clang::SrcMgr::C_User /*normal user code*/))
          .write(Stream);
    }

    generateHelperFunctions();

    // Save history repls to yaml file.
    auto &FileRelpsMap = DpctGlobalInfo::getFileRelpsMap();
    for (const auto &Entry : FileRelpsMap) {
      if (MainSrcFileMap[Entry.first])
        continue;
      for (const auto &Repl : Entry.second) {
          MainSrcFilesRepls.push_back(Repl);
      }
    }

    // Save history main src file and its content md5 hash to yaml file.
    auto &DigestMap = DpctGlobalInfo::getDigestMap();
    for (const auto &Entry : DigestMap) {
      if (!MainSrcFileMap[Entry.first]) {
        MainSrcFilesDigest.push_back(std::make_pair(Entry.first, Entry.second));
      }
    }

    save2Yaml(YamlFile, SrcFile, MainSrcFilesRepls, MainSrcFilesDigest);

    extern bool ProcessAllFlag;
    // Print the in-root path and the number of processed files
    size_t ProcessedFileNumber;
    if (ProcessAllFlag) {
      ProcessedFileNumber = IncludeFileMap.size();
    } else {
      ProcessedFileNumber = GroupResult.size();
    }
    std::string ReportMsg = "Processed " + std::to_string(ProcessedFileNumber) +
                            " file(s) in -in-root folder \"" + InRoot.str() + "\"";
    std::string ErrorFileMsg;
    int ErrNum=0;
    for (const auto& KV : ErrorCnt) {
      if(KV.second!=0) {
        ErrNum++;
        ErrorFileMsg += "  " + KV.first + ": ";
        if(KV.second & 0xffffffff) {
           ErrorFileMsg += std::to_string(KV.second & 0xffffffff) + " parsing error(s)";
        }
        if((KV.second & 0xffffffff) && ((KV.second>>32) & 0xffffffff))
            ErrorFileMsg += ", ";
        if((KV.second>>32) & 0xffffffff) {
            ErrorFileMsg += std::to_string((KV.second>>32) & 0xffffffff) + " segmentation fault(s) ";
        }
        ErrorFileMsg += "\n";
      }
    }
    if(ErrNum) {
        ReportMsg += ", " + std::to_string(ErrNum) + " file(s) with error(s):\n";
        ReportMsg +=ErrorFileMsg;
    } else {
        ReportMsg +="\n";
    }

    ReportMsg += "\n";
    ReportMsg += DiagRef;

    PrintMsg(ReportMsg);

    int RetJmp = 0;
    CHECKPOINT_FORMATTING_CODE_ENTRY(RetJmp);
    if (RetJmp == 0) {
      try {
        if (DpctGlobalInfo::getFormatRange() !=
            clang::format::FormatRange::none) {
          clang::format::setFormatRangeGetterHandler(
              clang::dpct::DpctGlobalInfo::getFormatRange);
          bool FormatResult = true;
          for (auto Iter : FileRangesMap) {
            clang::tooling::Replacements FormatChanges;
            FormatResult = formatFile(Iter.first, Iter.second, FormatChanges) &&
                           FormatResult;

            // If range is "all", one file only need to be formated once.
            if (DpctGlobalInfo::getFormatRange() ==
                clang::format::FormatRange::all)
              continue;

            auto BlockLevelFormatIter =
                FileBlockLevelFormatRangesMap.find(Iter.first);
            if (BlockLevelFormatIter != FileBlockLevelFormatRangesMap.end()) {
              clang::format::BlockLevelFormatFlag = true;

              std::vector<clang::tooling::Range>
                  BlockLevelFormatRangeAfterFisrtFormat =
                      calculateUpdatedRanges(FormatChanges,
                                             BlockLevelFormatIter->second);
              FormatResult = formatFile(BlockLevelFormatIter->first,
                                        BlockLevelFormatRangeAfterFisrtFormat,
                                        FormatChanges) &&
                             FormatResult;

              clang::format::BlockLevelFormatFlag = false;
            }
          }
          if (!FormatResult) {
            PrintMsg("[Warning] Error happened while formatting. Generating "
                     "unformatted code.\n");
          }
        }
      } catch (std::exception &e) {
        std::string FaultMsg =
            "Error: dpct internal error. Intel(R) DPC++ Compatibility Tool "
            "skips formatting the code and continues migration.\n";
        llvm::errs() << FaultMsg;
      }
    }
    CHECKPOINT_FORMATTING_CODE_EXIT();
  }

  // The necessary header files which have no no replacements will be copied to
  // "-out-root" directory.
  for (const auto &Entry : IncludeFileMap) {
    SmallString<512> FilePath = StringRef(Entry.first);
    if (!Entry.second) {
      makeCanonical(FilePath);

      // Awalys migrate *.cuh files to *.dp.hpp files,
      // Awalys migrate *.cu files to *.dp.cpp files.
      SourceProcessType FileType = GetSourceFileType(FilePath.str());
      if (FileType & TypeCudaHeader) {
        path::replace_extension(FilePath, "dp.hpp");
      } else if(FileType & TypeCudaSource) {
        path::replace_extension(FilePath, "dp.cpp");
      }

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
      std::ofstream File(FilePath.str().str(), std::ios::binary);

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

void loadYAMLIntoFileInfo(std::string Path) {
  SmallString<512> SourceFilePath(Path);

  SourceFilePath = StringRef(DpctGlobalInfo::removeSymlinks(
      DpctGlobalInfo::getFileManager(), Path));
  makeCanonical(SourceFilePath);

  std::string OriginPath = SourceFilePath.str().str();
  rewriteFileName(SourceFilePath);
  rewriteDir(SourceFilePath, DpctGlobalInfo::getInRoot(),
             DpctGlobalInfo::getOutRoot());

  std::string YamlFilePath = SourceFilePath.str().str() + ".yaml";
  auto PreTU = std::make_shared<clang::tooling::TranslationUnitReplacements>();
  if (fs::exists(YamlFilePath)) {
    if (loadFromYaml(std::move(YamlFilePath), *PreTU, false) == 0) {
      DpctGlobalInfo::getInstance().insertReplInfoFromYAMLToFileInfo(OriginPath,
                                                                     PreTU);
    }
  }
}
