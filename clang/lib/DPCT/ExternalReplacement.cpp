//===--- ExternalReplacement.cpp ------------------------*- C++-*---===//
//
// Copyright (C) 2018 - 2021 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
//  This file target to process replacement based operation:
//   -save replacement to external(disk file)
//   -load replacement from external(disk file)
//   -merage replacement in current migration with previous migration.

#include "Utility.h"
#include "AnalysisInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include "ExternalReplacement.h"
#include "clang/Tooling/Core/Diagnostic.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/ReplacementsYaml.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_os_ostream.h"

#include <algorithm>
#include <cassert>
#include <fstream>

using namespace llvm;
namespace path = llvm::sys::path;
namespace fs = llvm::sys::fs;
using clang::tooling::Replacements;

int save2Yaml(StringRef YamlFile, StringRef SrcFileName,
              const std::vector<clang::tooling::Replacement> &Replaces,
              const std::vector<std::pair<std::string, std::string>>
                  &MainSrcFilesDigest) {
  std::string YamlContent;
  llvm::raw_string_ostream YamlContentStream(YamlContent);
  llvm::yaml::Output YAMLOut(YamlContentStream);

  // list all the replacement.
  clang::tooling::TranslationUnitReplacements TUR;
  TUR.MainSourceFile = SrcFileName.str();
  TUR.Replacements.insert(TUR.Replacements.end(), Replaces.begin(),
                          Replaces.end());

  TUR.MainSourceFilesDigest.insert(TUR.MainSourceFilesDigest.end(),
                                   MainSrcFilesDigest.begin(),
                                   MainSrcFilesDigest.end());

  clang::dpct::DpctGlobalInfo::updateTUR(TUR);
  TUR.DpctVersion = getDpctVersionStr();
  TUR.MainHelperFileName =
      clang::dpct::DpctGlobalInfo::getCustomHelperFileName();
  YAMLOut << TUR;
  YamlContentStream.flush();
  // std::ios::binary prevents ofstream::operator<< from converting \n to \r\n
  // on windows.
  std::ofstream File(YamlFile.str(), std::ios::binary);
  llvm::raw_os_ostream Stream(File);
  Stream << YamlContent;
  return 0;
}

int loadFromYaml(StringRef Input,
                 clang::tooling::TranslationUnitReplacements &TU,
                 bool OverwriteHelperFilesInfo) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Buffer =
      llvm::MemoryBuffer::getFile(Input);
  if (!Buffer) {
    llvm::errs() << "error: failed to read " << Input << ": "
                 << Buffer.getError().message() << "\n";
    return -1;
  }

  llvm::yaml::Input YAMLIn(Buffer.get()->getBuffer());
  YAMLIn >> TU;

  // Do not return if YAMLIn.error(), we still need set other values.

  if (OverwriteHelperFilesInfo) {
    clang::dpct::DpctGlobalInfo::updateHelperNameContentMap(TU);
    if (!TU.MainHelperFileName.empty() &&
        TU.MainHelperFileName !=
            clang::dpct::DpctGlobalInfo::getCustomHelperFileName()) {
      clang::dpct::PrintMsg(
          "[WARNING] The cunstom helper file name in current migration "
          "is different from the name in previous migration, you need "
          "update the previous migrated code.");
    }
    emitDpctVersionWarningIfNeed(TU.DpctVersion);
  }

  return 0;
}

void mergeAndUniqueReps(Replacements &Replaces,
                        const std::vector<clang::tooling::Replacement> &PreRepls) {

  bool DupFlag = false;
  for (const auto &OldR : PreRepls) {
    DupFlag = false;

    for (const auto &CurrR : Replaces) {

      if (CurrR.getFilePath() != OldR.getFilePath()) {
        llvm::errs() << "Ignore " << OldR.getFilePath().str()
                     << " for differnt path!\n";
        return;
      }
      if ((CurrR.getFilePath() == OldR.getFilePath()) &&
          (CurrR.getOffset() == OldR.getOffset()) &&
          (CurrR.getLength() == OldR.getLength()) &&
          (CurrR.getReplacementText() == OldR.getReplacementText())) {
        DupFlag = true;
        break;
      }
    }
    if (DupFlag == false) {
      if (auto Err = Replaces.add(OldR)) {
        llvm::dbgs() << "Adding replacement when merging previous "
                        "replacement: Error occured!\n"
                     << Err << "\n";
      }
    }
  }
}

int mergeExternalReps(std::string InRootSrcFilePath,
                      std::string OutRootSrcFilePath, Replacements &Replaces) {
  std::string YamlFile = OutRootSrcFilePath + ".yaml";

  auto PreTU = clang::dpct::DpctGlobalInfo::getInstance()
                   .getReplInfoFromYAMLSavedInFileInfo(InRootSrcFilePath);

  if (PreTU) {
    llvm::errs() << YamlFile << " exist, try to merge it.\n";

    mergeAndUniqueReps(Replaces, (*PreTU).Replacements);
  }

  llvm::errs() << "Saved new version of " << YamlFile << " file\n";

  std::vector<clang::tooling::Replacement> Repls(Replaces.begin(), Replaces.end());

  std::vector<std::pair<std::string, std::string>> MainSrcFilesDigest;
  save2Yaml(std::move(YamlFile), std::move(OutRootSrcFilePath), Repls, MainSrcFilesDigest);
  return 0;
}
