//===--- ExternalReplacement.h --------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2021 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//

#ifndef __EXTERNAL_REPLACEMENT_H__
#define __EXTERNAL_REPLACEMENT_H__

#include "llvm/ADT/StringRef.h"
#include <map>

namespace llvm {
class StringRef;
}

namespace clang {
namespace tooling {
class RefactoringTool;
class Replacements;
} // namespace tooling
} // namespace clang

int mergeExternalReps(std::string InRootSrcFilePath,
                      std::string OutRootSrcFilePath,
                      clang::tooling::Replacements &Replaces);
int loadFromYaml(llvm::StringRef Input,
                 clang::tooling::TranslationUnitReplacements &TU,
                 bool OverwriteHelpFilesSet);
int save2Yaml(
    llvm::StringRef YamlFile, llvm::StringRef SrcFileName,
    const std::vector<clang::tooling::Replacement> &Replaces,
    const std::vector<std::pair<std::string, std::string>> &MainSrcFilesDigest);
void mergeAndUniqueReps(
    clang::tooling::Replacements &Replaces,
    const std::vector<clang::tooling::Replacement> &PreRepls);

#endif
