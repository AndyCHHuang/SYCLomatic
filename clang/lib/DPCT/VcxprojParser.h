//===--- VcxprojParser.h -------------------------------*- C++ -*---===//
//
// Copyright (C) 2019 - 2021 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//

#ifndef DPCT_VCXPROJPARSER_H
#define DPCT_VCXPROJPARSER_H

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>

/// Parses \p VcxprojFile file to generate compilation database
/// "compile_commands.json" in the building directory \p BuildPath.
///
/// \param BuildDir user's building path.
/// \param VcxprojFile vcxproj file path.
void vcxprojParser(std::string &BuildPath, std::string &VcxprojFile);

#endif // DPCT_VCXPROJPARSER_H
