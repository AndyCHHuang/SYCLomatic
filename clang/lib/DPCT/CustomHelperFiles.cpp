//===--- CustomHelperFiles.cpp ---------------------------*- C++ -*---===//
//
// Copyright (C) 2021 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "CustomHelperFiles.h"

#include "ASTTraversal.h"
#include "AnalysisInfo.h"
#include "Config.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_os_ostream.h"

#include <fstream>

namespace clang {
namespace dpct {

// Currently, the HelperFeatureEnum is only used to check
// if there is duplicated feature name in each inc file
enum class HelperFeatureEnum : unsigned int {
#include "clang/DPCT/HelperFeatureEnum.inc"
};

void requestFeature(clang::dpct::HelperFileEnum FileID,
                    std::string HelperFunctionName,
                    const std::string &UsedFile) {
  auto Key = std::make_pair(FileID, HelperFunctionName);
  auto Iter = MapNames::HelperNameContentMap.find(Key);
  if (Iter != MapNames::HelperNameContentMap.end()) {
    Iter->second.IsCalled = true;
    Iter->second.CallerSrcFiles.insert(UsedFile);
  } else {
#ifdef DPCT_DEBUG_BUILD
    std::cout << "Unknown feature: File:" << (unsigned int)FileID
              << ", Feature:" << HelperFunctionName << std::endl;
    exit(-999);
#else
    assert(0 && "Unknown requested feature.\n");
#endif
  }
}
void requestFeature(clang::dpct::HelperFileEnum FileID,
                    std::string HelperFunctionName, SourceLocation SL) {
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  auto ExpansionLoc = SM.getExpansionLoc(SL);

  std::string UsedFile = "";
  if (ExpansionLoc.isValid())
    UsedFile = dpct::DpctGlobalInfo::getLocInfo(ExpansionLoc).first;
  requestFeature(FileID, HelperFunctionName, UsedFile);
}
void requestFeature(clang::dpct::HelperFileEnum FileID,
                    std::string HelperFunctionName, const Stmt *Stmt) {
  if (!Stmt)
    return;
  requestFeature(FileID, HelperFunctionName, Stmt->getBeginLoc());
}
void requestFeature(clang::dpct::HelperFileEnum FileID,
                    std::string HelperFunctionName, const Decl *Decl) {
  if (!Decl)
    return;
  requestFeature(FileID, HelperFunctionName, Decl->getBeginLoc());
}

std::string getCopyrightHeader(const clang::dpct::HelperFileEnum File) {
  std::string CopyrightHeader =
      MapNames::HelperNameContentMap.at(std::make_pair(File, "License")).Code;
  if (File == HelperFileEnum::Dpct) {
    std::string Prefix = "//==----";
    std::string Suffix = "-*- C++ -*----------------==//";
    std::string FileName = " " + getCustomMainHelperFileName() + ".hpp ";
    const size_t ColumnLimit = 80;
    size_t NumOfDashes = 0;
    if (Prefix.size() + Suffix.size() + FileName.size() <= ColumnLimit) {
      NumOfDashes =
          ColumnLimit - Prefix.size() - Suffix.size() - FileName.size();
    }

    CopyrightHeader = Prefix + FileName + std::string(NumOfDashes, '-') +
                      Suffix + "\n" + CopyrightHeader;
  }
  replaceEndOfLine(CopyrightHeader);
  return CopyrightHeader;
}

std::pair<std::string, std::string>
getHeaderGuardPair(const clang::dpct::HelperFileEnum File) {
  std::string MacroName = "";
  if (File == HelperFileEnum::Dpct && getCustomMainHelperFileName() != "dpct") {
    MacroName = getCustomMainHelperFileName();
    for (size_t Idx = 0; Idx < MacroName.size(); ++Idx)
      MacroName[Idx] = llvm::toUpper(MacroName[Idx]);
    MacroName = "__" + MacroName + "_HPP__";
  } else {
    MacroName = MapNames::HelperFileHeaderGuardMacroMap.find(File)->second;
  }
  std::pair<std::string, std::string> Pair;
  Pair.first =
      "#ifndef " + MacroName + getNL() + "#define " + MacroName + getNL();
  Pair.second = "#endif // " + MacroName;
  return Pair;
}

void addDependencyIncludeDirectives(
    const clang::dpct::HelperFileEnum FileID,
    std::vector<clang::dpct::HelperFunc> &ContentVec) {

  auto isDplFile = [](clang::dpct::HelperFileEnum FileID) -> bool {
    if (FileID == clang::dpct::HelperFileEnum::DplExtrasAlgorithm ||
        FileID == clang::dpct::HelperFileEnum::DplExtrasFunctional ||
        FileID == clang::dpct::HelperFileEnum::DplExtrasIterators ||
        FileID == clang::dpct::HelperFileEnum::DplExtrasMemory ||
        FileID == clang::dpct::HelperFileEnum::DplExtrasNumeric ||
        FileID == clang::dpct::HelperFileEnum::DplExtrasVector) {
      return true;
    }
    return false;
  };

  bool IsCurrentFileInDpExtra = isDplFile(FileID);

  auto Iter = MapNames::HelperNameContentMap.find(
      std::make_pair(FileID, "local_include_dependency"));
  if (Iter == MapNames::HelperNameContentMap.end())
    return;

  auto Content = Iter->second;

  std::set<clang::dpct::HelperFileEnum> FileDependency;
  for (const auto &Item : ContentVec) {
    for (const auto &Pair : Item.Dependency) {
      if (clang::dpct::DpctGlobalInfo::getHelperFilesCustomizationLevel() ==
          HelperFilesCustomizationLevel::HFCL_API) {
        if (Pair.second == HelperFeatureDependencyKind::HFDK_UsmNone) {
          if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None)
            FileDependency.insert(Pair.first.first);
        } else if (Pair.second ==
                   HelperFeatureDependencyKind::HFDK_UsmRestricted) {
          if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted)
            FileDependency.insert(Pair.first.first);
        } else {
          FileDependency.insert(Pair.first.first);
        }
      } else {
        FileDependency.insert(Pair.first.first);
      }
    }
  }
  std::string Directives;
  for (const auto &Item : FileDependency) {
    if (Item == FileID)
      continue;
    if (IsCurrentFileInDpExtra) {
      if (isDplFile(Item))
        Directives = Directives + "#include \"" +
                     MapNames::HelperFileNameMap.at(Item) + "\"" + getNL();
      else
        Directives = Directives + "#include \"../" +
                     MapNames::HelperFileNameMap.at(Item) + "\"" + getNL();
    } else {
      Directives = Directives + "#include \"" +
                   MapNames::HelperFileNameMap.at(Item) + "\"" + getNL();
    }
  }
  Content.Code = Directives;
  ContentVec.push_back(Content);
}

std::string getCode(const HelperFunc &Item) {
  if (clang::dpct::DpctGlobalInfo::getHelperFilesCustomizationLevel() ==
      HelperFilesCustomizationLevel::HFCL_File) {
    return Item.Code;
  } else {
    // API level
    if (dpct::DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted) {
      if (!Item.USMCode.empty())
        return Item.USMCode;
    } else {
      if (!Item.NonUSMCode.empty())
        return Item.NonUSMCode;
    }
    return Item.Code;
  }
}

std::string
getHelperFileContent(const clang::dpct::HelperFileEnum File,
                     std::vector<clang::dpct::HelperFunc> ContentVec) {
  if (ContentVec.empty())
    return "";

  std::string ContentStr;

  ContentStr = ContentStr + getCopyrightHeader(File) + getNL();
  ContentStr = ContentStr + getHeaderGuardPair(File).first + getNL();

  if (File != clang::dpct::HelperFileEnum::Dpct &&
      File != clang::dpct::HelperFileEnum::DplUtils) {
    // For Dpct and DplUtils, the include directives are determined
    // by other files.
    addDependencyIncludeDirectives(File, ContentVec);
  }

  auto CompareAsc = [](clang::dpct::HelperFunc A, clang::dpct::HelperFunc B) {
    return A.PositionIdx < B.PositionIdx;
  };
  std::sort(ContentVec.begin(), ContentVec.end(), CompareAsc);

  std::string CurrentNamespace;
  for (const auto &Item : ContentVec) {
    if (Item.Namespace.empty()) {
      // no namespace
      if (CurrentNamespace == "dpct") {
        ContentStr = ContentStr + "} // namespace dpct" + getNL() + getNL();
      } else if (CurrentNamespace == "dpct::detail") {
        ContentStr = ContentStr + "} // namespace detail" + getNL() + getNL();
        ContentStr = ContentStr + "} // namespace dpct" + getNL() + getNL();
      } else if (CurrentNamespace == "dpct::internal") {
        ContentStr = ContentStr + "} // namespace internal" + getNL() + getNL();
        ContentStr = ContentStr + "} // namespace dpct" + getNL() + getNL();
      }
      CurrentNamespace = "";
      std::string Code = getCode(Item);
      replaceEndOfLine(Code);
      ContentStr = ContentStr + Code + getNL();
    } else if (Item.Namespace == "dpct") {
      // dpct namespace
      if (CurrentNamespace.empty()) {
        ContentStr = ContentStr + "namespace dpct {" + getNL() + getNL();
      } else if (CurrentNamespace == "dpct::detail") {
        ContentStr = ContentStr + "} // namespace detail" + getNL() + getNL();
      } else if (CurrentNamespace == "dpct::internal") {
        ContentStr = ContentStr + "} // namespace internal" + getNL() + getNL();
      }
      CurrentNamespace = "dpct";
      std::string Code = getCode(Item);
      replaceEndOfLine(Code);
      ContentStr = ContentStr + Code + getNL();
    } else if (Item.Namespace == "dpct::detail") {
      // dpct::detail namespace
      if (CurrentNamespace.empty()) {
        ContentStr = ContentStr + "namespace dpct {" + getNL() + getNL();
        ContentStr = ContentStr + "namespace detail {" + getNL() + getNL();
      } else if (CurrentNamespace == "dpct") {
        ContentStr = ContentStr + "namespace detail {" + getNL() + getNL();
      }
      CurrentNamespace = "dpct::detail";
      std::string Code = getCode(Item);
      replaceEndOfLine(Code);
      ContentStr = ContentStr + Code + getNL();
    } else if (Item.Namespace == "dpct::internal") {
      // dpct::internal namespace
      if (CurrentNamespace.empty()) {
        ContentStr = ContentStr + "namespace dpct {" + getNL() + getNL();
        ContentStr = ContentStr + "namespace internal {" + getNL() + getNL();
      } else if (CurrentNamespace == "dpct") {
        ContentStr = ContentStr + "namespace internal {" + getNL() + getNL();
      }
      CurrentNamespace = "dpct::internal";
      std::string Code = getCode(Item);
      replaceEndOfLine(Code);
      ContentStr = ContentStr + Code + getNL();
    }
  }

  if (CurrentNamespace == "dpct") {
    ContentStr = ContentStr + "} // namespace dpct" + getNL() + getNL();
  } else if (CurrentNamespace == "dpct::detail") {
    ContentStr = ContentStr + "} // namespace detail" + getNL() + getNL();
    ContentStr = ContentStr + "} // namespace dpct" + getNL() + getNL();
  } else if (CurrentNamespace == "dpct::internal") {
    ContentStr = ContentStr + "} // namespace internal" + getNL() + getNL();
    ContentStr = ContentStr + "} // namespace dpct" + getNL() + getNL();
  }

  ContentStr = ContentStr + getHeaderGuardPair(File).second + getNL();
  return ContentStr;
}

std::string getDpctVersionStr() {
  std::string Str;
  llvm::raw_string_ostream OS(Str);
  OS << DPCT_VERSION_MAJOR << "." << DPCT_VERSION_MINOR << "."
     << DPCT_VERSION_PATCH;
  return OS.str();
}

void emitDpctVersionWarningIfNeed(const std::string &VersionFromYaml) {
  // If yaml file does not exist, this function will not be called.
  std::string CurrentToolVersion = getDpctVersionStr();

  if (VersionFromYaml.empty()) {
    // This is an increamental migration, and the previous migration used 2021
    // gold update1
    clang::dpct::PrintMsg(
        "NOTE: This is an incremental migration. Previous version of the tool "
        "used: 2021.2.0, current version: " +
        CurrentToolVersion + "." + getNL());
  } else if (VersionFromYaml != CurrentToolVersion) {
    // This is an increamental migration, and the previous migration used gold
    // version
    clang::dpct::PrintMsg(
        "NOTE: This is an incremental migration. Previous version of the tool "
        "used: " +
        VersionFromYaml + ", current version: " + CurrentToolVersion + "." +
        getNL());
  }
  // No previous migration, or previous migration using the same tool version:
  // no warning emitted.
}

void generateAllHelperFiles() {
  std::string ToPath = clang::dpct::DpctGlobalInfo::getOutRoot() + "/include";
  if (!llvm::sys::fs::is_directory(ToPath))
    llvm::sys::fs::create_directory(Twine(ToPath));
  ToPath = ToPath + "/" + getCustomMainHelperFileName();
  if (!llvm::sys::fs::is_directory(ToPath))
    llvm::sys::fs::create_directory(Twine(ToPath));
  if (!llvm::sys::fs::is_directory(Twine(ToPath + "/dpl_extras")))
    llvm::sys::fs::create_directory(Twine(ToPath + "/dpl_extras"));

#define GENERATE_ALL_FILE_CONTENT(FILE_NAME)                                   \
  {                                                                            \
    std::ofstream FILE_NAME##File(                                             \
        ToPath + "/" +                                                         \
            MapNames::HelperFileNameMap.at(                                    \
                clang::dpct::HelperFileEnum::FILE_NAME),                       \
        std::ios::binary);                                                     \
    std::string Code = MapNames::FILE_NAME##AllContentStr;                     \
    replaceEndOfLine(Code);                                                    \
    FILE_NAME##File << Code;                                                   \
    FILE_NAME##File.flush();                                                   \
  }
#define GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(FILE_NAME)                        \
  {                                                                            \
    std::ofstream FILE_NAME##File(                                             \
        ToPath + "/dpl_extras/" +                                              \
            MapNames::HelperFileNameMap.at(                                    \
                clang::dpct::HelperFileEnum::FILE_NAME),                       \
        std::ios::binary);                                                     \
    std::string Code = MapNames::FILE_NAME##AllContentStr;                     \
    replaceEndOfLine(Code);                                                    \
    FILE_NAME##File << Code;                                                   \
    FILE_NAME##File.flush();                                                   \
  }
  GENERATE_ALL_FILE_CONTENT(Atomic)
  GENERATE_ALL_FILE_CONTENT(BlasUtils)
  GENERATE_ALL_FILE_CONTENT(Device)
  GENERATE_ALL_FILE_CONTENT(Dpct)
  GENERATE_ALL_FILE_CONTENT(DplUtils)
  GENERATE_ALL_FILE_CONTENT(Image)
  GENERATE_ALL_FILE_CONTENT(Kernel)
  GENERATE_ALL_FILE_CONTENT(Memory)
  GENERATE_ALL_FILE_CONTENT(Util)
  GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(DplExtrasAlgorithm)
  GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(DplExtrasFunctional)
  GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(DplExtrasIterators)
  GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(DplExtrasMemory)
  GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(DplExtrasNumeric)
  GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(DplExtrasVector)
#undef GENERATE_ALL_FILE_CONTENT
#undef GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT
}

void generateHelperFunctions() {
  auto getUsedAPINum = []() -> size_t {
    size_t Res = 0;
    for (const auto &Item : MapNames::HelperNameContentMap) {
      if (Item.second.IsCalled)
        Res++;
    }
    return Res;
  };

  // dpct.hpp is always exsit, so request its non_local_include_dependency
  // feature
  requestFeature(dpct::HelperFileEnum::Dpct, "non_local_include_dependency",
                 "");
  // 1. add dependent APIs
  size_t UsedAPINum = getUsedAPINum();
  do {
    UsedAPINum = getUsedAPINum();
    std::set<std::pair<HelperFeatureIDTy, std::set<std::string>>> NeedInsert;
    for (const auto &Item : MapNames::HelperNameContentMap) {
      if (Item.second.IsCalled) {
        for (const auto &DepItem : Item.second.Dependency) {
          if (clang::dpct::DpctGlobalInfo::getHelperFilesCustomizationLevel() ==
              HelperFilesCustomizationLevel::HFCL_API) {
            if (DepItem.second == HelperFeatureDependencyKind::HFDK_UsmNone) {
              if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None)
                NeedInsert.insert(
                    std::make_pair(DepItem.first, Item.second.CallerSrcFiles));
            } else if (DepItem.second ==
                       HelperFeatureDependencyKind::HFDK_UsmRestricted) {
              if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted)
                NeedInsert.insert(
                    std::make_pair(DepItem.first, Item.second.CallerSrcFiles));
            } else {
              NeedInsert.insert(
                  std::make_pair(DepItem.first, Item.second.CallerSrcFiles));
            }
          } else {
            NeedInsert.insert(
                std::make_pair(DepItem.first, Item.second.CallerSrcFiles));
          }
        }
      }
    }
    for (const auto &Item : NeedInsert) {
      auto Iter = MapNames::HelperNameContentMap.find(Item.first);
      if (Iter != MapNames::HelperNameContentMap.end()) {
        Iter->second.IsCalled = true;
        Iter->second.CallerSrcFiles.insert(Item.second.begin(),
                                           Item.second.end());
      } else {
#ifdef DPCT_DEBUG_BUILD
        std::cout << "Unknown dependency: File:"
                  << (unsigned int)Item.first.first
                  << ", Feature:" << Item.first.second << std::endl;
        exit(-998);
#else
        assert(0 && "Unknown dependency feature.\n");
#endif
      }
    }
  } while (getUsedAPINum() > UsedAPINum);

  // 2. build info of necessary headers to out-root
  if (clang::dpct::DpctGlobalInfo::getHelperFilesCustomizationLevel() ==
      HelperFilesCustomizationLevel::HFCL_None)
    return;
  else if (clang::dpct::DpctGlobalInfo::getHelperFilesCustomizationLevel() ==
           HelperFilesCustomizationLevel::HFCL_All) {
    generateAllHelperFiles();
    return;
  }

  std::vector<clang::dpct::HelperFunc> AtomicFileContent;
  std::vector<clang::dpct::HelperFunc> BlasUtilsFileContent;
  std::vector<clang::dpct::HelperFunc> DeviceFileContent;
  std::vector<clang::dpct::HelperFunc> DpctFileContent;
  std::vector<clang::dpct::HelperFunc> DplUtilsFileContent;
  std::vector<clang::dpct::HelperFunc> ImageFileContent;
  std::vector<clang::dpct::HelperFunc> KernelFileContent;
  std::vector<clang::dpct::HelperFunc> MemoryFileContent;
  std::vector<clang::dpct::HelperFunc> UtilFileContent;
  std::vector<clang::dpct::HelperFunc> DplExtrasAlgorithmFileContent;
  std::vector<clang::dpct::HelperFunc> DplExtrasFunctionalFileContent;
  std::vector<clang::dpct::HelperFunc> DplExtrasIteratorsFileContent;
  std::vector<clang::dpct::HelperFunc> DplExtrasMemoryFileContent;
  std::vector<clang::dpct::HelperFunc> DplExtrasNumericFileContent;
  std::vector<clang::dpct::HelperFunc> DplExtrasVectorFileContent;

  std::vector<bool> FileUsedFlagVec(
      (unsigned int)clang::dpct::HelperFileEnum::HelperFileEnumTypeSize, false);
  if (clang::dpct::DpctGlobalInfo::getHelperFilesCustomizationLevel() ==
      HelperFilesCustomizationLevel::HFCL_File) {
    // E.g., user code uses API2.
    // HelperFileA: API1(depends on API3), API2
    // HelperFileB: API3
    // In step1, only API2 is enabled. But current config is file, so API1 and
    // API2 are both printed, then we also need print API3.
    // But API1 and API3 are not set "IsCalled" flag, just insert elements into
    // content vector.
    auto getUsedFileNum = [&]() -> size_t {
      size_t Res = 0;
      for (const auto &Item : FileUsedFlagVec) {
        if (Item)
          Res++;
      }
      return Res;
    };

    for (const auto &Item : MapNames::HelperNameContentMap)
      if (Item.second.IsCalled)
        FileUsedFlagVec[size_t(Item.first.first)] = true;
    size_t UsedFileNum = getUsedFileNum();
    do {
      UsedFileNum = getUsedFileNum();
      for (unsigned int FileID = 0;
           FileID < (unsigned int)dpct::HelperFileEnum::HelperFileEnumTypeSize;
           ++FileID) {
        if (!FileUsedFlagVec[FileID])
          continue;
        for (const auto &Item : MapNames::HelperNameContentMap) {
          if (Item.first.first == (dpct::HelperFileEnum)FileID) {
            for (const auto &Dep : Item.second.Dependency) {
              if (clang::dpct::DpctGlobalInfo::
                      getHelperFilesCustomizationLevel() ==
                  HelperFilesCustomizationLevel::HFCL_API) {
                if (Dep.second == HelperFeatureDependencyKind::HFDK_UsmNone) {
                  if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None)
                    FileUsedFlagVec[(unsigned int)Dep.first.first] = true;
                } else if (Dep.second ==
                           HelperFeatureDependencyKind::HFDK_UsmRestricted) {
                  if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted)
                    FileUsedFlagVec[(unsigned int)Dep.first.first] = true;
                } else {
                  FileUsedFlagVec[(unsigned int)Dep.first.first] = true;
                }
              } else {
                FileUsedFlagVec[(unsigned int)Dep.first.first] = true;
              }
            }
          }
        }
      }
    } while (getUsedFileNum() > UsedFileNum);
  }

#define UPDATE_FILE(FILENAME)                                                  \
  case clang::dpct::HelperFileEnum::FILENAME:                                  \
    if (clang::dpct::DpctGlobalInfo::getHelperFilesCustomizationLevel() ==     \
        HelperFilesCustomizationLevel::HFCL_File) {                            \
      FILENAME##FileContent.push_back(Item.second);                            \
    } else if (clang::dpct::DpctGlobalInfo::                                   \
                   getHelperFilesCustomizationLevel() ==                       \
               HelperFilesCustomizationLevel::HFCL_API) {                      \
      if (Item.second.IsCalled)                                                \
        FILENAME##FileContent.push_back(Item.second);                          \
    }                                                                          \
    break;

  for (const auto &Item : MapNames::HelperNameContentMap) {
    if (Item.first.second == "local_include_dependency") {
      // local_include_dependency for dpct and dpl_utils is inserted in step3
      // local_include_dependency for others are inserted in
      // getHelperFileContent()
      continue;
    } else if (Item.first.second == "non_local_include_dependency") {
      // non_local_include_dependency for dpct is inserted here
      // non_local_include_dependency for others is inserted in step3
      if (Item.first.first == clang::dpct::HelperFileEnum::Dpct) {
        DpctFileContent.push_back(Item.second);
      }
      continue;
    } else if (Item.first.second == "License") {
      continue;
    } else if (clang::dpct::DpctGlobalInfo::
                   getHelperFilesCustomizationLevel() ==
               HelperFilesCustomizationLevel::HFCL_File) {
      if (!FileUsedFlagVec[size_t(Item.first.first)])
        continue;
    }

    switch (Item.first.first) {
      UPDATE_FILE(Atomic)
      UPDATE_FILE(BlasUtils)
      UPDATE_FILE(Device)
      UPDATE_FILE(Dpct)
      UPDATE_FILE(DplUtils)
      UPDATE_FILE(Image)
      UPDATE_FILE(Kernel)
      UPDATE_FILE(Memory)
      UPDATE_FILE(Util)
      UPDATE_FILE(DplExtrasAlgorithm)
      UPDATE_FILE(DplExtrasFunctional)
      UPDATE_FILE(DplExtrasIterators)
      UPDATE_FILE(DplExtrasMemory)
      UPDATE_FILE(DplExtrasNumeric)
      UPDATE_FILE(DplExtrasVector)
    default:
      assert(0 && "unknown helper file ID");
    }
  }
#undef UPDATE_FILE

  // 3. prepare folder and insert
  // non_local_include_dependency/local_include_dependency
  std::string ToPath = clang::dpct::DpctGlobalInfo::getOutRoot() + "/include";
  if (!llvm::sys::fs::is_directory(ToPath))
    llvm::sys::fs::create_directory(Twine(ToPath));
  ToPath = ToPath + "/" + getCustomMainHelperFileName();
  if (!llvm::sys::fs::is_directory(ToPath))
    llvm::sys::fs::create_directory(Twine(ToPath));
  if (!DplExtrasAlgorithmFileContent.empty() ||
      !DplExtrasFunctionalFileContent.empty() ||
      !DplExtrasIteratorsFileContent.empty() ||
      !DplExtrasMemoryFileContent.empty() ||
      !DplExtrasNumericFileContent.empty() ||
      !DplExtrasVectorFileContent.empty()) {
    if (!llvm::sys::fs::is_directory(Twine(ToPath + "/dpl_extras")))
      llvm::sys::fs::create_directory(Twine(ToPath + "/dpl_extras"));

    std::string IDDStr;

    // There is an extra function replaceEndOfLine() to convert "\n" to
    // platform speicific EOL for "#include ..." statement. Generally speaking,
    // for new added "#include ..." statement, developer should use "\n" instead
    // of getNL().
#define ADD_INCLUDE_DIRECTIVE_FOR_DPL(FILENAME)                                \
  if (!FILENAME##FileContent.empty()) {                                        \
    FILENAME##FileContent.push_back(MapNames::HelperNameContentMap.at(         \
        std::make_pair(clang::dpct::HelperFileEnum::FILENAME,                  \
                       "non_local_include_dependency")));                      \
    IDDStr = IDDStr + "#include \"dpl_extras/" +                               \
             MapNames::HelperFileNameMap.at(                                   \
                 clang::dpct::HelperFileEnum::FILENAME) +                      \
             "\"\n";                                                           \
  }
    ADD_INCLUDE_DIRECTIVE_FOR_DPL(DplExtrasAlgorithm)
    ADD_INCLUDE_DIRECTIVE_FOR_DPL(DplExtrasFunctional)
    ADD_INCLUDE_DIRECTIVE_FOR_DPL(DplExtrasIterators)
    ADD_INCLUDE_DIRECTIVE_FOR_DPL(DplExtrasMemory)
    ADD_INCLUDE_DIRECTIVE_FOR_DPL(DplExtrasNumeric)
    ADD_INCLUDE_DIRECTIVE_FOR_DPL(DplExtrasVector)
#undef ADD_INCLUDE_DIRECTIVE_FOR_DPL

    auto Item = MapNames::HelperNameContentMap.at(std::make_pair(
        clang::dpct::HelperFileEnum::DplUtils, "local_include_dependency"));
    Item.Code = IDDStr;
    DplUtilsFileContent.push_back(Item);
  }

  if (!DplUtilsFileContent.empty() ||
      MapNames::HelperNameContentMap
          .at(std::make_pair(clang::dpct::HelperFileEnum::DplUtils,
                             "non_local_include_dependency"))
          .IsCalled) {
    DplUtilsFileContent.push_back(MapNames::HelperNameContentMap.at(
        std::make_pair(clang::dpct::HelperFileEnum::DplUtils,
                       "non_local_include_dependency")));
  }

  std::string IDDStr;

#define ADD_INCLUDE_DIRECTIVE(FILENAME)                                        \
  if (!FILENAME##FileContent.empty()) {                                        \
    FILENAME##FileContent.push_back(MapNames::HelperNameContentMap.at(         \
        std::make_pair(clang::dpct::HelperFileEnum::FILENAME,                  \
                       "non_local_include_dependency")));                      \
    IDDStr = IDDStr + "#include \"" +                                          \
             MapNames::HelperFileNameMap.at(                                   \
                 clang::dpct::HelperFileEnum::FILENAME) +                      \
             "\"\n";                                                           \
  }
  ADD_INCLUDE_DIRECTIVE(Atomic)
  ADD_INCLUDE_DIRECTIVE(BlasUtils)
  ADD_INCLUDE_DIRECTIVE(Device)
  // Do not include dpl_utils in dpct.hpp, since there is a bug in dpl_extras
  // files. All those functions are without the "inline" specifier, so there
  // will be a multi definition issue. ADD_INCLUDE_DIRECTIVE(DplUtils)
  ADD_INCLUDE_DIRECTIVE(Image)
  ADD_INCLUDE_DIRECTIVE(Kernel)
  ADD_INCLUDE_DIRECTIVE(Memory)
  ADD_INCLUDE_DIRECTIVE(Util)
#undef ADD_INCLUDE_DIRECTIVE

  auto Item = MapNames::HelperNameContentMap.at(std::make_pair(
      clang::dpct::HelperFileEnum::Dpct, "local_include_dependency"));
  Item.Code = IDDStr;
  DpctFileContent.push_back(Item);

  // 4. generate headers to out-root
#define GENERATE_FILE(FILE_NAME)                                               \
  if (!FILE_NAME##FileContent.empty()) {                                       \
    std::string FILE_NAME##FileContentStr = getHelperFileContent(              \
        clang::dpct::HelperFileEnum::FILE_NAME, FILE_NAME##FileContent);       \
    std::ofstream FILE_NAME##File(                                             \
        ToPath + "/" +                                                         \
            MapNames::HelperFileNameMap.at(                                    \
                clang::dpct::HelperFileEnum::FILE_NAME),                       \
        std::ios::binary);                                                     \
    FILE_NAME##File << FILE_NAME##FileContentStr;                              \
    FILE_NAME##File.flush();                                                   \
  }
#define GENERATE_DPL_EXTRAS_FILE(FILE_NAME)                                    \
  if (!FILE_NAME##FileContent.empty()) {                                       \
    std::string FILE_NAME##FileContentStr = getHelperFileContent(              \
        clang::dpct::HelperFileEnum::FILE_NAME, FILE_NAME##FileContent);       \
    std::ofstream FILE_NAME##File(                                             \
        ToPath + "/dpl_extras/" +                                              \
            MapNames::HelperFileNameMap.at(                                    \
                clang::dpct::HelperFileEnum::FILE_NAME),                       \
        std::ios::binary);                                                     \
    FILE_NAME##File << FILE_NAME##FileContentStr;                              \
    FILE_NAME##File.flush();                                                   \
  }
  GENERATE_FILE(Atomic)
  GENERATE_FILE(BlasUtils)
  GENERATE_FILE(Device)
  GENERATE_FILE(Dpct)
  GENERATE_FILE(DplUtils)
  GENERATE_FILE(Image)
  GENERATE_FILE(Kernel)
  GENERATE_FILE(Memory)
  GENERATE_FILE(Util)
  GENERATE_DPL_EXTRAS_FILE(DplExtrasAlgorithm)
  GENERATE_DPL_EXTRAS_FILE(DplExtrasFunctional)
  GENERATE_DPL_EXTRAS_FILE(DplExtrasIterators)
  GENERATE_DPL_EXTRAS_FILE(DplExtrasMemory)
  GENERATE_DPL_EXTRAS_FILE(DplExtrasNumeric)
  GENERATE_DPL_EXTRAS_FILE(DplExtrasVector)
#undef GENERATE_FILE
#undef GENERATE_DPL_EXTRAS_FILE
}

#define ADD_HELPER_FEATURE_FOR_ENUM_NAMES(TYPE)                                \
  void requestHelperFeatureForEnumNames(const std::string Name, TYPE File) {   \
    auto HelperFeatureIter =                                                   \
        clang::dpct::EnumConstantRule::EnumNamesHelperFeaturesMap.find(Name);  \
    if (HelperFeatureIter !=                                                   \
        clang::dpct::EnumConstantRule::EnumNamesHelperFeaturesMap.end()) {     \
      requestFeature(HelperFeatureIter->second.first,                          \
                     HelperFeatureIter->second.second, File);                  \
    }                                                                          \
  }
#define ADD_HELPER_FEATURE_FOR_TYPE_NAMES(TYPE)                                \
  void requestHelperFeatureForTypeNames(const std::string Name, TYPE File) {   \
    auto HelperFeatureIter = MapNames::TypeNamesHelperFeaturesMap.find(Name);  \
    if (HelperFeatureIter != MapNames::TypeNamesHelperFeaturesMap.end()) {     \
      requestFeature(HelperFeatureIter->second.first,                          \
                     HelperFeatureIter->second.second, File);                  \
    }                                                                          \
  }
ADD_HELPER_FEATURE_FOR_ENUM_NAMES(const std::string)
ADD_HELPER_FEATURE_FOR_ENUM_NAMES(SourceLocation)
ADD_HELPER_FEATURE_FOR_ENUM_NAMES(const Stmt *)
ADD_HELPER_FEATURE_FOR_ENUM_NAMES(const Decl *)
ADD_HELPER_FEATURE_FOR_TYPE_NAMES(const std::string)
ADD_HELPER_FEATURE_FOR_TYPE_NAMES(SourceLocation)
ADD_HELPER_FEATURE_FOR_TYPE_NAMES(const Stmt *)
ADD_HELPER_FEATURE_FOR_TYPE_NAMES(const Decl *)
#undef ADD_HELPER_FEATURE_FOR_ENUM_NAMES
#undef ADD_HELPER_FEATURE_FOR_TYPE_NAMES

std::string getCustomMainHelperFileName() {
  return dpct::DpctGlobalInfo::getCustomHelperFileName();
}

bool isOnlyContainDigit(const std::string &Str) {
  for (const auto &C : Str) {
    if (!std::isdigit(C))
      return false;
  }
  return true;
}

/// The \p VersionStr style must be major.minor.patch
bool convertToIntVersion(std::string VersionStr, unsigned int &Result) {
  // get Major version
  size_t FirstDotLoc = VersionStr.find('.');
  if (FirstDotLoc == std::string::npos)
    return false;
  std::string MajorStr = VersionStr.substr(0, FirstDotLoc);
  if (MajorStr.empty() || !isOnlyContainDigit(MajorStr))
    return false;
  int Major = std::stoi(MajorStr);

  // get Minor version
  ++FirstDotLoc;
  size_t SecondDotLoc = VersionStr.find('.', FirstDotLoc);
  if (SecondDotLoc == std::string::npos || FirstDotLoc > VersionStr.size())
    return false;
  std::string MinorStr =
      VersionStr.substr(FirstDotLoc, SecondDotLoc - FirstDotLoc);
  if (MinorStr.empty() || !isOnlyContainDigit(MinorStr))
    return false;
  int Minor = std::stoi(MinorStr);

  // get Patch version
  ++SecondDotLoc;
  if (SecondDotLoc > VersionStr.size())
    return false;
  std::string PatchStr = VersionStr.substr(SecondDotLoc);
  int Patch = 0;
  if (!PatchStr.empty() && isOnlyContainDigit(PatchStr))
    Patch = std::stoi(PatchStr);

  Result = Major * 100 + Minor * 10 + Patch;
  return true;
}

enum class VersionCmpResult {
  VCR_CURRENT_IS_NEWER,
  VCR_CURRENT_IS_OLDER,
  VCR_VERSION_SAME,
  VCR_CMP_FAILED
};

/// The \p VersionInYaml style must be major.minor.patch
/// Return VCR_CMP_FAILED if meets error
/// Return VCR_VERSION_SAME if \p VersionInYaml is same as current version
/// Return VCR_CURRENT_IS_OLDER if \p VersionInYaml is later than current
/// version Return VCR_CURRENT_IS_NEWER if \p VersionInYaml is earlier than
/// current version
VersionCmpResult compareToolVersion(std::string VersionInYaml) {
  unsigned int PreviousVersion;
  if (convertToIntVersion(VersionInYaml, PreviousVersion)) {
    unsigned int CurrentVersion = std::stoi(DPCT_VERSION_MAJOR) * 100 +
                                  std::stoi(DPCT_VERSION_MINOR) * 10 +
                                  std::stoi(DPCT_VERSION_PATCH);
    if (PreviousVersion > CurrentVersion)
      return VersionCmpResult::VCR_CURRENT_IS_OLDER;
    if (PreviousVersion < CurrentVersion)
      return VersionCmpResult::VCR_CURRENT_IS_NEWER;
    else
      return VersionCmpResult::VCR_VERSION_SAME;
  } else {
    return VersionCmpResult::VCR_CMP_FAILED;
  }
}

void emitHelperFeatureChangeWarning(
    VersionCmpResult CompareResult, std::string PreviousMigrationToolVersion,
    const std::map<std::string, std::set<std::string>>
        &APINameCallerSrcFilesMap) {
  if (CompareResult == VersionCmpResult::VCR_CMP_FAILED ||
      CompareResult == VersionCmpResult::VCR_VERSION_SAME) {
    return;
  }

  for (const auto &Item : APINameCallerSrcFilesMap) {
    std::string APIName = Item.first;
    std::string Text = "";
    if (CompareResult == VersionCmpResult::VCR_CURRENT_IS_OLDER) {
      Text = "NOTE: The helper API \"" + APIName +
             "\" used in the previous migration was added in Intel(R) DPC++ "
             "Compatibility Tool " +
             PreviousMigrationToolVersion +
             " and is not available in Intel(R) DPC++ Compatibility Tool " +
             getDpctVersionStr() +
             " used for the current migration. Migrate all files with the same "
             "version of the tool or update migrated files manually.\n";
    } else {
      // CompareResult == VersionCmpResult::VCR_CURRENT_IS_NEWER
      Text = "NOTE: The helper API \"" + APIName +
             "\" used in the previous migration was removed in "
             "Intel(R) DPC++ Compatibility Tool " +
             getDpctVersionStr() +
             ". Migrate all files with the same version of the tool "
             "or update migrated files manually.\n";
    }
    Text =
        Text + "File(s) used \"" + APIName + "\" in the previous migration:\n";
    for (const auto &File : Item.second) {
      Text = Text + File + "\n";
    }
    clang::dpct::PrintMsg(Text);
  }
}

void collectInfo(
    std::pair<std::string, clang::tooling::HelperFuncForYaml> Feature,
    VersionCmpResult CompareResult,
    std::map<std::string, std::set<std::string>> &APINameCallerSrcFilesMap) {
  if (Feature.second.CallerSrcFiles.empty() ||
      Feature.second.CallerSrcFiles[0].empty()) {
    return;
  }

  std::string APIName = Feature.second.APIName;
  if (APIName.empty()) {
    if (CompareResult == VersionCmpResult::VCR_CURRENT_IS_NEWER) {
      // This code path is for case:
      // Previous migration tool is 2021.3 and an API in 2021.3 is removed
      // in current version.

      // Currently, no API removed, so return directly.
      return;
      // In the future, if there is an API removed, below code should be
      // enabled:
      // auto FeatureID = std::make_pair(File, Feature.first);
      // auto Iter = MapNames::RemovedFeatureMap.find(FeatureID);
      // if (Iter == MapNames::RemovedFeatureMap.end()) {
      //   return;
      // } else {
      //   APIName = Iter->second;
      // }
    } else {
      // VCR_CURRENT_IS_OLDER || VCR_VERSION_SAME || VCR_CMP_FAILED
      return;
    }
  }

  auto Iter = APINameCallerSrcFilesMap.find(APIName);
  if (Iter == APINameCallerSrcFilesMap.end()) {
    std::set<std::string> FilesSet;
    FilesSet.insert(Feature.second.CallerSrcFiles.begin(),
                    Feature.second.CallerSrcFiles.end());
    APINameCallerSrcFilesMap.insert(std::make_pair(APIName, FilesSet));
  } else {
    Iter->second.insert(Feature.second.CallerSrcFiles.begin(),
                        Feature.second.CallerSrcFiles.end());
  }
}

void processFeatureMap(
    const std::map<std::string, clang::tooling::HelperFuncForYaml> &FeatureMap,
    HelperFileEnum CurrentFileID, VersionCmpResult CompareResult,
    std::map<std::string, std::set<std::string>> &APINameCallerSrcFilesMap) {
  for (const auto &FeatureFromYaml : FeatureMap) {
    HelperFeatureIDTy FeatureKey(CurrentFileID, FeatureFromYaml.first);
    auto FeatureIter = MapNames::HelperNameContentMap.find(FeatureKey);
    if (FeatureIter != MapNames::HelperNameContentMap.end()) {
      FeatureIter->second.IsCalled =
          FeatureIter->second.IsCalled || FeatureFromYaml.second.IsCalled;
      for (auto &CallerFileName : FeatureFromYaml.second.CallerSrcFiles) {
        FeatureIter->second.CallerSrcFiles.insert(CallerFileName);
      }

      // Process sub-features
      if (!FeatureFromYaml.second.SubFeatureMap.empty()) {
        processFeatureMap(FeatureFromYaml.second.SubFeatureMap, CurrentFileID,
                          CompareResult, APINameCallerSrcFilesMap);
      }
    } else {
      // Feature added/removed, need emit warning
      collectInfo(FeatureFromYaml, CompareResult, APINameCallerSrcFilesMap);
    }
  }
}

// Update MapNames::HelperNameContentMap from TUR
void updateHelperNameContentMap(
    const clang::tooling::TranslationUnitReplacements &TUR) {
  std::map<std::string, std::set<std::string>> APINameCallerSrcFilesMap;
  VersionCmpResult CompareResult = compareToolVersion(TUR.DpctVersion);

  for (const auto &FileFeatureMap : TUR.FeatureMap) {
    auto Iter = MapNames::HelperFileIDMap.find(FileFeatureMap.first);
    if (Iter != MapNames::HelperFileIDMap.end()) {
      auto CurrentFileID = Iter->second;
      processFeatureMap(FileFeatureMap.second, CurrentFileID, CompareResult,
                        APINameCallerSrcFilesMap);
    } else {
      // New helper file added, need emit warning
      for (auto &FeatureFromYaml : FileFeatureMap.second) {
        // Feature added, need emit warning
        collectInfo(FeatureFromYaml, CompareResult, APINameCallerSrcFilesMap);
      }
    }
  }
  emitHelperFeatureChangeWarning(CompareResult, TUR.DpctVersion,
                                 APINameCallerSrcFilesMap);
}

// Update TUR from MapNames::HelperNameContentMap
void updateTUR(clang::tooling::TranslationUnitReplacements &TUR) {
  auto updateAPIName = [](HelperFeatureIDTy Feature,
                          clang::tooling::HelperFuncForYaml &HFFY) {
    // If this feature can be found in the map, then save the API name (from
    // the map) into yaml file; otherwise save the feature name into yaml
    // file
    auto Iter = MapNames::FeatureNameToAPINameMap.find(Feature);
    if (Iter != MapNames::FeatureNameToAPINameMap.end()) {
      HFFY.APIName = Iter->second;
    } else {
      HFFY.APIName = Feature.second;
    }
  };

  for (auto Entry : MapNames::HelperNameContentMap) {
    if (Entry.second.IsCalled) {
      std::string FileName = MapNames::HelperFileNameMap[Entry.first.first];
      if (Entry.second.ParentFeature.first == HelperFileEnum::Unknown &&
          Entry.second.ParentFeature.second.empty()) {
        // This is not a sub-feature
        TUR.FeatureMap[FileName][Entry.first.second].IsCalled =
            Entry.second.IsCalled;
        TUR.FeatureMap[FileName][Entry.first.second].CallerSrcFiles.clear();

        for (auto CallerFileName : Entry.second.CallerSrcFiles) {
          TUR.FeatureMap[FileName][Entry.first.second].CallerSrcFiles.push_back(
              CallerFileName);
        }

        updateAPIName(Entry.first,
                      TUR.FeatureMap[FileName][Entry.first.second]);
      } else {
        // This is a sub-feature
        std::string ParentFeatureName = Entry.second.ParentFeature.second;
        TUR.FeatureMap[FileName][ParentFeatureName]
            .SubFeatureMap[Entry.first.second]
            .IsCalled = Entry.second.IsCalled;
        TUR.FeatureMap[FileName][ParentFeatureName]
            .SubFeatureMap[Entry.first.second]
            .CallerSrcFiles.clear();

        for (auto CallerFileName : Entry.second.CallerSrcFiles) {
          TUR.FeatureMap[FileName][ParentFeatureName]
              .SubFeatureMap[Entry.first.second]
              .CallerSrcFiles.push_back(CallerFileName);
        }
      }
    }
  }
}

void replaceAllOccurredStrsInStr(std::string &StrNeedProcess,
                                 const std::string &Pattern,
                                 const std::string &Repl) {
  if (StrNeedProcess.empty() || Pattern.empty()) {
    return;
  }

  size_t PatternLen = Pattern.size();
  size_t ReplLen = Repl.size();
  size_t Offset = 0;
  Offset = StrNeedProcess.find(Pattern, Offset);

  while (Offset != std::string::npos) {
    StrNeedProcess.replace(Offset, PatternLen, Repl);
    Offset = Offset + ReplLen;
    Offset = StrNeedProcess.find(Pattern, Offset);
  }
}

void replaceEndOfLine(std::string &StrNeedProcess) {
#ifdef _WIN64
  replaceAllOccurredStrsInStr(StrNeedProcess, "\n", "\r\n");
#endif
}

} // namespace dpct
} // namespace clang