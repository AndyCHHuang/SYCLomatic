//===--- AnalysisInfo.cpp -------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2020 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "AnalysisInfo.h"
#include "Debug.h"
#include "ExprAnalysis.h"
#include "Utility.h"

#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"

#define TYPELOC_CAST(Target) static_cast<const Target &>(TL)

namespace clang {
namespace dpct {
std::string DpctGlobalInfo::InRoot = std::string();
// TODO: implement one of this for each source language.
std::string DpctGlobalInfo::CudaPath = std::string();
UsmLevel DpctGlobalInfo::UsmLvl = UsmLevel::none;
format::FormatRange DpctGlobalInfo::FmtRng = format::FormatRange::none;
DPCTFormatStyle DpctGlobalInfo::FmtST = DPCTFormatStyle::llvm;
bool DpctGlobalInfo::EnableCtad = false;
bool DpctGlobalInfo::EnableComments = false;
CompilerInstance *DpctGlobalInfo::CI = nullptr;
ASTContext *DpctGlobalInfo::Context = nullptr;
SourceManager *DpctGlobalInfo::SM = nullptr;
FileManager   *DpctGlobalInfo::FM = nullptr;
bool DpctGlobalInfo::KeepOriginCode = false;
bool DpctGlobalInfo::SyclNamedLambda = false;
std::map<const char *, std::shared_ptr<DpctGlobalInfo::MacroExpansionRecord>>
    DpctGlobalInfo::ExpansionRangeToMacroRecord;
std::map<MacroInfo *, bool> DpctGlobalInfo::MacroDefines;
std::set<std::string> DpctGlobalInfo::IncludingFileSet;
std::set<std::string> DpctGlobalInfo::FileSetInCompiationDB;
const std::string MemVarInfo::ExternVariableName = "dpct_local";
const int TextureObjectInfo::ReplaceTypeLength = strlen("cudaTextureObject_t");
bool DpctGlobalInfo::GuessIndentWidthMatcherFlag = false;
unsigned int DpctGlobalInfo::IndentWidth = 0;
std::unordered_map<std::string, int> DpctGlobalInfo::LocationInitIndexMap;
int DpctGlobalInfo::CurrentMaxIndex = 0;
int DpctGlobalInfo::CurrentIndexInRule = 0;

bool DpctFileInfo::isInRoot() { return DpctGlobalInfo::isInRoot(FilePath); }
// TODO: implement one of this for each source language.
bool DpctFileInfo::isInCudaPath() {
  return DpctGlobalInfo::isInCudaPath(FilePath);
}

void DpctFileInfo::buildLinesInfo() {
  if (FilePath.empty())
    return;
  auto &SM = DpctGlobalInfo::getSourceManager();

  auto FE = SM.getFileManager().getFile(FilePath);
  if (std::error_code ec = FE.getError())
    return;
  auto FID = SM.getOrCreateFileID(FE.get(), SrcMgr::C_User);
  auto Content = SM.getSLocEntry(FID).getFile().getContentCache();
  if (!Content->SourceLineCache)
    SM.getLineNumber(FID, 0);
  auto LineCache = Content->SourceLineCache;
  auto NumLines = Content->NumLines;
  const char *Buffer = nullptr;
  if (!LineCache) {
    return;
  }
  if (DpctGlobalInfo::isKeepOriginCode())
    Buffer = Content->getBuffer(SM.getDiagnostics(), SM.getFileManager())
                 ->getBufferStart();
  for (unsigned L = 1; L < Content->NumLines; ++L)
    Lines.emplace_back(L, LineCache, Buffer);
  Lines.emplace_back(NumLines, LineCache[NumLines - 1], Content->getSize(),
                     Buffer);
}

void DpctFileInfo::buildReplacements() {
  if (!isInRoot())
    return;
  for (auto &Kernel : KernelMap)
    Kernel.second->buildInfo();

  // DPCT need collect the information in curandGenerator_t decl,
  // curandCreateGenerator API call and curandSetPseudoRandomGeneratorSeed API
  // call, then can migrate them to MKL API.
  for (auto &RandomEngine : RandomEngineMap)
    RandomEngine.second->buildInfo();
}

void DpctFileInfo::emplaceReplacements(tooling::Replacements &ReplSet) {
  for (auto &D : FuncMap)
    D.second->emplaceReplacement();
  Repls.emplaceIntoReplSet(ReplSet);
}

void DpctGlobalInfo::insertCudaMalloc(const CallExpr *CE) {
  if (auto MallocVar = CudaMallocInfo::getMallocVar(CE->getArg(0)))
    insertCudaMallocInfo(MallocVar)->setSizeExpr(CE->getArg(1));
}
void DpctGlobalInfo::insertCublasAlloc(const CallExpr *CE) {
  if (auto MallocVar = CudaMallocInfo::getMallocVar(CE->getArg(2)))
    insertCudaMallocInfo(MallocVar)->setSizeExpr(CE->getArg(0), CE->getArg(1));
}
std::shared_ptr<CudaMallocInfo> DpctGlobalInfo::findCudaMalloc(const Expr *E) {
  if (auto Src = CudaMallocInfo::getMallocVar(E))
    return findCudaMallocInfo(Src);
  return std::shared_ptr<CudaMallocInfo>();
}

void DpctGlobalInfo::insertRandomEngine(const Expr *E) {
  if (auto Src = RandomEngineInfo::getHandleVar(E)) {
    insertRandomEngineInfo(Src);
  }
}
std::shared_ptr<RandomEngineInfo>
DpctGlobalInfo::findRandomEngine(const Expr *E) {
  if (auto Src = RandomEngineInfo::getHandleVar(E)) {
    return findRandomEngineInfo(Src);
  }
  return std::shared_ptr<RandomEngineInfo>();
}

void KernelCallExpr::buildExecutionConfig(
    const CUDAKernelCallExpr *KernelCall) {
  auto Config = KernelCall->getConfig();
  bool LocalReversed = false, GroupReversed = false;
  for (unsigned Idx = 0; Idx < 4; ++Idx) {
    KernelConfigAnalysis A(IsInMacroDefine);
    A.analyze(Config->getArg(Idx), Idx < 2);
    ExecutionConfig.Config[Idx] = A.getReplacedString();
    if (Idx == 0) {
      GroupReversed = A.reversed();
      ExecutionConfig.GroupDirectRef = A.isDirectRef();
    } else if (Idx == 1) {
      LocalReversed = A.reversed();
      ExecutionConfig.LocalDirectRef = A.isDirectRef();
    }
  }
  ExecutionConfig.DeclLocalRange =
      !LocalReversed && !ExecutionConfig.LocalDirectRef;
  ExecutionConfig.DeclGroupRange =
      LocalReversed && !GroupReversed && !ExecutionConfig.GroupDirectRef;
  ExecutionConfig.DeclGlobalRange = !LocalReversed && !GroupReversed;
}

void KernelCallExpr::buildKernelInfo(const CUDAKernelCallExpr *KernelCall) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  SourceLocation Begin = KernelCall->getBeginLoc();
  LocInfo.NL = getNL();
  LocInfo.Indent = getIndent(Begin, SM).str();
  LocInfo.LocHash = getHashAsString(Begin.printToString(SM)).substr(0, 6);
  buildExecutionConfig(KernelCall);
  buildNeedBracesInfo(KernelCall);
}
void KernelCallExpr::buildNeedBracesInfo(const CUDAKernelCallExpr *KernelCall) {
  NeedBraces = true;
  auto &Context = dpct::DpctGlobalInfo::getContext();
  // if parenet is CompoundStmt, then find if it has more than 1 children.
  // else if parent is ExprWithCleanups, then do futher check.
  // else it must be case like:  if/for/while(1) kernel-call, pair of
  // braces are needed.
  auto Parents = Context.getParents(*KernelCall);
  while (Parents.size() == 1) {
    if (auto *Parent = Parents[0].get<CompoundStmt>()) {
      NeedBraces = (Parent->size() > 1);
      return;
    } else if (auto *EWC = Parents[0].get<ExprWithCleanups>()) {
      // treat ExprWithCleanups same as CUDAKernelCallExpr when they shows
      // up together
      Parents = Context.getParents(Parents[0]);
    } else {
      return;
    }
  }
}

void KernelCallExpr::addAccessorDecl() {
  auto &VM = getVarMap();
  if (VM.hasExternShared()) {
    addAccessorDecl(VM.getMap(MemVarInfo::Extern).begin()->second);
  }
  addAccessorDecl(MemVarInfo::Local);
  addAccessorDecl(MemVarInfo::Global);
  for (auto &Tex : VM.getTextureMap())
    SubmitStmtsList.TextureList.emplace_back(Tex.second->getAccessorDecl());
  for (auto &Tex : getTextureObjectList()) {
    if (Tex) {
      if (!Tex->getType()) {
        // Type PlaceHolder
        Tex->setType("PlaceHolder/*Fix the type mannually*/", 1);
      }
      SubmitStmtsList.TextureList.emplace_back(Tex->getAccessorDecl());
    }
  }
}

void KernelCallExpr::addAccessorDecl(MemVarInfo::VarScope Scope) {
  for (auto &VI : getVarMap().getMap(Scope)) {
    addAccessorDecl(VI.second);
  }
}

void KernelCallExpr::addAccessorDecl(std::shared_ptr<MemVarInfo> VI) {
  if (VI->isShared()) {
    if (VI->getType()->getDimension() > 1) {
      SubmitStmtsList.RangeList.emplace_back(
          VI->getRangeDecl(ExecutionConfig.ExternMemSize));
    }
  } else if (!VI->isGlobal()) {
    SubmitStmtsList.MemoryList.emplace_back(
        VI->getMemoryDecl(ExecutionConfig.ExternMemSize));
  } else if (getFilePath() != VI->getFilePath()) {
    // Global variable definition and global variable reference are not in the
    // same file, and are not a share varible, insert extern variable
    // declaration.
    SubmitStmtsList.ExternList.emplace_back(VI->getExternGlobalVarDecl());
  }
  VI->appendAccessorOrPointerDecl(ExecutionConfig.ExternMemSize,
                                  SubmitStmtsList.AccessorList,
                                  SubmitStmtsList.PtrList);
}

void KernelCallExpr::buildKernelArgsStmt() {
  size_t ArgCounter = 0;
  for (auto &Arg : getArgsInfo()) {
    // if current arg is the first arg with default value, insert extra args
    // before current arg
    if (ArgCounter == getFuncInfo()->NonDefaultParamNum) {
      KernelArgs += getExtraArguments();
    }
    if(ArgCounter != 0)
      KernelArgs += ", ";

    if (Arg.IsPointer) {
      auto BufferName = Arg.getIdStringWithSuffix("buf");
      // If Arg is used as lvalue after its most recent memory allocation,
      // offsets are necessary; otherwise, offsets are not necessary.
      if (Arg.IsUsedAsLvalueAfterMalloc) {
        OuterStmts.emplace_back(
            buildString("std::pair<dpct::buffer_t, size_t> ", BufferName,
                        " = dpct::get_buffer_and_offset(", Arg.getArgString(),
                        Arg.IsDefinedOnDevice ? ".get_ptr());" : ");"));
        SubmitStmtsList.AccessorList.emplace_back(buildString(
            "auto ", Arg.getIdStringWithSuffix("acc"), " = ", BufferName,
            ".first.get_access<" + MapNames::getClNamespace() +
                "::access::mode::read_write>(cgh);"));
        OuterStmts.emplace_back(buildString("size_t ",
                                            Arg.getIdStringWithSuffix("offset"),
                                            " = ", BufferName, ".second;"));
        KernelStmts.emplace_back(buildString(
            Arg.getTypeString(), Arg.getIdStringWithIndex(), " = (",
            Arg.getTypeString(), ")(&", Arg.getIdStringWithSuffix("acc"),
            "[0] + ", Arg.getIdStringWithSuffix("offset"), ");"));
        KernelArgs += Arg.getIdStringWithIndex();
      } else {
        OuterStmts.emplace_back(buildString(
            "dpct::buffer_t ", BufferName, " = dpct::get_buffer(",
            Arg.getArgString(), Arg.IsDefinedOnDevice ? ".get_ptr());" : ");"));
        SubmitStmtsList.AccessorList.emplace_back(buildString(
            "auto ", Arg.getIdStringWithSuffix("acc"), " = ", BufferName,
            ".get_access<" + MapNames::getClNamespace() +
                "::access::mode::read_write>(cgh);"));
        KernelArgs += buildString("(", Arg.getTypeString(), ")(&",
                                  Arg.getIdStringWithSuffix("acc"), "[0])");
      }
    } else if (Arg.IsRedeclareRequired || IsInMacroDefine) {
      std::string ReDeclStr = buildString("auto ", Arg.getIdStringWithIndex(),
                                          " = ", Arg.getArgString());
      if (!Arg.IsDefinedOnDevice) {
        ReDeclStr = ReDeclStr + ";";
      } else {
        if (Arg.IsKernelParamPtr) {
          ReDeclStr = ReDeclStr + ".get_ptr();";
        } else {
          ReDeclStr = ReDeclStr + "[0];";
        }
      }
      SubmitStmtsList.CommandGroupList.emplace_back(ReDeclStr);
      KernelArgs += Arg.getIdStringWithIndex();
    } else {
      KernelArgs += Arg.getArgString();
    }
    ArgCounter += 1;
  }

  // if all params have no default value, insert extra args in the end of params
  if (ArgCounter == getFuncInfo()->NonDefaultParamNum) {
    KernelArgs = KernelArgs + getExtraArguments();
  }

  if (KernelArgs.empty()) {
    KernelArgs += getExtraArguments();
  }
}

void KernelCallExpr::print(KernelPrinter &Printer) {
  std::unique_ptr<KernelPrinter::Block> Block;
  if (!OuterStmts.empty()) {
    if (NeedBraces)
      Block = std::move(Printer.block(true));
    else
      Block = std::move(Printer.block(false));
    for (auto &S : OuterStmts)
      Printer.line(S);
  }
  printSubmit(Printer);
  Block.reset();
  if (!getEvent().empty() && isSync())
    Printer.line(getEvent(), ".wait();");
}

void KernelCallExpr::printSubmit(KernelPrinter &Printer) {
  Printer.indent();
  if (!getEvent().empty()) {
    Printer << getEvent() << " = ";
  }
  if (ExecutionConfig.Stream == "0") {
    Printer << "dpct::get_default_queue";
    Printer << "().";
  } else {
    if (ExecutionConfig.Stream[0] == '*' || ExecutionConfig.Stream[0] == '&') {
      Printer << "(" << ExecutionConfig.Stream << ")";
    } else {
      Printer << ExecutionConfig.Stream;
    }
    Printer << "->";
  }
  (Printer << "submit(").newLine();
  printSubmitLamda(Printer);
}

void KernelCallExpr::printSubmitLamda(KernelPrinter &Printer) {
  auto Lamda = Printer.block();
  Printer.line("[&](" + MapNames::getClNamespace() + "::handler &cgh) {");
  {
    auto Body = Printer.block();
    SubmitStmtsList.print(Printer);
    printParallelFor(Printer);
  }
  Printer.line("});");
}

void KernelCallExpr::printParallelFor(KernelPrinter &Printer) {
  if (!SubmitStmtsList.NdRangeList.empty() &&
      DpctGlobalInfo::isCommentsEnabled())
    Printer.line("// run the kernel within defined ND range");
  if (DpctGlobalInfo::isSyclNamedLambda()) {
    Printer.line("cgh.parallel_for<dpct_kernel_name<class ", getName(), "_",
                 LocInfo.LocHash,
                 (hasTemplateArgs() ? (", " + getTemplateArguments(true)) : ""),
                 ">>(");
  } else {
    Printer.line("cgh.parallel_for(");
  }
  auto B = Printer.block();
  DpctGlobalInfo::printCtadClass(Printer.indent(),
                                 MapNames::getClNamespace() + "::nd_range", 3)
      << "(";
  static std::string CanIgnoreRangeStr =
      DpctGlobalInfo::getCtadClass(MapNames::getClNamespace() + "::range", 3) +
      "(1, 1, 1)";
  if (ExecutionConfig.DeclGlobalRange) {
    printReverseRange(Printer, "dpct_global_range");
  } else if (ExecutionConfig.GroupSize == CanIgnoreRangeStr) {
    printKernelRange(Printer, ExecutionConfig.LocalSize, "dpct_local_range",
                     ExecutionConfig.DeclLocalRange,
                     ExecutionConfig.LocalDirectRef);
  } else if (ExecutionConfig.LocalSize == CanIgnoreRangeStr) {
    printKernelRange(Printer, ExecutionConfig.GroupSize, "dpct_group_range",
                     ExecutionConfig.DeclGroupRange,
                     ExecutionConfig.GroupDirectRef);
  } else {
    printKernelRange(Printer, ExecutionConfig.GroupSize, "dpct_group_range",
                     ExecutionConfig.DeclGroupRange,
                     ExecutionConfig.GroupDirectRef);
    Printer << " * ";
    printKernelRange(Printer, ExecutionConfig.LocalSize, "dpct_local_range",
                     ExecutionConfig.DeclLocalRange,
                     ExecutionConfig.LocalDirectRef);
  }
  Printer << ", ";
  printKernelRange(Printer, ExecutionConfig.LocalSize, "dpct_local_range",
                   ExecutionConfig.DeclLocalRange,
                   ExecutionConfig.LocalDirectRef);
  (Printer << "), ").newLine();
  Printer.line("[=](" + MapNames::getClNamespace() + "::nd_item<3> ",
               DpctGlobalInfo::getItemName(), ") {");
  printKernel(Printer);
  Printer.line("});");
}

void KernelCallExpr::printKernel(KernelPrinter &Printer) {
  auto B = Printer.block();
  for (auto &S : KernelStmts)
    Printer.line(S);
  Printer.indent() << getName()
                   << (hasTemplateArgs()
                           ? buildString("<", getTemplateArguments(), ">")
                           : "")
                   << "(" << KernelArgs << ");";
  Printer.newLine();
}

std::string KernelCallExpr::getReplacement() {
  addAccessorDecl();
  addStreamDecl();
  buildKernelArgsStmt();
  addNdRangeDecl();

  if (IsInMacroDefine) {
    LocInfo.NL = "\\" + LocInfo.NL;
  }
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  KernelPrinter Printer(LocInfo.NL, LocInfo.Indent, OS);
  print(Printer);
  return Printer.str();
}


CallFunctionExpr::CallFunctionExpr(unsigned Offset, const std::string &FilePathIn,
  const CallExpr *CE)
  : FilePath(FilePathIn), BeginLoc(Offset),
  TextureObjectList(CE->getNumArgs(),
    std::shared_ptr<TextureObjectInfo>()) {
  buildTextureObjectArgsInfo(CE);
}

inline std::string CallFunctionExpr::getExtraArguments() {
  if (!FuncInfo)
    return "";
  return getVarMap().getExtraCallArguments(FuncInfo->NonDefaultParamNum,
                                           FuncInfo->ParamsNum -
                                               FuncInfo->NonDefaultParamNum);
}

void KernelCallExpr::buildInfo() {
  CallFunctionExpr::buildInfo();
  // TODO: Output debug info.
  DpctGlobalInfo::getInstance().addReplacement(std::make_shared<ExtReplacement>(
      getFilePath(), getBegin(), 0, getReplacement(), nullptr));
}

void CallFunctionExpr::buildTemplateArgumentsFromTypeLoc(const TypeLoc &TL) {
  switch (TL.getTypeLocClass()) {
  /// e.g. X<T>;
  case TypeLoc::TemplateSpecialization:
    return buildTemplateArgumentsFromSpecializationType(
        TYPELOC_CAST(TemplateSpecializationTypeLoc));
  /// e.g.: X<T1>::template Y<T2>
  case TypeLoc::DependentTemplateSpecialization:
    return buildTemplateArgumentsFromSpecializationType(
        TYPELOC_CAST(DependentTemplateSpecializationTypeLoc));
  default:
    break;
  }
}

void KernelCallExpr::setIsInMacroDefine(const CUDAKernelCallExpr *KernelCall) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto calleeSpelling = KernelCall->getCallee()->getBeginLoc();
  if (SM.isMacroArgExpansion(calleeSpelling)) {
    calleeSpelling = SM.getImmediateExpansionRange(calleeSpelling).getBegin();
  }
  calleeSpelling = SM.getSpellingLoc(calleeSpelling);
  auto ItMatch = dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
      SM.getCharacterData(calleeSpelling));
  if (ItMatch != dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end()) {
    IsInMacroDefine = true;
  }
}

void CallFunctionExpr::buildCallExprInfo(const CallExpr *CE) {
  HasArgs = CE->getNumArgs();
  auto Callee = CE->getCallee()->IgnoreImplicitAsWritten();

  if (auto CallDecl = CE->getDirectCallee()) {
    Name = getName(CallDecl);
    FuncInfo = DeviceFunctionDecl::LinkRedecls(CallDecl);
    if (auto DRE = dyn_cast<DeclRefExpr>(CE->getCallee()->IgnoreImpCasts()))
      buildTemplateArguments(DRE->template_arguments());
  } else if (auto Unresolved = dyn_cast<UnresolvedLookupExpr>(Callee)) {
    Name = Unresolved->getName().getAsString();
    FuncInfo = DeviceFunctionDecl::LinkUnresolved(Unresolved);
    buildTemplateArguments(Unresolved->template_arguments());
  } else if (auto DependentScope =
                 dyn_cast<CXXDependentScopeMemberExpr>(Callee)) {
    Name = DependentScope->getMember().getAsString();
    buildTemplateArguments(DependentScope->template_arguments());
  } else if (auto DSDRE = dyn_cast<DependentScopeDeclRefExpr>(Callee)) {
    Name = DSDRE->getDeclName().getAsString();
    buildTemplateArgumentsFromTypeLoc(DSDRE->getQualifierLoc().getTypeLoc());
  }


  if (FuncInfo) {
    if (FuncInfo->ParamsNum == 0) {
      ExtraArgLoc =
          DpctGlobalInfo::getSourceManager().getFileOffset(CE->getRParenLoc());
    } else if (FuncInfo->NonDefaultParamNum == 0) {
      // if all params have default value
      ExtraArgLoc = DpctGlobalInfo::getSourceManager().getFileOffset(
          CE->getArg(0)->getBeginLoc());
    } else {
      // if some params have default value, set ExtraArgLoc to the location
      // before the comma
      auto &SM = DpctGlobalInfo::getSourceManager();
      auto TokenLoc = Lexer::getLocForEndOfToken(
          SM.getSpellingLoc(
              CE->getArg(FuncInfo->NonDefaultParamNum - 1)->getEndLoc()),
          0, SM, DpctGlobalInfo::getContext().getLangOpts());
      ExtraArgLoc = DpctGlobalInfo::getSourceManager().getFileOffset(TokenLoc);
    }
  }
}

std::shared_ptr<TextureObjectInfo> CallFunctionExpr::addTextureObjectArg(
    unsigned ArgIdx, const DeclRefExpr *TexRef, bool isKernelCall) {
  if (TextureObjectInfo::isTextureObject(TexRef)) {
    if (isKernelCall) {
      if (auto VD = dyn_cast<VarDecl>(TexRef->getDecl())) {
        return addTextureObjectArgInfo(ArgIdx,
                                       std::make_shared<TextureObjectInfo>(VD));
      }
    } else if (auto PVD = dyn_cast<ParmVarDecl>(TexRef->getDecl())) {
      return addTextureObjectArgInfo(ArgIdx,
                                     std::make_shared<TextureObjectInfo>(PVD));
    }
  }
  return std::shared_ptr<TextureObjectInfo>();
}

void CallFunctionExpr::mergeTextureObjectTypeInfo() {
  for (unsigned Idx = 0; Idx < TextureObjectList.size(); ++Idx) {
    if (auto &Obj = TextureObjectList[Idx]) {
      Obj->setType(FuncInfo->getTextureTypeInfo(Idx));
    }
  }
}

std::string CallFunctionExpr::getName(const NamedDecl *D) {
  if (auto ID = D->getIdentifier())
    return ID->getName().str();
  return "";
}

void CallFunctionExpr::buildInfo() {
  if (!FuncInfo)
    return;

  const std::string &DefFilePath = FuncInfo->getDefinitionFilePath();
  if (!DefFilePath.empty() && DefFilePath != getFilePath()) {
    FuncInfo->setNeedSyclExternMacro();
  }
  FuncInfo->buildInfo();
  VarMap.merge(FuncInfo->getVarMap(), TemplateArgs);
  mergeTextureObjectTypeInfo();
}

void CallFunctionExpr::emplaceReplacement() {
  buildInfo();
  DpctGlobalInfo::getInstance().addReplacement(std::make_shared<ExtReplacement>(
      FilePath, ExtraArgLoc, 0, getExtraArguments(), nullptr));
}

std::string CallFunctionExpr::getTemplateArguments(bool WithScalarWrapped) {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  for (auto &TA : TemplateArgs) {
    if (WithScalarWrapped && !TA.isType())
      appendString(OS, "dpct_kernel_scalar<", TA.getString(), ">, ");
    else
      appendString(OS, TA.getString(), ", ");
  }
  OS.flush();
  return (Result.empty()) ? Result : Result.erase(Result.size() - 2);
}

void DeviceFunctionInfo::merge(std::shared_ptr<DeviceFunctionInfo> Other) {
  if (this == Other.get())
    return;
  VarMap.merge(Other->getVarMap());
  dpct::merge(CallExprMap, Other->CallExprMap);
  mergeTextureTypeList(Other->TextureObjectTypeList);
  IsStatic = Other->IsStatic || IsStatic;
}

void DeviceFunctionInfo::mergeTextureTypeList(
    const std::vector<std::shared_ptr<TextureTypeInfo>> &Other) {
  auto SelfItr = TextureObjectTypeList.begin();
  auto BranchItr = Other.begin();
  while ((SelfItr != TextureObjectTypeList.end()) &&
         (BranchItr != Other.end())) {
    if (!(*SelfItr))
      *SelfItr = *BranchItr;
    ++SelfItr;
    ++BranchItr;
  }
  TextureObjectTypeList.insert(SelfItr, BranchItr, Other.end());
}

void DeviceFunctionInfo::mergeCalledTexObj(
    const std::vector<std::shared_ptr<TextureObjectInfo>> &TexObjList) {
  for (auto &Ty : TexObjList) {
    if (Ty) {
      TextureObjectTypeList[Ty->getParamIdx()] = Ty->getType();
    }
  }
}

void DeviceFunctionInfo::buildInfo() {
  if (isBuilt())
    return;
  setBuilt();
  for (auto &Call : CallExprMap) {
    Call.second->emplaceReplacement();
    VarMap.merge(Call.second->getVarMap());
    mergeCalledTexObj(Call.second->getTextureObjectList());
  }
}

inline void DeviceFunctionDecl::emplaceReplacement() {
  // TODO: Output debug info.
  auto Repl = std::make_shared<ExtReplacement>(
      FilePath, ReplaceOffset, ReplaceLength,
      FuncInfo->getExtraParameters(IsExtraParamWithNL, Indent), nullptr);
  Repl->setNotFormatFlag();
  DpctGlobalInfo::getInstance().addReplacement(Repl);

  if (FuncInfo->IsSyclExternMacroNeeded()) {
    std::string StrRepl = "SYCL_EXTERNAL ";
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(FilePath, Offset, 0, StrRepl,
                                         nullptr));
  }
  for (auto &Obj : TextureObjectList) {
    if (Obj) {
      Obj->setType(FuncInfo->getTextureTypeInfo(Obj->getParamIdx()));
      if (!Obj->getType()) {
        // Type PlaceHolder
        Obj->setType("PlaceHolder/*Fix the type mannually*/", 1);
      }
      Obj->addParamDeclReplacement();
    }
  }
}

void DeviceFunctionDecl::buildReplaceLocInfo(const FunctionDecl *FD) {
  if (FD->isImplicit()) {
    NonDefaultParamNum = FD->getNumParams();
    return;
  }

  auto &SM = DpctGlobalInfo::getSourceManager();
  auto &LO = DpctGlobalInfo::getContext().getLangOpts();

  // Need to get the first decl if there are many decl of the same function
  auto FisrtFD = FD->getFirstDecl();
  SourceLocation NextToken;
  // ItEndParam will be the last parameter which has no DefaultArg.
  // The new parameter will be inserted right after ItEndParam.
  auto ItEndParam = FD->param_end() - 1;
  if (FisrtFD->param_empty()) {
    NextToken = FD->getNameInfo().getEndLoc();
    NextToken = Lexer::getLocForEndOfToken(NextToken, 0, SM, LO);
    NonDefaultParamNum = 0;
  } else {
    while ((*ItEndParam)->hasDefaultArg() && ItEndParam != FisrtFD->param_begin()) {
      ItEndParam = ItEndParam - 1;
    }
    if (ItEndParam == FisrtFD->param_begin() && (*ItEndParam)->hasDefaultArg()) {
      NextToken = (*FD->param_begin())->getBeginLoc();
      NonDefaultParamNum = 0;
    }
    else {
      NextToken = (*ItEndParam)->getEndLoc();
      if (SM.isMacroArgExpansion(NextToken)) {
        NextToken = SM.getSpellingLoc(SM.getImmediateExpansionRange(NextToken).getEnd());
      }
      NextToken = Lexer::getLocForEndOfToken(NextToken, 0, SM, LO);
      NonDefaultParamNum = ItEndParam - FD->param_begin() + 1;
    }
  }

  // The rule of wrapping extra parameters in device function declaration:
  // 1. Origin parameters number < 2
  //    Do not add new line.
  // 2. Origin parameters number >= 2
  //    2.1 The first parameter and the last parameter are not in one line
  //        Add new line, and the extra parameters are aligned with the last
  //        parameter.
  //    2.2 The first parameter and the last parameter are in one line
  //        Add new line, and the extra parameters are aligned with the first
  //        parameter.
  if (NonDefaultParamNum >= 2) {
    IsExtraParamWithNL = true;
    auto BeginParam = *(FD->param_begin());
    SourceLocation BeginParamLoc = BeginParam->getBeginLoc();

    unsigned int NeedRemoveLength = 0;
    calculateRemoveLength<CUDAGlobalAttr>(FD, "__global__", NeedRemoveLength,
                                          BeginParamLoc, SM, LO);
    calculateRemoveLength<CUDADeviceAttr>(FD, "__device__", NeedRemoveLength,
                                          BeginParamLoc, SM, LO);
    calculateRemoveLength<CUDAHostAttr>(FD, "__host__", NeedRemoveLength,
                                        BeginParamLoc, SM, LO);

    auto BeginExpLoc = BeginParamLoc;
    auto EndExpLoc = NextToken;
    if (BeginParamLoc.isMacroID())
      BeginExpLoc = SM.getExpansionLoc(BeginParamLoc);
    if (NextToken.isMacroID())
      EndExpLoc = SM.getExpansionLoc(NextToken);

    auto ItMatch = dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
      SM.getCharacterData(NextToken));

    if (ItMatch !=
      dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end() &&
      ItMatch->second->IsFunctionLike) {
      IsExtraParamWithNL = false;
      Indent = "";
      if (NextToken.isMacroID()) {
        NextToken =
          SM.getSpellingLoc(SM.getImmediateExpansionRange(NextToken).getEnd());
      }
    } else {
      if (BeginParamLoc.isMacroID())
        BeginParamLoc = BeginExpLoc;
      if (NextToken.isMacroID())
        NextToken = EndExpLoc;
      auto EndLocInfo = SM.getDecomposedLoc(NextToken);
      auto Buffer = SM.getBufferData(EndLocInfo.first);
      auto NLOffest = Buffer.find_last_of('\n', EndLocInfo.second);
      bool InValidFlag = false;
      if (isInSameLine(BeginParamLoc, NextToken, SM, InValidFlag) &&
          !InValidFlag) {
        // the first param and the last param are in the same line
        // use the first param begin location as the extra param's indent
        Indent = std::string(
            SM.getDecomposedLoc(BeginParamLoc).second - NLOffest - 1, ' ');
        Indent = Indent.substr(NeedRemoveLength);
      } else {
        // the first param and the last param are not in the same line
        // use the indent of the last param line as the extra param's indent
        Indent = getIndent(NextToken, SM).str();
      }
    }
  }

  // Find the correct ReplaceOffset to insert new parameter
  Token Tok;
  auto Result = Lexer::getRawToken(NextToken, Tok, SM, LO, true);
  while (!Result) {
    static const llvm::StringRef VoidId = "void";
    switch (Tok.getKind()) {
    case tok::r_paren:
      ReplaceOffset = SM.getFileOffset(Tok.getLocation());
      return;
    case tok::comma:
      ReplaceOffset = SM.getFileOffset(Tok.getLocation());
      return;
    case tok::raw_identifier:
      if (Tok.getRawIdentifier() == VoidId) {
        ReplaceOffset = SM.getFileOffset(Tok.getLocation());
        ReplaceLength = Tok.getLength();
        return;
      }
      else {
        ReplaceOffset = SM.getFileOffset(Tok.getLocation());
        return;
      }
    // May fall through
    default:
      Result = Lexer::getRawToken(Tok.getEndLoc(), Tok, SM, LO, true);
    }
  }
}

void DeviceFunctionDecl::setFuncInfo(std::shared_ptr<DeviceFunctionInfo> Info) {
  FuncInfo = Info;
  if (IsDefFilePathNeeded)
    FuncInfo->setDefinitionFilePath(FilePath);
}

void DeviceFunctionDecl::LinkDecl(const FunctionDecl *FD, DeclList &List,
                                  std::shared_ptr<DeviceFunctionInfo> &Info) {
  if (!DpctGlobalInfo::isInRoot(FD->getBeginLoc()))
    return;
  auto D = DpctGlobalInfo::getInstance().insertDeviceFunctionDecl(FD);
  if (Info) {
    if (auto FuncInfo = D->getFuncInfo())
      Info->merge(FuncInfo);
    D->setFuncInfo(Info);
  } else if (auto FuncInfo = D->getFuncInfo())
    Info = FuncInfo;
  else
    List.push_back(D);
}

void DeviceFunctionDecl::LinkRedecls(
    const FunctionDecl *FD, DeclList &List,
    std::shared_ptr<DeviceFunctionInfo> &Info) {
  LinkDeclRange(FD->redecls(), List, Info);
}

void DeviceFunctionDecl::LinkDecl(const FunctionTemplateDecl *FTD,
                                  DeclList &List,
                                  std::shared_ptr<DeviceFunctionInfo> &Info) {
  LinkDecl(FTD->getAsFunction(), List, Info);
  LinkDeclRange(FTD->specializations(), List, Info);
}

void DeviceFunctionDecl::LinkDecl(const NamedDecl *ND, DeclList &List,
                                  std::shared_ptr<DeviceFunctionInfo> &Info) {
  switch (ND->getKind()) {
  case Decl::Function:
    return LinkRedecls(static_cast<const FunctionDecl *>(ND), List, Info);
  case Decl::FunctionTemplate:
    return LinkDecl(static_cast<const FunctionTemplateDecl *>(ND), List, Info);
  case Decl::UsingShadow:
    return LinkDecl(
        static_cast<const UsingShadowDecl *>(ND)->getUnderlyingDecl(), List,
        Info);
    break;
  default:
    llvm::dbgs() << "[DeviceFunctionDecl::LinkDecl] Unexpected decl type: "
                 << ND->getDeclKindName();
    return;
  }
}

std::shared_ptr<MemVarInfo> MemVarInfo::buildMemVarInfo(const VarDecl *Var) {
  if (auto Func = Var->getParentFunctionOrMethod()) {
    auto LocInfo = DpctGlobalInfo::getLocInfo(Var);
    auto VI = std::make_shared<MemVarInfo>(LocInfo.second, LocInfo.first, Var);
    DeviceFunctionDecl::LinkRedecls(dyn_cast<FunctionDecl>(Func))->addVar(VI);
    return VI;
  }

  return DpctGlobalInfo::getInstance().insertMemVarInfo(Var);
}

MemVarInfo::VarAttrKind MemVarInfo::getAddressAttr(const AttrVec &Attrs) {
  for (auto VarAttr : Attrs) {
    auto Kind = VarAttr->getKind();
    if (Kind == attr::CUDAManaged)
      return Managed;
  }
  for (auto VarAttr : Attrs) {
    auto Kind = VarAttr->getKind();
    if (Kind == attr::CUDAConstant)
      return Constant;
    else if (Kind == attr::CUDADevice)
      return Device;
    else if (Kind == attr::CUDAShared)
      return Shared;
  }
  return Host;
}

std::string MemVarInfo::getMemoryType() {
  switch (Attr) {
  case clang::dpct::MemVarInfo::Device: {
    static std::string DeviceMemory = "dpct::device_memory";
    return getMemoryType(DeviceMemory, getType());
  }
  case clang::dpct::MemVarInfo::Constant: {
    static std::string ConstantMemory = "dpct::constant_memory";
    return getMemoryType(ConstantMemory, getType());
  }
  case clang::dpct::MemVarInfo::Shared: {
    static std::string SharedMemory = "dpct::local_memory";
    static std::string ExternSharedMemory = "dpct::extern_local_memory";
    if (isExtern())
      return ExternSharedMemory;
    return getMemoryType(SharedMemory, getType());
  }
  case clang::dpct::MemVarInfo::Managed: {
    static std::string ManagedMemory = "dpct::shared_memory";
    return getMemoryType(ManagedMemory, getType());
  }
  default:
    llvm::dbgs() << "[MemVarInfo::getMemoryType] Unexpected attribute.";
    return "";
  }
}

const std::string &MemVarInfo::getMemoryAttr() {
  switch (Attr) {
  case clang::dpct::MemVarInfo::Device: {
    static std::string DeviceMemory = "dpct::device";
    return DeviceMemory;
  }
  case clang::dpct::MemVarInfo::Constant: {
    static std::string ConstantMemory = "dpct::constant";
    return ConstantMemory;
  }
  case clang::dpct::MemVarInfo::Shared: {
    static std::string SharedMemory = "dpct::local";
    return SharedMemory;
  }
  case clang::dpct::MemVarInfo::Managed: {
    static std::string ManagedMemory = "dpct::shared";
    return ManagedMemory;
  }
  default:
    llvm::dbgs() << "[MemVarInfo::getMemoryAttr] Unexpected attribute.";
    static std::string NullString;
    return NullString;
  }
}

std::string MemVarInfo::getDeclarationReplacement() {
  switch (Scope) {
  case clang::dpct::MemVarInfo::Local:
    return "";
  case clang::dpct::MemVarInfo::Extern:
    return buildString("auto ", getName(), " = (", getType()->getBaseName(),
                       " *)", ExternVariableName, ";");
  case clang::dpct::MemVarInfo::Global: {
    if (isShared())
      return "";
    return getMemoryDecl();
  }
  default:
    llvm::dbgs() << "[MemVarInfo::getMemoryType] Unexpected scope.";
    return "";
  }
}

std::string MemVarMap::getExtraCallArguments(bool HasPreParam, bool HasPostParam) const {
  return getArgumentsOrParameters<CallArgument>(HasPreParam, HasPostParam);
}
std::string MemVarMap::getExtraDeclParam(bool HasPreParam, bool HasPostParam,
                                         bool IsExtraParamWithNL,
                                         std::string Indent) const {
  return getArgumentsOrParameters<DeclParameter>(HasPreParam, HasPostParam, IsExtraParamWithNL,
                                                 Indent);
}
std::string MemVarMap::getKernelArguments(bool HasPreParam, bool HasPostParam) const {
  return getArgumentsOrParameters<KernelArgument>(HasPreParam, HasPostParam);
}

CtTypeInfo::CtTypeInfo(const TypeLoc &TL, bool NeedSizeFold) : CtTypeInfo() {
  setTypeInfo(TL, NeedSizeFold);
}

std::string CtTypeInfo::getRangeArgument(const std::string &MemSize,
                                         bool MustArguments) {
  std::string Arg = "(";
  for (auto &R : Range) {
    auto Size = R.getSize();
    if (Size.empty()) {
      if (MemSize.empty()) {
        Arg += "1";
      } else {
        Arg += MemSize;
      }
    } else
      Arg += Size;
    Arg += ", ";
  }
  return (Arg.size() == 1) ? (MustArguments ? (Arg + ")") : "")
                           : Arg.replace(Arg.size() - 2, 2, ")");
}

void CtTypeInfo::setTypeInfo(const TypeLoc &TL, bool NeedSizeFold) {
  switch (TL.getTypeLocClass()) {
  case TypeLoc::Qualified:
    BaseName = TL.getType().getLocalQualifiers().getAsString(
        DpctGlobalInfo::getContext().getPrintingPolicy());
    return setTypeInfo(TYPELOC_CAST(QualifiedTypeLoc).getUnqualifiedLoc(),
                       NeedSizeFold);
  case TypeLoc::ConstantArray:
    return setArrayInfo(TYPELOC_CAST(ConstantArrayTypeLoc), NeedSizeFold);
  case TypeLoc::DependentSizedArray:
    return setArrayInfo(TYPELOC_CAST(DependentSizedArrayTypeLoc), NeedSizeFold);
  case TypeLoc::IncompleteArray:
    return setArrayInfo(TYPELOC_CAST(IncompleteArrayTypeLoc), NeedSizeFold);
  case TypeLoc::Pointer:
    IsPointer = true;
    return setTypeInfo(TYPELOC_CAST(PointerTypeLoc).getPointeeLoc());
  case TypeLoc::LValueReference:
  case TypeLoc::RValueReference:
    IsReference = true;
    return setTypeInfo(TYPELOC_CAST(ReferenceTypeLoc).getPointeeLoc());
  case TypeLoc::Elaborated:
    return setTypeInfo(TYPELOC_CAST(ElaboratedTypeLoc).getNamedTypeLoc());
  case TypeLoc::TemplateTypeParm:
  case TypeLoc::TemplateSpecialization:
  case TypeLoc::DependentTemplateSpecialization:
    setTemplateInfo(TL);
  default:
    setName(TL.getType());
  }
}

void CtTypeInfo::setTemplateInfo(const TypeLoc &TL) {
  IsTemplate = true;
  ExprAnalysis EA;
  EA.analyze(TL);
  TDSI = EA.getTemplateDependentStringInfo();
}

void CtTypeInfo::setArrayInfo(const IncompleteArrayTypeLoc &TL,
                              bool NeedSizeFold) {
  Range.emplace_back();
  setTypeInfo(TL.getElementLoc(), NeedSizeFold);
}

void CtTypeInfo::setArrayInfo(const DependentSizedArrayTypeLoc &TL,
                              bool NeedSizeFold) {
  ExprAnalysis EA;
  EA.analyze(TL.getSizeExpr());
  Range.emplace_back(EA.getTemplateDependentStringInfo());
  setTypeInfo(TL.getElementLoc(), NeedSizeFold);
}

void CtTypeInfo::setArrayInfo(const ConstantArrayTypeLoc &TL,
                              bool NeedSizeFold) {
  if (NeedSizeFold) {
    Range.emplace_back(getFoldedArraySize(TL));
  } else {
    Range.emplace_back(getUnfoldedArraySize(TL));
  }
  setTypeInfo(TL.getElementLoc(), NeedSizeFold);
}

std::string CtTypeInfo::getUnfoldedArraySize(const ConstantArrayTypeLoc &TL) {
  ExprAnalysis A;
  A.analyze(TL.getSizeExpr());
  return A.getReplacedString();
}

void CtTypeInfo::setName(QualType Ty) {
  auto &PP = DpctGlobalInfo::getContext().getPrintingPolicy();
  BaseNameWithoutQualifiers = Ty.getUnqualifiedType().getAsString(PP);

  OrginalBaseType = BaseNameWithoutQualifiers;
  if (!isTemplate())
    MapNames::replaceName(MapNames::TypeNamesMap, BaseNameWithoutQualifiers);
  auto Q = Ty.getLocalQualifiers();
  if (BaseName.empty())
    BaseName = BaseName = BaseNameWithoutQualifiers;
  else
    BaseName = buildString(BaseName, " ", BaseNameWithoutQualifiers);
}

std::shared_ptr<CtTypeInfo> CtTypeInfo::applyTemplateArguments(
    const std::vector<TemplateArgumentInfo> &TA) {
  auto NewType = std::make_shared<CtTypeInfo>(*this);
  if (TDSI)
    NewType->TDSI = TDSI->applyTemplateArguments(TA);
  for (auto &R : NewType->Range)
    R.setTemplateList(TA);
  return NewType;
}

void SizeInfo::setTemplateList(
    const std::vector<TemplateArgumentInfo> &TemplateList) {
  if (TDSI)
    TDSI = TDSI->applyTemplateArguments(TemplateList);
}

void RandomEngineInfo::buildInfo() {
  if (!NeedPrint)
    return;

  // insert engine arguments
  if (IsClassMember) {
    // replace type
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(DeclFilePath, TypeBeginOffest,
                                         TypeLength, TypeReplacement + "*",
                                         nullptr));
    if (IsQuasiEngine) {
      DpctGlobalInfo::getInstance().addReplacement(
          std::make_shared<ExtReplacement>(
              CreateCallFilePath, CreateAPIBegin, CreateAPILength,
              DeclaratorDeclName + " = new " + TypeReplacement +
                  "(dpct::get_default_queue(), " + DimExpr + ")",
              nullptr));
    } else {
      DpctGlobalInfo::getInstance().addReplacement(
          std::make_shared<ExtReplacement>(
              CreateCallFilePath, CreateAPIBegin, CreateAPILength,
              DeclaratorDeclName + " = new " + TypeReplacement +
                  "(dpct::get_default_queue(), " + SeedExpr + ")",
              nullptr));
    }
  } else {
    // replace type
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(DeclFilePath, TypeBeginOffest,
                                         TypeLength, TypeReplacement, nullptr));
    if (IsQuasiEngine) {
      DpctGlobalInfo::getInstance().addReplacement(
          std::make_shared<ExtReplacement>(
              DeclFilePath, IdentifierEndOffest, 0,
              "(dpct::get_default_queue(), " + DimExpr + ")", nullptr));
    } else {
      DpctGlobalInfo::getInstance().addReplacement(
          std::make_shared<ExtReplacement>(
              DeclFilePath, IdentifierEndOffest, 0,
              "(dpct::get_default_queue(), " + SeedExpr + ")", nullptr));
    }
  }
}

} // namespace dpct
} // namespace clang
