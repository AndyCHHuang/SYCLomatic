//===--- AnalysisInfo.cpp -------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2021 Intel Corporation. All rights reserved.
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
#include "Diagnostics.h"

#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Tooling/Tooling.h"
#include <deque>

#define TYPELOC_CAST(Target) static_cast<const Target &>(TL)

namespace clang {
extern std::function<bool(SourceLocation)> IsInRootFunc;
extern std::function<unsigned int()> GetRunRound;
namespace dpct {
std::string DpctGlobalInfo::InRoot = std::string();
std::string DpctGlobalInfo::OutRoot = std::string();
// TODO: implement one of this for each source language.
std::string DpctGlobalInfo::CudaPath = std::string();
UsmLevel DpctGlobalInfo::UsmLvl = UsmLevel::none;
unsigned int DpctGlobalInfo::AssumedNDRangeDim = 3;
std::unordered_set<std::string> DpctGlobalInfo::PrecAndDomPairSet;
std::unordered_set<FFTTypeEnum> DpctGlobalInfo::FFTTypeSet;
std::unordered_set<int> DpctGlobalInfo::DeviceRNGReturnNumSet;
std::unordered_set<std::string> DpctGlobalInfo::HostRNGEngineTypeSet;
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
std::map<std::string, std::shared_ptr<DpctGlobalInfo::MacroExpansionRecord>>
    DpctGlobalInfo::ExpansionRangeToMacroRecord;
std::tuple<unsigned int, std::string, SourceRange> DpctGlobalInfo::LastMacroRecord =
  std::make_tuple<unsigned int, std::string, SourceRange>(0, "", SourceRange());
std::map<std::string, SourceLocation> DpctGlobalInfo::EndifLocationOfIfdef;
std::vector<std::pair<std::string, size_t>> DpctGlobalInfo::ConditionalCompilationLoc;
std::map<std::string, std::shared_ptr<DpctGlobalInfo::MacroDefRecord>>
    DpctGlobalInfo::MacroTokenToMacroDefineLoc;
std::map<std::string, std::string> DpctGlobalInfo::FunctionCallInMacroMigrateRecord;
std::map<std::string, SourceLocation> DpctGlobalInfo::EndOfEmptyMacros;
std::map<std::string, SourceLocation> DpctGlobalInfo::BeginOfEmptyMacros;
std::map<std::string, bool> DpctGlobalInfo::MacroDefines;
std::set<std::string> DpctGlobalInfo::IncludingFileSet;
std::set<std::string> DpctGlobalInfo::FileSetInCompiationDB;
std::unordered_map<std::string, std::vector<clang::tooling::Replacement>>
    DpctGlobalInfo::FileRelpsMap;
std::unordered_map<std::string, std::string> DpctGlobalInfo::DigestMap;
const std::string DpctGlobalInfo::YamlFileName = "MainSourceFiles.yaml";
std::set<std::string> DpctGlobalInfo::GlobalVarNameSet;
const std::string MemVarInfo::ExternVariableName = "dpct_local";
std::unordered_map<const DeclStmt *, int> MemVarInfo::AnonymousTypeDeclStmtMap;
const int TextureObjectInfo::ReplaceTypeLength = strlen("cudaTextureObject_t");
bool DpctGlobalInfo::GuessIndentWidthMatcherFlag = false;
unsigned int DpctGlobalInfo::IndentWidth = 0;
std::map<unsigned int, unsigned int> DpctGlobalInfo::KCIndentWidthMap;
std::unordered_map<std::string, int> DpctGlobalInfo::LocationInitIndexMap;
int DpctGlobalInfo::CurrentMaxIndex = 0;
int DpctGlobalInfo::CurrentIndexInRule = 0;
clang::format::FormatStyle DpctGlobalInfo::CodeFormatStyle;
bool DpctGlobalInfo::HasFoundDeviceChanged = false;
std::unordered_map<int, DpctGlobalInfo::HelperFuncReplInfo>
    DpctGlobalInfo::HelperFuncReplInfoMap;
int DpctGlobalInfo::HelperFuncReplInfoIndex = 1;
std::unordered_map<std::string, DpctGlobalInfo::TempVariableDeclCounter>
    DpctGlobalInfo::TempVariableDeclCounterMap;
std::unordered_set<std::string> DpctGlobalInfo::TempVariableHandledSet;
bool DpctGlobalInfo::UsingDRYPattern = true;
bool DpctGlobalInfo::SpBLASUnsupportedMatrixTypeFlag = false;
std::unordered_map<std::string, FFTExecAPIInfo> DpctGlobalInfo::FFTExecAPIInfoMap;
std::unordered_map<std::string, FFTHandleInfo>
    DpctGlobalInfo::FFTHandleInfoMap;
unsigned int DpctGlobalInfo::CudaBuiltinXDFIIndex = 1;
std::unordered_map<unsigned int, std::shared_ptr<DeviceFunctionInfo>>
    DpctGlobalInfo::CudaBuiltinXDFIMap;
bool DpctGlobalInfo::HasFFTSetStream = false;
unsigned int DpctGlobalInfo::RunRound = 0;
bool DpctGlobalInfo::NeedRunAgain = false;
std::unordered_map<std::string, std::shared_ptr<DeviceFunctionInfo>>
    DeviceFunctionDecl::FuncInfoMap;
CudaArchPPMap DpctGlobalInfo::CAPPInfoMap;
HDCallMap DpctGlobalInfo::HostDeviceFCallIMap;
HDDefMap DpctGlobalInfo::HostDeviceFDefIMap;
HDDeclMap DpctGlobalInfo::HostDeviceFDeclIMap;
std::unordered_map<std::string, std::shared_ptr<ExtReplacements>>
    DpctGlobalInfo::FileReplCache;
std::set<std::string> DpctGlobalInfo::ReProcessFile;
std::set<std::string> DpctGlobalInfo::ProcessedFile;

void DpctGlobalInfo::resetInfo() {
  FileMap.clear();
  PrecAndDomPairSet.clear();
  FFTTypeSet.clear();
  DeviceRNGReturnNumSet.clear();
  HostRNGEngineTypeSet.clear();
  KCIndentWidthMap.clear();
  LocationInitIndexMap.clear();
  ExpansionRangeToMacroRecord.clear();
  EndifLocationOfIfdef.clear();
  ConditionalCompilationLoc.clear();
  MacroTokenToMacroDefineLoc.clear();
  FunctionCallInMacroMigrateRecord.clear();
  EndOfEmptyMacros.clear();
  BeginOfEmptyMacros.clear();
  FileRelpsMap.clear();
  DigestMap.clear();
  MacroDefines.clear();
  CurrentMaxIndex = 0;
  CurrentIndexInRule = 0;
  IncludingFileSet.clear();
  FileSetInCompiationDB.clear();
  GlobalVarNameSet.clear();
  HasFoundDeviceChanged = false;
  HelperFuncReplInfoMap.clear();
  HelperFuncReplInfoIndex = 1;
  TempVariableDeclCounterMap.clear();
  TempVariableHandledSet.clear();
  UsingDRYPattern = true;
  SpBLASUnsupportedMatrixTypeFlag = false;
  FFTExecAPIInfoMap.clear();
  FFTHandleInfoMap.clear();
  HasFFTSetStream = false;
  NeedRunAgain = false;
}

DpctGlobalInfo::DpctGlobalInfo() {
  IsInRootFunc = DpctGlobalInfo::checkInRoot;
  GetRunRound = DpctGlobalInfo::getRunRound;
  tooling::SetGetRunRound(DpctGlobalInfo::getRunRound);
  tooling::SetReProcessFile(DpctGlobalInfo::ReProcessFile);
  tooling::SetProcessedFile(DpctGlobalInfo::ProcessedFile);
}

std::shared_ptr<KernelCallExpr>
DpctGlobalInfo::buildLaunchKernelInfo(const CallExpr *LaunchKernelCall) {
  auto LocInfo = getLocInfo(LaunchKernelCall->getBeginLoc());
  auto FileInfo = insertFile(LocInfo.first);
  if (FileInfo->findNode<KernelCallExpr>(LocInfo.second))
    return std::shared_ptr<KernelCallExpr>();

  auto KernelInfo =
      KernelCallExpr::buildFromCudaLaunchKernel(LocInfo, LaunchKernelCall);
  if (KernelInfo) {
    FileInfo->insertNode(LocInfo.second, KernelInfo);
  }

  return KernelInfo;
}

void DpctGlobalInfo::insertFFTPlanAPIInfo(SourceLocation SL,
                                          FFTPlanAPIInfo Info) {
  auto LocInfo = getLocInfo(SL);
  auto FileInfo = insertFile(LocInfo.first);
  auto &M = FileInfo->getFFTPlanAPIInfoMap();
  if (M.find(LocInfo.second) == M.end()) {
    Info.FilePath = LocInfo.first;
    M.insert(std::make_pair(LocInfo.second, Info));
  }
}

void DpctGlobalInfo::insertFFTExecAPIInfo(SourceLocation SL,
                                          FFTExecAPIInfo Info) {
  auto LocInfo = getLocInfo(SL);
  auto FileInfo = insertFile(LocInfo.first);
  auto &M = FileInfo->getFFTExecAPIInfoMap();
  if (M.find(LocInfo.second) == M.end()) {
    Info.FilePath = LocInfo.first;
    M.insert(std::make_pair(LocInfo.second, Info));
  }
}

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
  auto &Content = SM.getSLocEntry(FID).getFile().getContentCache();
  if (!Content.SourceLineCache) {
    bool Invalid;
    SM.getLineNumber(FID, 0, &Invalid);
    if (Invalid)
      return;
  }
  auto RawBuffer =
      Content.getBufferOrNone(SM.getDiagnostics(), SM.getFileManager())
          .getValueOr(llvm::MemoryBufferRef())
          .getBuffer();
  if (RawBuffer.empty())
    return;
  FileContentCache = RawBuffer.str();
  FileSize = RawBuffer.size();
  auto LineCache = Content.SourceLineCache.getLines();
  auto NumLines = Content.SourceLineCache.size();
  StringRef CacheBuffer(FileContentCache);
  for (unsigned L = 1; L < NumLines; ++L)
    Lines.emplace_back(L, LineCache, CacheBuffer);
  Lines.emplace_back(NumLines, LineCache[NumLines - 1], FileSize, CacheBuffer);
}

void DpctFileInfo::buildUnionFindSet() {
  for (auto &Kernel : KernelMap)
    Kernel.second->buildUnionFindSet();
}

void DpctFileInfo::setDim() {
  for (auto &Kernel : KernelMap)
    Kernel.second->setDim();
}

void DpctFileInfo::buildKernelInfo() {
  for (auto &Kernel : KernelMap)
    Kernel.second->buildInfo();
}
void DpctFileInfo::postProcess(){
  if(!isInRoot())
    return;
  for (auto &D : FuncMap)
    D.second->emplaceReplacement();
  if(!Repls->empty()){
    Repls->postProcess();
    if(DpctGlobalInfo::getRunRound() == 0){
      DpctGlobalInfo::getInstance().cacheFileRepl(FilePath, Repls);
    }
  }
}

void DpctFileInfo::buildReplacements() {
  if (!isInRoot())
    return;

  if(FilePath.empty())
    return;
  // Traver all the global variables stored one by one to check if its name is
  // same with normal global variable's name in host side, if the one is found,
  // postfix "_ct" is added to this __constant__ symbol's name.
  std::unordered_map<unsigned int, std::string> ReplUpdated;
  for (auto Entry : MemVarMap) {
    if (Entry.second->isIgnore())
      continue;

    auto Name = Entry.second->getName();
    auto &GlobalVarNameSet = dpct::DpctGlobalInfo::getGlobalVarNameSet();
    if (GlobalVarNameSet.find(Name) != end(GlobalVarNameSet)) {
      Entry.second->setName(Name + "_ct");
    }

    std::string Repl = Entry.second->getDeclarationReplacement();
    auto FilePath = Entry.second->getFilePath();
    auto Offset = Entry.second->getNewConstVarOffset();
    auto Length = Entry.second->getNewConstVarLength();

    auto &ReplText = ReplUpdated[Offset];
    if (!ReplText.empty()) {
      ReplText += getNL() + Repl;
    } else {
      ReplText = Repl;
    }

    auto R = std::make_shared<ExtReplacement>(FilePath, Offset, Length,
                                              ReplText, nullptr);

    addReplacement(R);
  }

  for (auto &Kernel : KernelMap)
    Kernel.second->addReplacements();

  for (auto &BuiltinVar : BuiltinVarInfoMap) {
    auto Ptr = MemVarMap::getHeadWithoutPathCompression(
        &(BuiltinVar.second.DFI->getVarMap()));
    if (Ptr) {
      unsigned int ID = (Ptr->Dim == 1) ? 0 : 2;
      BuiltinVar.second.buildInfo(FilePath, BuiltinVar.first, ID);
    }
  }

  // Below four maps are used for device RNG API migration
  for (auto &StateTypeInfo : DeviceRandomStateTypeMap)
    StateTypeInfo.second.buildInfo(FilePath, StateTypeInfo.first);

  for (auto &InitAPIInfo : DeviceRandomInitAPIMap)
    InitAPIInfo.second.buildInfo(FilePath, InitAPIInfo.first);

  buildDeviceDistrDeclInfo();
  for (auto &Info : DeviceRandomGenerateAPIMap)
    Info.second.buildInfo(FilePath, Info.first);
  for (auto &Info : DeviceRandomDistrDeclMap)
    Info.second.buildInfo(FilePath, Info.first);

  // DPCT need collect the information in curandGenerator_t decl,
  // curandCreateGenerator API call and curandSetPseudoRandomGeneratorSeed API
  // call, then can migrate them to MKL API.
  for (auto &RandomEngine : RandomEngineMap) {
    RandomEngine.second->updateEngineType();
    RandomEngine.second->buildInfo();
  }

  for (auto &EngineType : HostRandomEngineTypeMap)
    EngineType.second.buildInfo(FilePath, EngineType.first);

  for (auto &DistrInfo : HostRandomDistrMap) {
    DistrInfo.second.buildInfo(
        FilePath, std::get<0>(DistrInfo.first), std::get<1>(DistrInfo.first),
        std::get<2>(DistrInfo.first), std::get<3>(DistrInfo.first));
  }

  if (DpctGlobalInfo::getSpBLASUnsupportedMatrixTypeFlag()) {
    for (auto &SpBLASWarningLocOffset : SpBLASSet) {
      DiagnosticsUtils::report(getFilePath(), SpBLASWarningLocOffset,
                               Diagnostics::UNSUPPORT_MATRIX_TYPE, true, false);
    }
  }

  for (auto &AtomicInfo : AtomicMap) {
    if (std::get<2>(AtomicInfo.second))
      DiagnosticsUtils::report(getFilePath(), std::get<0>(AtomicInfo.second),
                               Diagnostics::API_NOT_OCCURRED_IN_AST, true,
                               false, std::get<1>(AtomicInfo.second));
  }

  for (auto &DescInfo : FFTDescriptorTypeMap) {
    DescInfo.second.buildInfo(FilePath, DescInfo.first);
  }

  for (auto &DescInfo : EventSyncTypeMap) {
    DescInfo.second.buildInfo(FilePath, DescInfo.first);
  }

  for (auto &PlanInfo : FFTPlanAPIInfoMap) {
    PlanInfo.second.buildInfo();
  }

  for (auto &ExecInfo : FFTExecAPIInfoMap) {
    ExecInfo.second.buildInfo();
  }

  const auto &EventMallocFreeMap = getEventMallocFreeMap();
  for (const auto &Entry : EventMallocFreeMap) {
      auto &Pair = Entry.second;
      for (auto &R0 : Pair.first) {
        addReplacement(R0);
      }
      for (auto &R1 : Pair.second) {
        addReplacement(R1);
      }
    }
}

void DpctFileInfo::emplaceReplacements(ReplTy &ReplSet) {
  if(!Repls->empty())
    Repls->emplaceIntoReplSet(ReplSet[FilePath]);
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
  if (auto Src = getHandleVar(E)) {
    insertRandomEngineInfo(Src);
  }
}
std::shared_ptr<RandomEngineInfo>
DpctGlobalInfo::findRandomEngine(const Expr *E) {
  if (auto Src = getHandleVar(E)) {
    return findRandomEngineInfo(Src);
  }
  return std::shared_ptr<RandomEngineInfo>();
}

void DpctGlobalInfo::insertBuiltinVarInfo(
    SourceLocation SL, unsigned int Len, std::string Repl,
    std::shared_ptr<DeviceFunctionInfo> DFI) {
  auto LocInfo = getLocInfo(SL);
  auto FileInfo = insertFile(LocInfo.first);
  auto &M = FileInfo->getBuiltinVarInfoMap();
  auto Iter = M.find(LocInfo.second);
  if (Iter == M.end()) {
    BuiltinVarInfo BVI(Len, Repl, DFI);
    M.insert(std::make_pair(LocInfo.second, BVI));
  }
}

int KernelCallExpr::calculateOriginArgsSize() const {
  int Size = 0;
  for (auto &ArgInfo : ArgsInfo) {
    Size += ArgInfo.ArgSize;
  }
  return Size;
}



template <class ArgsRange>
void KernelCallExpr::buildExecutionConfig(const ArgsRange &ConfigArgs) {
  int Idx = 0;
  for (auto Arg : ConfigArgs) {
    KernelConfigAnalysis A(IsInMacroDefine);
    A.analyze(Arg, Idx, Idx < 2);
    ExecutionConfig.Config[Idx] = A.getReplacedString();

    if (Idx == 0) {
      ExecutionConfig.GroupDirectRef = A.isDirectRef();
    } else if (Idx == 1) {
      ExecutionConfig.LocalDirectRef = A.isDirectRef();

      // Using another analysis because previous analysis may return directly
      // when in macro is true.
      // Here set the argument of KFA as false, so it will not return directly.
      KernelConfigAnalysis KFA(false);
      KFA.analyze(Arg, 1, true);
      if (KFA.isNeedEmitWGSizeWarning())
        DiagnosticsUtils::report(getFilePath(), getBegin(),
                                 Diagnostics::EXCEED_MAX_WORKGROUP_SIZE, true, false);
    }
    ++Idx;
  }

  if (DpctGlobalInfo::getAssumedNDRangeDim() == 1) {
    Idx = 0;
    for (auto Arg : ConfigArgs) {
      if (Idx > 1)
        break;
      KernelConfigAnalysis AnalysisTry1D(IsInMacroDefine);
      AnalysisTry1D.IsTryToUseOneDimension = true;
      AnalysisTry1D.analyze(Arg, Idx, Idx < 2);
      if (Idx == 0) {
        GridDim = AnalysisTry1D.Dim;
        ExecutionConfig.GroupSizeFor1D = AnalysisTry1D.getReplacedString();
      } else if (Idx == 1) {
        BlockDim = AnalysisTry1D.Dim;
        ExecutionConfig.LocalSizeFor1D = AnalysisTry1D.getReplacedString();
      }
      ++Idx;
    }
  }


  if (ExecutionConfig.Stream == "0") {
    int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
    ExecutionConfig.Stream = "{{NEEDREPLACEQ" + std::to_string(Index) + "}}";
    buildTempVariableMap(Index, *ConfigArgs.begin(),
                         HelperFuncType::DefaultQueue);
  }
}

void KernelCallExpr::buildKernelInfo(const CUDAKernelCallExpr *KernelCall) {
  buildLocationInfo(KernelCall);
  buildExecutionConfig(KernelCall->getConfig()->arguments());
  buildNeedBracesInfo(KernelCall);
}

void KernelCallExpr::buildLocationInfo(const CallExpr *KernelCall) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  SourceLocation Begin = KernelCall->getBeginLoc();
  LocInfo.NL = getNL();
  LocInfo.Indent = getIndent(Begin, SM).str();
  LocInfo.LocHash = getHashAsString(Begin.printToString(SM)).substr(0, 6);
}

void KernelCallExpr::buildNeedBracesInfo(const CallExpr *KernelCall) {
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
    } else if (Parents[0].get<ExprWithCleanups>()) {
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
  for (auto &Tex : VM.getTextureMap()) {
    SubmitStmtsList.TextureList.emplace_back(Tex.second->getAccessorDecl());
    SubmitStmtsList.SamplerList.emplace_back(Tex.second->getSamplerDecl());
  }
  for (auto &Tex : getTextureObjectList()) {
    if (Tex) {
      if (!Tex->getType()) {
        // Type dpct_placeholder
        Tex->setType("dpct_placeholder/*Fix the type manually*/", 1);
        DiagnosticsUtils::report(getFilePath(), getBegin(),
                                 Diagnostics::UNDEDUCED_TYPE, true, false,
                                 "image_accessor_ext");
      }
      SubmitStmtsList.TextureList.emplace_back(Tex->getAccessorDecl());
      SubmitStmtsList.SamplerList.emplace_back(Tex->getSamplerDecl());
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
  } else {
    SubmitStmtsList.InitList.emplace_back(
        VI->getInitStmt(isDefaultStream() ? "" : ExecutionConfig.Stream));
    if (VI->isLocal()) {
      SubmitStmtsList.MemoryList.emplace_back(
          VI->getMemoryDecl(ExecutionConfig.ExternMemSize));
    } else if (getFilePath() != VI->getFilePath() &&
               !isIncludedFile(getFilePath(), VI->getFilePath())) {
      // Global variable definition and global variable reference are not in the
      // same file, and are not a share varible, insert extern variable
      // declaration.
      SubmitStmtsList.ExternList.emplace_back(VI->getExternGlobalVarDecl());
    }
  }
  VI->appendAccessorOrPointerDecl(ExecutionConfig.ExternMemSize,
                                  SubmitStmtsList.AccessorList,
                                  SubmitStmtsList.PtrList);
  if (VI->isTypeDeclaredLocal()) {
    if (DiagnosticsUtils::report(getFilePath(), getBegin(),
                                 Diagnostics::TYPE_IN_FUNCTION, false, false,
                                 VI->getName(), VI->getLocalTypeName())) {
      if (!SubmitStmtsList.AccessorList.empty()) {
        SubmitStmtsList.AccessorList.back().Warning =
            DiagnosticsUtils::getWarningTextAndUpdateUniqueID(
                Diagnostics::TYPE_IN_FUNCTION, VI->getName(),
                VI->getLocalTypeName());
      }
    }
  }
}

void KernelCallExpr::buildKernelArgsStmt() {
  size_t ArgCounter = 0;
  for (auto &Arg : getArgsInfo()) {
    // if current arg is the first arg with default value, insert extra args
    // before current arg
    if (getFuncInfo()) {
      if (ArgCounter == getFuncInfo()->NonDefaultParamNum) {
        KernelArgs += getExtraArguments();
      }
    }
    if(ArgCounter != 0)
      KernelArgs += ", ";

    if (Arg.IsDoublePointer) {
      DiagnosticsUtils::report(getFilePath(), getBegin(),
                               Diagnostics::VIRTUAL_POINTER, true, false,
                               Arg.getArgString());
    }

    if (Arg.TryGetBuffer) {
      auto BufferName = Arg.getIdStringWithSuffix("buf");
      // If Arg is used as lvalue after its most recent memory allocation,
      // offsets are necessary; otherwise, offsets are not necessary.

      // If we found this is a RNG state type, we add the vec_size here.
      std::string TypeStr = Arg.getTypeString();
      if (Arg.IsDeviceRandomGeneratorType) {
        if (DpctGlobalInfo::getDeviceRNGReturnNumSet().size() == 1) {
          TypeStr = TypeStr + "<" +
                    std::to_string(
                        *DpctGlobalInfo::getDeviceRNGReturnNumSet().begin()) +
                    "> *";
        } else {
          DiagnosticsUtils::report(getFilePath(), getBegin(),
                                   Diagnostics::UNDEDUCED_TYPE, true, false,
                                   "RNG engine");
          TypeStr =
              TypeStr + "<dpct_placeholder/*Fix the vec_size manually*/>*";
        }
      }

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
            TypeStr, Arg.getIdStringWithIndex(), " = (", TypeStr,
                        ")(&", Arg.getIdStringWithSuffix("acc"),
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

        KernelArgs += buildString("(", TypeStr, ")(&",
                                  Arg.getIdStringWithSuffix("acc"), "[0])");
      }
    } else if (Arg.IsRedeclareRequired || IsInMacroDefine) {
      SubmitStmtsList.CommandGroupList.emplace_back(buildString(
          "auto ", Arg.getIdStringWithIndex(), " = ", Arg.getArgString(), ";"));
      KernelArgs += Arg.getIdStringWithIndex();
    } else if (Arg.Texture) {
      ParameterStream OS;
      Arg.Texture->getKernelArg(OS);
      KernelArgs += OS.Str;
    } else {
      KernelArgs += Arg.getArgString();
    }
    ArgCounter += 1;
  }

  // if all params have no default value, insert extra args in the end of params
  if (getFuncInfo()) {
    if (ArgCounter == getFuncInfo()->NonDefaultParamNum) {
      KernelArgs = KernelArgs + getExtraArguments();
    }
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
      Printer.line(S.StmtStr);
  }
  if (NeedLambda) {
    Block = std::move(Printer.block(true));
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
  if (ExecutionConfig.Stream[0] == '*' || ExecutionConfig.Stream[0] == '&') {
    Printer << "(" << ExecutionConfig.Stream << ")";
  }
  else {
    Printer << ExecutionConfig.Stream;
  }
  if (isDefaultStream())
    Printer << ".";
  else
    Printer << "->";
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
    Printer.line(
        "cgh.parallel_for<dpct_kernel_name<class ", getName(), "_",
        LocInfo.LocHash,
        (hasTemplateArgs() ? (", " + getTemplateArguments(false, true)) : ""),
        ">>(");
  } else {
    Printer.line("cgh.parallel_for(");
  }
  auto B = Printer.block();
  static std::string CanIgnoreRangeStr3D =
      DpctGlobalInfo::getCtadClass(MapNames::getClNamespace() + "::range", 3) +
      "(1, 1, 1)";
  static std::string CanIgnoreRangeStr1D =
      DpctGlobalInfo::getCtadClass(MapNames::getClNamespace() + "::range", 1) +
      "(1)";

  if (DpctGlobalInfo::getAssumedNDRangeDim() == 1 && getFuncInfo() &&
      MemVarMap::getHeadWithoutPathCompression(&(getFuncInfo()->getVarMap())) &&
      MemVarMap::getHeadWithoutPathCompression(&(getFuncInfo()->getVarMap()))
              ->Dim == 1) {
    DpctGlobalInfo::printCtadClass(Printer.indent(),
                                   MapNames::getClNamespace() + "::nd_range", 1)
        << "(";
    if (ExecutionConfig.GroupSizeFor1D == CanIgnoreRangeStr1D) {
      Printer << ExecutionConfig.LocalSizeFor1D;
    } else if (ExecutionConfig.LocalSizeFor1D == CanIgnoreRangeStr1D) {
      Printer << ExecutionConfig.GroupSizeFor1D;
    } else {
      Printer << ExecutionConfig.GroupSizeFor1D << " * "
              << ExecutionConfig.LocalSizeFor1D;
    }
    Printer << ", ";
    Printer << ExecutionConfig.LocalSizeFor1D;
    (Printer << "), ").newLine();
    Printer.line("[=](" + MapNames::getClNamespace() + "::nd_item<1> ",
                 DpctGlobalInfo::getItemName(), ") {");
  } else {
    DpctGlobalInfo::printCtadClass(Printer.indent(),
                                   MapNames::getClNamespace() + "::nd_range", 3)
        << "(";
    if (ExecutionConfig.GroupSize == CanIgnoreRangeStr3D) {
      Printer << ExecutionConfig.LocalSize;
    } else if (ExecutionConfig.LocalSize == CanIgnoreRangeStr3D) {
      Printer << ExecutionConfig.GroupSize;
    } else {
      Printer << ExecutionConfig.GroupSize << " * "
              << ExecutionConfig.LocalSize;
    }
    Printer << ", ";
    Printer << ExecutionConfig.LocalSize;
    (Printer << "), ").newLine();
    Printer.line("[=](" + MapNames::getClNamespace() + "::nd_item<3> ",
                 DpctGlobalInfo::getItemName(), ") {");
  }

  printKernel(Printer);
  Printer.line("});");
}

void KernelCallExpr::printKernel(KernelPrinter &Printer) {
  auto B = Printer.block();
  for (auto &S : KernelStmts)
    Printer.line(S.StmtStr);
  Printer.indent() << getName()
                   << (hasWrittenTemplateArgs()
                           ? buildString("<", getTemplateArguments(), ">")
                           : "")
                   << "(" << KernelArgs << ");";
  Printer.newLine();
}

std::string KernelCallExpr::getReplacement() {
  addAccessorDecl();
  addStreamDecl();
  buildKernelArgsStmt();

  if (IsInMacroDefine) {
    LocInfo.NL = "\\" + LocInfo.NL;
  }
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  KernelPrinter Printer(LocInfo.NL, LocInfo.Indent, OS);
  print(Printer);
  auto ResultStr = Printer.str();
  if (NeedLambda) {
    ResultStr = "[&]()" + Printer.str() + "()";
  }
  return ResultStr;
}

inline std::string CallFunctionExpr::getExtraArguments() {
  if (!FuncInfo)
    return "";
  return getVarMap().getExtraCallArguments(FuncInfo->NonDefaultParamNum,
                                           FuncInfo->ParamsNum -
                                               FuncInfo->NonDefaultParamNum);
}

const DeclRefExpr *getAddressedRef(const Expr *E) {
  E = E->IgnoreImplicitAsWritten();
  if (auto DRE = dyn_cast<DeclRefExpr>(E)) {
    if (DRE->getDecl()->getKind() == Decl::Function) {
      return DRE;
    }
  } else if (auto Paren = dyn_cast<ParenExpr>(E)) {
    return getAddressedRef(Paren->getSubExpr());
  } else if (auto Cast = dyn_cast<CastExpr>(E)) {
    return getAddressedRef(Cast->getSubExprAsWritten());
  } else if (auto UO = dyn_cast<UnaryOperator>(E)) {
    if (UO->getOpcode() == UO_AddrOf) {
      return getAddressedRef(UO->getSubExpr());
    }
  }
  return nullptr;
}

std::shared_ptr<KernelCallExpr> KernelCallExpr::buildFromCudaLaunchKernel(
    const std::pair<std::string, unsigned> &LocInfo, const CallExpr *CE) {
  auto LaunchFD = CE->getDirectCallee();
  if (!LaunchFD || (LaunchFD->getName() != "cudaLaunchKernel" &&
      LaunchFD->getName() != "cudaLaunchCooperativeKernel")) {
    return std::shared_ptr<KernelCallExpr>();
  }
  if (auto Callee = getAddressedRef(CE->getArg(0))) {
    auto Kernel = std::shared_ptr<KernelCallExpr>(
        new KernelCallExpr(LocInfo.second, LocInfo.first));
    Kernel->buildCalleeInfo(Callee);
    Kernel->buildLocationInfo(CE);
    Kernel->buildExecutionConfig(ArrayRef<const Expr *>{
        CE->getArg(1), CE->getArg(2), CE->getArg(4), CE->getArg(5)});
    Kernel->buildNeedBracesInfo(CE);
    auto FD =
        dyn_cast_or_null<FunctionDecl>(Callee->getReferencedDeclOfCallee());
    auto FuncInfo = Kernel->getFuncInfo();
    if (FD && FuncInfo) {
      auto ArgsArray = ExprAnalysis::ref(CE->getArg(3));
      if (!isa<DeclRefExpr>(CE->getArg(3)->IgnoreImplicitAsWritten())) {
        ArgsArray = "(" + ArgsArray + ")";
      }
      Kernel->resizeTextureObjectList(FD->getNumParams());
      for (auto &Parm : FD->parameters()) {
        Kernel->ArgsInfo.emplace_back(Parm, ArgsArray, Kernel.get());
      }
    }
    return Kernel;
  }
  return std::shared_ptr<KernelCallExpr>();
}

void KernelCallExpr::setDim() {
  if (auto InfoPtr = getFuncInfo()) {
    if (auto HeadMemVarMapPtr = MemVarMap::getHead(&(InfoPtr->getVarMap()))) {
      if (InfoPtr->getVarMap().Dim == 3) {
        HeadMemVarMapPtr->Dim = 3;
      }
    } else {
      llvm_unreachable("The head of current node is not available!");
    }
  }
}

void KernelCallExpr::buildUnionFindSet() {
  if (auto Ptr = getFuncInfo()) {
    if (GridDim == 1 && BlockDim == 1)
      Ptr->getVarMap().Dim = 1;
    else
      Ptr->getVarMap().Dim = 3;
    constructUnionFindSetRecursively(Ptr);
  }
}

void KernelCallExpr::buildInfo() {
  CallFunctionExpr::buildInfo();
  TotalArgsSize =
      getVarMap().calculateExtraArgsSize() + calculateOriginArgsSize();
}

void KernelCallExpr::addReplacements() {
  if (TotalArgsSize >
      MapNames::KernelArgTypeSizeMap.at(KernelArgType::MaxParameterSize))
    DiagnosticsUtils::report(getFilePath(), getBegin(),
                             Diagnostics::EXCEED_MAX_PARAMETER_SIZE, true,
                             false);
  // TODO: Output debug info.
  auto R = std::make_shared<ExtReplacement>(getFilePath(), getBegin(), 0,
                                            getReplacement(), nullptr);
  R->setBlockLevelFormatFlag();
  DpctGlobalInfo::getInstance().addReplacement(R);
}

void CallFunctionExpr::buildTemplateArgumentsFromTypeLoc(const TypeLoc &TL) {
  if (!TL)
    return;
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
  // Check if the whole kernel call is in macro arg
  auto CallBegin = KernelCall->getBeginLoc();
  auto CallEnd = KernelCall->getEndLoc();

  if (SM.isMacroArgExpansion(CallBegin) && SM.isMacroArgExpansion(CallEnd) &&
      isLocInSameMacroArg(CallBegin, CallEnd)) {
    IsInMacroDefine = false;
    return;
  }

  auto CalleeSpelling = KernelCall->getCallee()->getBeginLoc();
  if (SM.isMacroArgExpansion(CalleeSpelling)) {
    CalleeSpelling = SM.getImmediateExpansionRange(CalleeSpelling).getBegin();
  }
  CalleeSpelling = SM.getSpellingLoc(CalleeSpelling);

  auto ItMatch = dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
    getCombinedStrFromLoc(CalleeSpelling));
  if (ItMatch != dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end()) {
    IsInMacroDefine = true;
  }
}

// If the kernel call is in a ParenExpr
void KernelCallExpr::setNeedAddLambda(const CUDAKernelCallExpr *KernelCall) {
  if (dyn_cast<ParenExpr>(getParentStmt(KernelCall))) {
    NeedLambda = true;
  }
}

#define TYPE_CAST(qual_type, type) dyn_cast<type>(qual_type)
#define ARG_TYPE_CAST(type) TYPE_CAST(ArgType, type)
#define PARM_TYPE_CAST(type) TYPE_CAST(ParmType, type)

bool TemplateArgumentInfo::isPlaceholderType(QualType QT) {
  if (auto BT = QT->getAs<BuiltinType>()) {
    if (BT->isPlaceholderType() || BT->isDependentType())
      return true;
  }
  return false;
}

template <class T>
void setTypeTemplateArgument(std::vector<TemplateArgumentInfo> &TAILis,
                             unsigned Idx, T Ty) {
  auto &TA = TAILis[Idx];
  if (TA.isNull())
    TA.setAsType(Ty);
}
template <class T>
void setNonTypeTemplateArgument(std::vector<TemplateArgumentInfo> &TAILis,
                                unsigned Idx, T Ty) {
  auto &TA = TAILis[Idx];
  if (TA.isNull())
    TA.setAsNonType(Ty);
}

/// Return true if Ty is TypedefType.
bool getInnerType(QualType &Ty, TypeLoc &TL) {
  if (auto TypedefTy = dyn_cast<TypedefType>(Ty)) {
    if (!TemplateArgumentInfo::isPlaceholderType(TypedefTy->desugar())) {
      Ty = TypedefTy->desugar();
      TL = TypedefTy->getDecl()->getTypeSourceInfo()->getTypeLoc();
      return true;
    }
  } else if (auto ElaboratedTy = dyn_cast<ElaboratedType>(Ty)) {
    Ty = ElaboratedTy->getNamedType();
    if (TL)
      TL = TYPELOC_CAST(ElaboratedTypeLoc).getNamedTypeLoc();
    return true;
  }
  return false;
}

void deduceTemplateArgumentFromType(std::vector<TemplateArgumentInfo> &TAIList,
                                    QualType ParmType, QualType ArgType,
                                    TypeLoc TL = TypeLoc());

template <class NonTypeValueT>
void deduceNonTypeTemplateArgument(std::vector<TemplateArgumentInfo> &TAIList,
                                   const Expr *Parm,
                                   const NonTypeValueT &Value) {
  Parm = Parm->IgnoreImplicitAsWritten();
  if (auto DRE = dyn_cast<DeclRefExpr>(Parm)) {
    if (auto NTTPD = dyn_cast<NonTypeTemplateParmDecl>(DRE->getDecl())) {
      setNonTypeTemplateArgument(TAIList, NTTPD->getIndex(), Value);
    }
  } else if (auto C = dyn_cast<ConstantExpr>(Parm)) {
    deduceNonTypeTemplateArgument(TAIList, C->getSubExpr(), Value);
  } else if (auto S = dyn_cast<SubstNonTypeTemplateParmExpr>(Parm)) {
    deduceNonTypeTemplateArgument(TAIList, S->getReplacement(), Value);
  }
}

void deduceTemplateArgumentFromTemplateArgs(
    std::vector<TemplateArgumentInfo> &TAIList, const TemplateArgument &Parm,
    const TemplateArgument &Arg,
    const TemplateArgumentLoc &ArgLoc = TemplateArgumentLoc()) {
  switch (Parm.getKind()) {
  case TemplateArgument::Expression:
    switch (Arg.getKind()) {
    case TemplateArgument::Expression:
      deduceNonTypeTemplateArgument(TAIList, Parm.getAsExpr(), Arg.getAsExpr());
      return;
    case TemplateArgument::Integral:
      if (ArgLoc.getArgument().isNull())
        deduceNonTypeTemplateArgument(TAIList, Parm.getAsExpr(),
                                      Arg.getAsIntegral());
      else
        deduceNonTypeTemplateArgument(TAIList, Parm.getAsExpr(),
                                      ArgLoc.getSourceExpression());
      break;
    default:
      break;
    }
    break;
  case TemplateArgument::Type:
    if (Arg.getKind() != TemplateArgument::Type)
      return;
    if (ArgLoc.getArgument().isNull()) {
      deduceTemplateArgumentFromType(TAIList, Parm.getAsType(),
                                     Arg.getAsType());
    } else {
      deduceTemplateArgumentFromType(TAIList, Parm.getAsType(),
                                     ArgLoc.getTypeSourceInfo()->getType(),
                                     ArgLoc.getTypeSourceInfo()->getTypeLoc());
    }
  default:
    break;
  }
}

void deduceTemplateArgumentFromTemplateSpecialization(
    std::vector<TemplateArgumentInfo> &TAIList,
    const TemplateSpecializationType *ParmType, QualType ArgType,
    TypeLoc TL = TypeLoc()) {
  switch (ArgType->getTypeClass()) {
  case Type::Record:
    if (auto CTSD = dyn_cast<ClassTemplateSpecializationDecl>(
            ARG_TYPE_CAST(RecordType)->getDecl())) {
      if (CTSD->getTypeAsWritten() &&
          CTSD->getTypeAsWritten()->getType()->getTypeClass() ==
              Type::TemplateSpecialization) {
        auto TL = CTSD->getTypeAsWritten()->getTypeLoc();
        auto &TSTL = TYPELOC_CAST(TemplateSpecializationTypeLoc);
        for (unsigned i = 0; i < ParmType->getNumArgs(); ++i) {
          deduceTemplateArgumentFromTemplateArgs(
              TAIList, ParmType->getArg(i), TSTL.getArgLoc(i).getArgument(),
              TSTL.getArgLoc(i));
        }
      }
    }
    break;
  case Type::TemplateSpecialization:
    if (TL) {
      auto &TSTL = TYPELOC_CAST(TemplateSpecializationTypeLoc);
      for (unsigned i = 0; i < ParmType->getNumArgs(); ++i) {
        deduceTemplateArgumentFromTemplateArgs(TAIList, ParmType->getArg(i),
                                               TSTL.getArgLoc(i).getArgument(),
                                               TSTL.getArgLoc(i));
      }
    } else {
      auto TST = ARG_TYPE_CAST(TemplateSpecializationType);
      for (unsigned i = 0; i < ParmType->getNumArgs(); ++i) {
        deduceTemplateArgumentFromTemplateArgs(TAIList, ParmType->getArg(i),
                                               TST->getArg(i));
      }
    }
    break;
  default:
    break;
  }
}

TypeLoc getPointeeTypeLoc(TypeLoc TL) {
  if (!TL)
    return TL;
  switch (TL.getTypeLocClass()) {
  case TypeLoc::ConstantArray:
  case TypeLoc::DependentSizedArray:
  case TypeLoc::IncompleteArray:
    return TYPELOC_CAST(ArrayTypeLoc).getElementLoc();
  case TypeLoc::Pointer:
    return TYPELOC_CAST(PointerTypeLoc).getPointeeLoc();
  default:
    return TypeLoc();
  }
}

void deduceTemplateArgumentFromArrayElement(
    std::vector<TemplateArgumentInfo> &TAIList, QualType ParmType,
    QualType ArgType, TypeLoc TL = TypeLoc()) {
  const ArrayType *ParmArray = PARM_TYPE_CAST(ArrayType);
  const ArrayType *ArgArray = ARG_TYPE_CAST(ArrayType);
  if (!ParmArray || !ArgArray) {
    return;
  }
  deduceTemplateArgumentFromType(TAIList, ParmArray->getElementType(),
                                 ArgArray->getElementType(),
                                 getPointeeTypeLoc(TL));
}

void deduceTemplateArgumentFromType(std::vector<TemplateArgumentInfo> &TAIList,
                                    QualType ParmType, QualType ArgType,
                                    TypeLoc TL) {
  ParmType = ParmType.getCanonicalType();
  if (!ParmType->isDependentType())
    return;

  if (TL) {
    TL = TL.getUnqualifiedLoc();
    if (TL.getTypePtr()->getTypeClass() != ArgType->getTypeClass() ||
        TL.getTypeLocClass() == TypeLoc::SubstTemplateTypeParm)
      TL = TypeLoc();
  }

  switch (ParmType->getTypeClass()) {
  case Type::TemplateTypeParm:
    if (TL) {
      setTypeTemplateArgument(
          TAIList, PARM_TYPE_CAST(TemplateTypeParmType)->getIndex(), TL);
    } else {
      ArgType.removeLocalCVRQualifiers(ParmType.getCVRQualifiers());
      setTypeTemplateArgument(
          TAIList, PARM_TYPE_CAST(TemplateTypeParmType)->getIndex(), ArgType);
    }
    break;
  case Type::TemplateSpecialization:
    deduceTemplateArgumentFromTemplateSpecialization(
        TAIList, PARM_TYPE_CAST(TemplateSpecializationType), ArgType, TL);
    break;
  case Type::Pointer:
    if (auto ArgPointer = ARG_TYPE_CAST(PointerType)) {
      deduceTemplateArgumentFromType(TAIList, ParmType->getPointeeType(),
                                     ArgPointer->getPointeeType(),
                                     getPointeeTypeLoc(TL));
    } else if (auto ArgArray = ARG_TYPE_CAST(ArrayType)) {
      deduceTemplateArgumentFromType(TAIList, ParmType->getPointeeType(),
                                     ArgArray->getElementType(),
                                     getPointeeTypeLoc(TL));
    }
    break;
  case Type::LValueReference: {
    auto ParmPointeeType =
        PARM_TYPE_CAST(LValueReferenceType)->getPointeeTypeAsWritten();
    if (auto LVRT = ARG_TYPE_CAST(LValueReferenceType)) {
      deduceTemplateArgumentFromType(
          TAIList, ParmPointeeType, LVRT->getPointeeTypeAsWritten(),
          TL ? TYPELOC_CAST(LValueReferenceTypeLoc).getPointeeLoc() : TL);
    } else if (ParmPointeeType.getQualifiers().hasConst()) {
      deduceTemplateArgumentFromType(TAIList, ParmPointeeType, ArgType, TL);
    }
    break;
  }
  case Type::RValueReference:
    if (auto RVRT = ARG_TYPE_CAST(RValueReferenceType)) {
      deduceTemplateArgumentFromType(
          TAIList,
          PARM_TYPE_CAST(RValueReferenceType)->getPointeeTypeAsWritten(),
          RVRT->getPointeeTypeAsWritten(),
          TL ? TYPELOC_CAST(RValueReferenceTypeLoc).getPointeeLoc() : TL);
    }
    break;
  case Type::ConstantArray: {
    auto ArgConstArray = ARG_TYPE_CAST(ConstantArrayType);
    auto ParmConstArray = PARM_TYPE_CAST(ConstantArrayType);
    if (ArgConstArray &&
        ArgConstArray->getSize() == ParmConstArray->getSize()) {
      deduceTemplateArgumentFromArrayElement(TAIList, ParmType, ArgType, TL);
    }
    break;
  }
  case Type::IncompleteArray:
    deduceTemplateArgumentFromArrayElement(TAIList, ParmType, ArgType, TL);
    break;
  case Type::DependentSizedArray: {
    deduceTemplateArgumentFromArrayElement(TAIList, ParmType, ArgType, TL);
    auto ParmSizeExpr = PARM_TYPE_CAST(DependentSizedArrayType)->getSizeExpr();
    if (TL && TL.getTypePtr()->isArrayType()) {
      deduceNonTypeTemplateArgument(TAIList, ParmSizeExpr,
                                    TYPELOC_CAST(ArrayTypeLoc).getSizeExpr());
    } else if (auto DSAT = ARG_TYPE_CAST(DependentSizedArrayType)) {
      deduceNonTypeTemplateArgument(TAIList, ParmSizeExpr, DSAT->getSizeExpr());
    } else if (auto CAT = ARG_TYPE_CAST(ConstantArrayType)) {
      deduceNonTypeTemplateArgument(TAIList, ParmSizeExpr, CAT->getSize());
    }
    break;
  }
  default:
    break;
  }

  if (getInnerType(ArgType, TL)) {
    deduceTemplateArgumentFromType(TAIList, ParmType, ArgType, TL);
  }
}

void deduceTemplateArgument(std::vector<TemplateArgumentInfo> &TAIList,
                            const Expr *Arg, const ParmVarDecl *PVD) {
  auto ArgType = Arg->getType();
  auto ParmType = PVD->getType();

  TypeLoc TL;
  if (auto DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreImplicitAsWritten())) {
    if (auto DD = dyn_cast<DeclaratorDecl>(DRE->getDecl()))
      TL = DD->getTypeSourceInfo()->getTypeLoc();
  }

  deduceTemplateArgumentFromType(TAIList, ParmType, ArgType, TL);
}

template <class CallT>
void deduceTemplateArguments(const CallT *C, const FunctionTemplateDecl *FTD,
                             std::vector<TemplateArgumentInfo> &TAIList) {
  if (!FTD)
    return;

  if (!DpctGlobalInfo::isInRoot(FTD->getBeginLoc()))
    return;
  auto &TemplateParmsList = *FTD->getTemplateParameters();
  if (TAIList.size() == TemplateParmsList.size())
    return;

  TAIList.resize(TemplateParmsList.size());

  auto ArgItr = C->arg_begin();
  auto ParmItr = FTD->getTemplatedDecl()->param_begin();
  while (ArgItr != C->arg_end() &&
         ParmItr != FTD->getTemplatedDecl()->param_end()) {
    deduceTemplateArgument(TAIList, *ArgItr, *ParmItr);
    ++ArgItr;
    ++ParmItr;
  }
  for (size_t i = 0; i < TAIList.size(); ++i) {
    auto &Arg = TAIList[i];
    if (!Arg.isNull())
      continue;
    auto TemplateParm = TemplateParmsList.getParam(i);
    if (auto TTPD = dyn_cast<TemplateTypeParmDecl>(TemplateParm)) {
      if (TTPD->hasDefaultArgument())
        Arg.setAsType(TTPD->getDefaultArgumentInfo()->getTypeLoc());
    } else if (auto NTTPD = dyn_cast<NonTypeTemplateParmDecl>(TemplateParm)) {
      if (NTTPD->hasDefaultArgument())
        Arg.setAsNonType(NTTPD->getDefaultArgument());
    }
  }
}

template <class CallT>
void deduceTemplateArguments(const CallT *C, const FunctionDecl *FD,
                             std::vector<TemplateArgumentInfo> &TAIList) {
  if (FD)
    return deduceTemplateArguments(C, FD->getPrimaryTemplate(), TAIList);
}

template <class CallT>
void deduceTemplateArguments(const CallT *C, const NamedDecl *ND,
                             std::vector<TemplateArgumentInfo> &TAIList) {
  if (!ND)
    return;
  if (auto FTD = dyn_cast<FunctionTemplateDecl>(ND)) {
    deduceTemplateArguments(C, FTD, TAIList);
  } else if (auto FD = dyn_cast<FunctionDecl>(ND)) {
    deduceTemplateArguments(C, FD, TAIList);
  } else if (auto UD = dyn_cast<UsingShadowDecl>(ND)) {
    deduceTemplateArguments(C, UD->getUnderlyingDecl(), TAIList);
  }
}

/// This function gets the \p FD name with the necessary qualified namespace at
/// \p Callee position.
/// Method:
/// 1. record all NamespaceDecl nodes of the ancestors \p FD and \p Callee, get
/// two namespace sequences. E.g.,
///   decl: aaa,bbb,ccc; callee: aaa,eee;
/// 2. Remove the longest continuous common subsequence
/// 3. the rest sequence of \p FD is the namespace sequence
std::string CallFunctionExpr::getNameWithNamespace(const FunctionDecl *FD,
                                                   const Expr *Callee) {
  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto getNamespaceSeq =
      [&](DynTypedNodeList Parents) -> std::deque<std::string> {
    std::deque<std::string> Seq;
    while (Parents.size() > 0) {
      auto *Parent = Parents[0].get<NamespaceDecl>();
      if (Parent) {
        Seq.push_front(Parent->getNameAsString());
      }
      Parents = Context.getParents(Parents[0]);
    }
    return Seq;
  };

  std::deque<std::string> FDNamespaceSeq =
      getNamespaceSeq(Context.getParents(*FD));
  std::deque<std::string> CalleeNamespaceSeq =
      getNamespaceSeq(Context.getParents(*Callee));

  auto FDIter = FDNamespaceSeq.begin();
  for (auto CalleeNamespace : CalleeNamespaceSeq) {
    if (FDNamespaceSeq.empty())
      break;

    if (CalleeNamespace == *FDIter) {
      FDIter++;
      FDNamespaceSeq.pop_front();
    } else {
      break;
    }
  }

  std::string Result;
  for (auto I : FDNamespaceSeq) {
    // If I is empty, it means this namespace is an unnamed namespace. So its
    // members have internal linkage. So just remove it.
    if (I.empty())
      continue;
    Result = Result + I + "::";
  }

  return Result + getName(FD);
}

void CallFunctionExpr::buildCalleeInfo(const Expr *Callee) {
  if (auto CallDecl =
          dyn_cast_or_null<FunctionDecl>(Callee->getReferencedDeclOfCallee())) {
    Name = getNameWithNamespace(CallDecl, Callee);
    FuncInfo = DeviceFunctionDecl::LinkRedecls(CallDecl);
    if (auto DRE = dyn_cast<DeclRefExpr>(Callee)) {
      buildTemplateArguments(DRE->template_arguments());
    }
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
}
SourceLocation getActualInsertLocation(SourceLocation InsertLoc,
                                       const SourceManager &SM,
                                       const LangOptions &LO);
void CallFunctionExpr::buildCallExprInfo(const CXXConstructExpr *Ctor) {
  if (!Ctor)
    return;
  if (Ctor->getParenOrBraceRange().isInvalid())
    return;

  buildTextureObjectArgsInfo(Ctor);

  auto CtorDecl = Ctor->getConstructor();
  Name = getName(CtorDecl);
  FuncInfo = DeviceFunctionDecl::LinkRedecls(CtorDecl);
  deduceTemplateArguments(Ctor, CtorDecl, TemplateArgs);

  SourceLocation InsertLocation;
  auto &SM = DpctGlobalInfo::getSourceManager();
  if (FuncInfo) {
    if (FuncInfo->NonDefaultParamNum) {
      if (Ctor->getNumArgs() >= FuncInfo->NonDefaultParamNum) {
        InsertLocation =
            Ctor->getArg(FuncInfo->NonDefaultParamNum - 1)->getEndLoc();
      } else {
        ExtraArgLoc = 0;
        return;
      }
    } else {
      InsertLocation = Ctor->getParenOrBraceRange().getBegin();
    }
  }
  ExtraArgLoc = SM.getFileOffset(Lexer::getLocForEndOfToken(
      getActualInsertLocation(InsertLocation, SM,
                              DpctGlobalInfo::getContext().getLangOpts()),
      0, SM, DpctGlobalInfo::getContext().getLangOpts()));
}

void CallFunctionExpr::buildCallExprInfo(const CallExpr *CE) {
  if (!CE)
    return;
  buildCalleeInfo(CE->getCallee()->IgnoreParenImpCasts());
  buildTextureObjectArgsInfo(CE);

  bool HasImplicitArg = false;
  if (auto FD = CE->getDirectCallee()) {
    deduceTemplateArguments(CE, FD, TemplateArgs);
    HasImplicitArg = isa<CXXOperatorCallExpr>(CE) && isa<CXXMethodDecl>(FD);
  } else if (auto Unresolved = dyn_cast<UnresolvedLookupExpr>(
                 CE->getCallee()->IgnoreImplicitAsWritten())) {
    if (Unresolved->getNumDecls())
      deduceTemplateArguments(CE, Unresolved->decls_begin().getDecl(),
		  TemplateArgs);
  }

  if (HasImplicitArg) {
    HasArgs = CE->getNumArgs() == 1;
  } else {
    HasArgs = CE->getNumArgs();
  }

  if (FuncInfo) {
    if (FuncInfo->ParamsNum == 0) {
      ExtraArgLoc =
          DpctGlobalInfo::getSourceManager().getFileOffset(CE->getRParenLoc());
    } else if (FuncInfo->NonDefaultParamNum == 0) {
      // if all params have default value
      ExtraArgLoc = DpctGlobalInfo::getSourceManager().getFileOffset(
          CE->getArg(HasImplicitArg ? 1 : 0)->getBeginLoc());
    } else {
      // if some params have default value, set ExtraArgLoc to the location
      // before the comma
      if (CE->getNumArgs() > FuncInfo->NonDefaultParamNum - 1) {
        auto &SM = DpctGlobalInfo::getSourceManager();
        auto TokenLoc = Lexer::getLocForEndOfToken(
            getActualInsertLocation(
                CE->getArg(FuncInfo->NonDefaultParamNum - 1 + HasImplicitArg)
                    ->getEndLoc(),
                SM, DpctGlobalInfo::getContext().getLangOpts()),
            0, SM, DpctGlobalInfo::getContext().getLangOpts());
        ExtraArgLoc =
            DpctGlobalInfo::getSourceManager().getFileOffset(TokenLoc);
      } else {
        ExtraArgLoc = 0;
      }
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

std::shared_ptr<TextureObjectInfo> CallFunctionExpr::addTextureObjectArg(
    unsigned ArgIdx, const ArraySubscriptExpr *TexRef, bool isKernelCall) {
  if (TextureObjectInfo::isTextureObject(TexRef)) {
    if (auto Base =
            dyn_cast<DeclRefExpr>(TexRef->getBase()->IgnoreImpCasts())) {
      if (isKernelCall) {
        if (auto VD = dyn_cast<VarDecl>(Base->getDecl())) {
          return addTextureObjectArgInfo(
              ArgIdx, std::make_shared<TextureObjectInfo>(
                          VD, ExprAnalysis::ref(TexRef->getIdx())));
        }
      } else if (auto PVD = dyn_cast<ParmVarDecl>(Base->getDecl())) {
        return addTextureObjectArgInfo(
            ArgIdx, std::make_shared<TextureObjectInfo>(
                        PVD, ExprAnalysis::ref(TexRef->getIdx())));
      }
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
  if (!DefFilePath.empty() && DefFilePath != getFilePath() &&
      !isIncludedFile(getFilePath(), DefFilePath)) {
    FuncInfo->setNeedSyclExternMacro();
  }

  FuncInfo->buildInfo();
  VarMap.merge(FuncInfo->getVarMap(), TemplateArgs);
  mergeTextureObjectTypeInfo();
}

void CallFunctionExpr::emplaceReplacement() {
  buildInfo();

  if (ExtraArgLoc)
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(FilePath, ExtraArgLoc, 0,
                                         getExtraArguments(), nullptr));
}

std::string CallFunctionExpr::getTemplateArguments(bool WrittenArgsOnly,
                                                   bool WithScalarWrapped) {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  for (auto &TA : TemplateArgs) {
    if ((TA.isNull() || !TA.isWritten()) && WrittenArgsOnly)
      continue;
    if (WithScalarWrapped && (!TA.isType() && !TA.isNull()))
      appendString(OS, "dpct_kernel_scalar<", TA.getString(), ">, ");
    else
      appendString(OS, TA.getString(), ", ");
  }
  OS.flush();
  return (Result.empty()) ? Result : Result.erase(Result.size() - 2);
}

void ExplicitInstantiationDecl::initTemplateArgumentList(
    const TemplateArgumentListInfo &TAList,
    const FunctionDecl *Specialization) {
  ExprAnalysis EA;
  auto &SM = DpctGlobalInfo::getSourceManager();
  for (auto &ArgLoc : TAList.arguments()) {
    EA.analyze(ArgLoc);
    if (EA.hasReplacement()) {
      DpctGlobalInfo::getInstance().addReplacement(
          std::make_shared<ExtReplacement>(SM, &ArgLoc, EA.getReplacedString(),
                                           nullptr));
    }
  }

  if (Specialization->getTemplateSpecializationArgs() == nullptr)
    return;
  for (auto &Arg : Specialization->getTemplateSpecializationArgs()->asArray()) {
    TemplateArgumentInfo TA;
    switch (Arg.getKind()) {
    case TemplateArgument::Integral:
      TA.setAsNonType(Arg.getAsIntegral());
      break;
    case TemplateArgument::Expression:
      TA.setAsNonType(Arg.getAsExpr());
      break;
    case TemplateArgument::Type:
      TA.setAsType(Arg.getAsType());
      break;
    default:
      break;
    }
    InstantiationArgs.emplace_back(std::move(TA));
  }
}

void processTypeLoc(const TypeLoc &TL, ExprAnalysis &EA,
                    const SourceManager &SM) {
  EA.analyze(TL);
  if (EA.hasReplacement()) {
    DpctGlobalInfo::getInstance().addReplacement(
      std::make_shared<ExtReplacement>(SM, &TL, EA.getReplacedString(),
        nullptr));
  }
}

void ExplicitInstantiationDecl::processFunctionTypeLoc(
    const FunctionTypeLoc &FTL) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  ExprAnalysis EA;
  processTypeLoc(FTL.getReturnLoc(), EA, SM);
  for (auto Parm : FTL.getParams()) {
    processTypeLoc(Parm->getTypeSourceInfo()->getTypeLoc(), EA, SM);
  }
}

void DeviceFunctionInfo::merge(std::shared_ptr<DeviceFunctionInfo> Other) {
  if (this == Other.get())
    return;
  VarMap.merge(Other->getVarMap());
  dpct::merge(CallExprMap, Other->CallExprMap);
  mergeTextureTypeList(Other->TextureObjectTypeList);
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
  VarMap.removeDuplicateVar();
}

std::string DeviceFunctionDecl::getExtraParameters() {
  return FuncInfo->getExtraParameters(FormatInformation);
}

std::string ExplicitInstantiationDecl::getExtraParameters() {
  return getFuncInfo()->getExtraParameters(InstantiationArgs, getFormatInfo());
}

inline void DeviceFunctionDecl::emplaceReplacement() {
  // TODO: Output debug info.
  auto Repl = std::make_shared<ExtReplacement>(
      FilePath, ReplaceOffset, ReplaceLength, getExtraParameters(), nullptr);
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
        // Type dpct_placeholder
        Obj->setType("dpct_placeholder/*Fix the type manually*/", 1);
        DiagnosticsUtils::report(Obj->getFilePath(), Obj->getOffset(),
                                 Diagnostics::UNDEDUCED_TYPE, true, false,
                                 "image_accessor_ext");
      }
      Obj->addParamDeclReplacement();
    }
  }
}

DeviceFunctionDecl::DeviceFunctionDecl(unsigned Offset,
                                       const std::string &FilePathIn,
                                       const FunctionDecl *FD)
    : Offset(Offset), FilePath(FilePathIn), ParamsNum(FD->param_size()),
      ReplaceOffset(0), ReplaceLength(0),
      NonDefaultParamNum(FD->getMostRecentDecl()->getMinRequiredArguments()),
      FuncInfo(getFuncInfo(FD)) {
  if (!FuncInfo)
    FuncInfo = std::make_shared<DeviceFunctionInfo>(FD->param_size(),
                                                    NonDefaultParamNum);
  if (!FilePath.empty()) {
    SourceProcessType FileType = GetSourceFileType(FilePath);
    if (!(FileType & TypeCudaHeader) && !(FileType & TypeCppHeader) &&
        FD->isThisDeclarationADefinition()) {
      FuncInfo->setDefinitionFilePath(FilePath);
    }
  }

  static AttrVec NullAttrs;
  buildReplaceLocInfo(
      FD->getTypeSourceInfo()->getTypeLoc().getAs<FunctionTypeLoc>(),
      FD->hasAttrs() ? FD->getAttrs() : NullAttrs);
  buildTextureObjectParamsInfo(FD->parameters());
}

DeviceFunctionDecl::DeviceFunctionDecl(unsigned Offset,
                                       const std::string &FilePathIn,
                                       const FunctionTypeLoc &FTL,
                                       const ParsedAttributes &Attrs,
                                       const FunctionDecl *Specialization)
    : Offset(Offset), FilePath(FilePathIn),
      ParamsNum(Specialization->getNumParams()), ReplaceOffset(0),
      ReplaceLength(0),
      NonDefaultParamNum(
          Specialization->getMostRecentDecl()->getMinRequiredArguments()),
      FuncInfo(getFuncInfo(Specialization)) {
  IsDefFilePathNeeded = false;

  buildReplaceLocInfo(FTL, Attrs);
  buildTextureObjectParamsInfo(FTL.getParams());
}

bool isInSameLine(SourceLocation First, SourceLocation Second,
                  const SourceManager &SM) {
  bool Invalid = false;
  return ::isInSameLine(SM.getExpansionLoc(First), SM.getExpansionLoc(Second),
                        SM, Invalid) &&
         !Invalid;
}

unsigned calculateCudaAttrLength(const AttributeCommonInfo &A,
  SourceLocation AlignLocation,
  const SourceManager &SM,
  const LangOptions &LO) {
  std::string Expected;
  switch (A.getParsedKind()) {
  case AttributeCommonInfo::AT_CUDAGlobal:
    Expected = "__global__";
    break;
  case AttributeCommonInfo::AT_CUDADevice:
    Expected = "__device__";
    break;
  case AttributeCommonInfo::AT_CUDAHost:
    Expected = "__host__";
    break;
  default:
    return 0;
  }

  auto Begin = SM.getExpansionLoc(A.getRange().getBegin());
  if (!isInSameLine(Begin, AlignLocation, SM))
    return 0;
  auto Length = Lexer::MeasureTokenLength(Begin, SM, LO);
  if (Expected.compare(0, std::string::npos, SM.getCharacterData(Begin),
    Length))
    return 0;
  return getLenIncludingTrailingSpaces(
    SourceRange(Begin, Begin.getLocWithOffset(Length)), SM);
}

template <class IteratorT>
unsigned calculateCudaAttrLength(IteratorT AttrBegin, IteratorT AttrEnd,
                                 SourceLocation AlignLoc,
                                 const SourceManager &SM,
                                 const LangOptions &LO) {
  unsigned Length = 0;

  if (SM.isMacroArgExpansion(AlignLoc))
    return 0;
  AlignLoc = SM.getExpansionLoc(AlignLoc);

  std::for_each(AttrBegin, AttrEnd, [&](const AttributeCommonInfo &A) {
    Length += calculateCudaAttrLength(A, AlignLoc, SM, LO);
  });

  return Length;
}

unsigned calculateCudaAttrLength(const ParsedAttributes &Attrs,
                                 SourceLocation AlignLoc,
                                 const SourceManager &SM,
                                 const LangOptions &LO) {
  return calculateCudaAttrLength(Attrs.begin(), Attrs.end(), AlignLoc, SM, LO);
}

unsigned calculateCudaAttrLength(const AttrVec &Attrs, SourceLocation AlignLoc,
                                 const SourceManager &SM,
                                 const LangOptions &LO) {
  struct AttrIterator
      : llvm::iterator_adaptor_base<AttrIterator, AttrVec::const_iterator,
                                    std::random_access_iterator_tag, Attr> {
    AttrIterator(AttrVec::const_iterator I) : iterator_adaptor_base(I) {}

    reference operator*() const { return **I; }
    friend class ParsedAttributesView;
  };
  return calculateCudaAttrLength(AttrIterator(Attrs.begin()),
                                 AttrIterator(Attrs.end()), AlignLoc, SM, LO);
}

bool isEachParamEachLine(const ArrayRef<ParmVarDecl *> Parms,
                         SourceManager &SM) {
  if (Parms.size() < 2)
    return false;
  auto Itr = Parms.begin();
  auto NextItr = Itr;
  while (++NextItr != Parms.end()) {
    if (isInSameLine((*Itr)->getBeginLoc(), (*NextItr)->getBeginLoc(), SM))
      return false;
    Itr = NextItr;
  }
  return true;
}

// PARAMETER INSERTING LOCATION RULES:
// 1. Origin parameters number <= 1
//    Do not add new line until longer than 80. The new line begin is aligned
//    with the end location of "("
// 2. Origin parameters number > 1
//    2.1 If each parameter is in a single line:
//           Each added parameter is in a single line.
//           The new line begin is aligned with the last parameter's line
//           begin
//    2.2 There are 2 parameters in one line:
//           Do not add new line until longer than 80.
//           The new line begin is aligned with the last parameter's line
//           begin
template <class AttrsT>
FormatInfo buildFormatInfo(const FunctionTypeLoc &FTL,
                           SourceLocation InsertLocation, const AttrsT &Attrs,
                           SourceManager &SM, const LangOptions &LO) {
  SourceLocation AlignLocation;
  FormatInfo Format;
  Format.EnableFormat = true;

  bool CurrentSameLineWithAlign = false;
  Format.IsAllParamsOneLine = false;
  Format.CurrentLength = SM.getExpansionColumnNumber(InsertLocation);

  if (FTL.getNumParams()) {
    Format.IsEachParamNL = isEachParamEachLine(FTL.getParams(), SM);
    auto FirstParmLoc = SM.getExpansionLoc(FTL.getParam(0)->getBeginLoc());
    if (CurrentSameLineWithAlign =
            isInSameLine(FirstParmLoc, InsertLocation, SM)) {
      AlignLocation = FirstParmLoc;
    } else {
      Format.NewLineIndentStr = getIndent(InsertLocation, SM).str();
      Format.NewLineIndentLength = Format.NewLineIndentStr.length();
      return Format;
    }
  } else {
    Format.IsEachParamNL = false;
    AlignLocation = SM.getExpansionLoc(FTL.getLParenLoc()).getLocWithOffset(1);
    CurrentSameLineWithAlign = isInSameLine(AlignLocation, InsertLocation, SM);
  }

  auto CudaAttrLength = calculateCudaAttrLength(Attrs, AlignLocation, SM, LO);
  Format.NewLineIndentLength =
      SM.getExpansionColumnNumber(AlignLocation) - CudaAttrLength - 1;
  Format.NewLineIndentStr.assign(Format.NewLineIndentLength, ' ');
  if (CurrentSameLineWithAlign)
    Format.CurrentLength -= CudaAttrLength;

  return Format;
}

SourceLocation getActualInsertLocation(SourceLocation InsertLoc,
                                       const SourceManager &SM,
                                       const LangOptions &LO) {
  do {
    if (InsertLoc.isFileID())
      return InsertLoc;

    if (SM.isAtEndOfImmediateMacroExpansion(InsertLoc.getLocWithOffset(
            Lexer::MeasureTokenLength(SM.getSpellingLoc(InsertLoc), SM, LO)))) {
      /// If InsertLoc is at the end of macro definition, continue find immediate
      /// expansion.
      /// example:
      /// #define BBB int bbb
      /// #define CALL foo(int aaa, BBB)
      /// The insert location should be at the end of BBB instead of the end of bbb.
      InsertLoc = SM.getImmediateExpansionRange(InsertLoc).getBegin();
    } else if (SM.isMacroArgExpansion(InsertLoc)) {
      /// If is macro argument, continue find if argument is macro or written
      /// code.
      /// example:
      /// #define BBB int b, int c = 0
      /// #define CALL(x) foo(int aaa, x)
      /// CALL(BBB)
      InsertLoc = SM.getImmediateSpellingLoc(InsertLoc);
    } else {
      /// Else return insert location directly,
      return InsertLoc;
    }
  } while (true);

  return InsertLoc;
}

template <class AttrsT>
void DeviceFunctionDecl::buildReplaceLocInfo(const FunctionTypeLoc &FTL,
                                             const AttrsT &Attrs) {
  if (!FTL)
    return;

  SourceLocation InsertLocation;
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto&LO = DpctGlobalInfo::getContext().getLangOpts();
  if (NonDefaultParamNum) {
    InsertLocation = FTL.getParam(NonDefaultParamNum - 1)->getEndLoc();
  } else {
    InsertLocation = FTL.getLParenLoc();
  }

  InsertLocation = getActualInsertLocation(InsertLocation, SM, LO);
  if (InsertLocation.isMacroID()) {
    InsertLocation = Lexer::getLocForEndOfToken(
        SM.getSpellingLoc(InsertLocation), 0, SM, LO);
    FormatInformation.EnableFormat = true;
    FormatInformation.IsAllParamsOneLine = true;
  } else {
    InsertLocation = Lexer::getLocForEndOfToken(InsertLocation, 0, SM, LO);
    FormatInformation = buildFormatInfo(FTL, InsertLocation, Attrs, SM, LO);
  }
  FormatInformation.IsFirstArg = (NonDefaultParamNum == 0);

  // Keep skiping #ifdef #endif pair
  Token TokOfHash;
  if (!Lexer::getRawToken(InsertLocation, TokOfHash, SM, LO, true)) {
    auto ItIf = DpctGlobalInfo::getEndifLocationOfIfdef().find(
      getHashStrFromLoc(TokOfHash.getEndLoc()));
    while (ItIf != DpctGlobalInfo::getEndifLocationOfIfdef().end()) {
      InsertLocation = Lexer::getLocForEndOfToken(ItIf->second, 0, SM, LO);
      InsertLocation = Lexer::GetBeginningOfToken(
        Lexer::findNextToken(InsertLocation, SM, LO)->getLocation(), SM, LO);
      if (Lexer::getRawToken(InsertLocation, TokOfHash, SM, LO, true))
        break;
      ItIf = DpctGlobalInfo::getEndifLocationOfIfdef().find(
        getHashStrFromLoc(TokOfHash.getEndLoc()));
    }
  }

  // Skip whitespace, e.g. void foo(        void) {}
  //                                        |
  //                                      need get here
  if (!Lexer::getRawToken(InsertLocation, TokOfHash, SM, LO, true)) {
    InsertLocation = TokOfHash.getLocation();
  }

  ReplaceOffset = SM.getFileOffset(InsertLocation);
  if (FTL.getNumParams() == 0) {
    Token Tok;
    if (!Lexer::getRawToken(InsertLocation, Tok, SM, LO, true) &&
      Tok.is(tok::raw_identifier) && Tok.getRawIdentifier() == "void") {
      ReplaceLength = Tok.getLength();
    }
  }
}

void DeviceFunctionDecl::setFuncInfo(std::shared_ptr<DeviceFunctionInfo> Info) {
  if (FuncInfo.get() == Info.get())
    return;
  FuncInfo = Info;
  if (IsDefFilePathNeeded)
    FuncInfo->setDefinitionFilePath(FilePath);
}

void DeviceFunctionDecl::LinkDecl(const FunctionDecl *FD, DeclList &List,
                                  std::shared_ptr<DeviceFunctionInfo> &Info) {
  if (!DpctGlobalInfo::isInRoot(FD->getBeginLoc()))
    return;
  if (!FD->hasAttr<CUDADeviceAttr>() && !FD->hasAttr<CUDAGlobalAttr>())
    return;

  /// Ignore explicit instantiation definition, as the decl in AST has wrong
  /// location info. And it is processed in
  /// DPCTConsumer::HandleCXXExplicitFunctionInstantiation
  if (FD->getTemplateSpecializationKind() ==
      TSK_ExplicitInstantiationDefinition)
    return;

  if (FD->isImplicit() ||
      FD->getTemplateSpecializationKind() == TSK_ImplicitInstantiation) {
    auto &FuncInfo = getFuncInfo(FD);
    if (Info) {
      if (FuncInfo)
        Info->merge(FuncInfo);
      FuncInfo = Info;
    } else if (FuncInfo) {
      Info = FuncInfo;
    }
    return;
  }
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
  case Decl::CXXMethod:
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
    DpctDiags() << "[DeviceFunctionDecl::LinkDecl] Unexpected decl type: "
      << ND->getDeclKindName() << "\n";
    return;
  }
}

MemVarInfo::MemVarInfo(unsigned Offset, const std::string &FilePath,
    const VarDecl *Var)
  : VarInfo(Offset, FilePath, Var), Attr(getAddressAttr(Var)),
  Scope(isLexicallyInLocalScope(Var)
    ? (Var->getStorageClass() == SC_Extern ? Extern : Local)
    : Global),
  PointerAsArray(false) {
  if (getType()->isPointer() && getScope() == Global) {
    Attr = Device;
    getType()->adjustAsMemType();
    PointerAsArray = true;
  }
  if (Var->hasInit())
    setInitList(Var->getInit());
  if (getType()->getDimension() == 0 && Attr == Constant) {
    AccMode = Value;
  } else if (getType()->getDimension() <= 1) {
    AccMode = Pointer;
  } else {
    AccMode = Accessor;
  }
  if (Var->getStorageClass() == SC_Static) {
    IsStatic = true;
  }

  if (auto Func = Var->getParentFunctionOrMethod()) {
    if (DeclOfVarType = Var->getType()->getAsCXXRecordDecl()) {
      auto F = DeclOfVarType->getParentFunctionOrMethod();
      if (F && (F == Func)) {
        IsTypeDeclaredLocal = true;

        auto getParentDeclStmt = [&](const Decl *D) -> const DeclStmt * {
          auto P = getParentStmt(D);
          if (!P)
            return nullptr;
          auto DS = dyn_cast<DeclStmt>(P);
          if (!DS)
            return nullptr;
          return DS;
        };

        auto DS1 = getParentDeclStmt(Var);
        auto DS2 = getParentDeclStmt(DeclOfVarType);
        if (DS1 && DS2 && DS1 == DS2) {
          IsAnonymousType = true;
          DeclStmtOfVarType = DS2;
          auto Iter = AnonymousTypeDeclStmtMap.find(DS2);
          if (Iter != AnonymousTypeDeclStmtMap.end()) {
            LocalTypeName = "type_ct" + std::to_string(Iter->second);
          }
          else {
            LocalTypeName =
              "type_ct" + std::to_string(AnonymousTypeDeclStmtMap.size() + 1);
            AnonymousTypeDeclStmtMap.insert(
              std::make_pair(DS2, AnonymousTypeDeclStmtMap.size() + 1));
          }
        }
        else if (DS2) {
          DeclStmtOfVarType = DS2;
        }
      }
    }
  }

  newConstVarInit(Var);
}

std::shared_ptr<DeviceFunctionInfo> &
DeviceFunctionDecl::getFuncInfo(const FunctionDecl *FD) {
  DpctNameGenerator G;
  return FuncInfoMap[G.getName(FD)];
}

std::shared_ptr<MemVarInfo> MemVarInfo::buildMemVarInfo(const VarDecl *Var) {
  if (auto Func =
          DpctGlobalInfo::findAncestor<FunctionDecl>(Var)) {
    if (Func->getTemplateSpecializationKind() ==
            TSK_ExplicitInstantiationDefinition ||
        Func->getTemplateSpecializationKind() == TSK_ImplicitInstantiation)
      return std::shared_ptr<MemVarInfo>();
    auto LocInfo = DpctGlobalInfo::getLocInfo(Var);
    auto VI = std::make_shared<MemVarInfo>(LocInfo.second, LocInfo.first, Var);
    DeviceFunctionDecl::LinkRedecls(Func)->addVar(VI);
    return VI;
  }

  return DpctGlobalInfo::getInstance().insertMemVarInfo(Var);
}

MemVarInfo::VarAttrKind MemVarInfo::getAddressAttr(const AttrVec &Attrs) {
  for (auto VarAttr : Attrs) {
    auto Kind = VarAttr->getKind();
    if (Kind == attr::HIPManaged)
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
    static std::string DeviceMemory = "dpct::global_memory";
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
    if (isShared() && getType()->getDimension() > 1) {
      // For case like:
      // extern __shared__ int shad_mem[][2][3];
      // int p = shad_mem[0][0][2];
      // will be migrated to:
      // auto shad_mem = (int(*)[2][3])dpct_local;
      std::string Dimension;
      size_t Index = 0;
      for (auto &Entry : getType()->getRange()) {
        Index++;
        if (Index == 1)
          continue;
        Dimension = Dimension + "[" + Entry.getSize() + "]";
      }
      return buildString("auto ", getName(), " = (", getType()->getBaseName(),
                         "(*)", Dimension, ")", ExternVariableName, ";");
    }

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

void MemVarInfo::appendAccessorOrPointerDecl(const std::string &ExternMemSize,
                                             StmtList &AccList, StmtList &PtrList){
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  if (isShared()) {
    auto Dimension = getType()->getDimension();
    OS << MapNames::getClNamespace() + "::accessor<"
       << getAccessorDataType() << ", " << Dimension
       << ", " + MapNames::getClNamespace() + "::access::mode::read_write, " +
          MapNames::getClNamespace() + "::access::target::local> "
       << getAccessorName() << "(";
    if (Dimension > 1) {
      OS << getRangeName() << ", ";
    } else if (Dimension == 1) {
      OS << getRangeClass()
         << getType()->getRangeArgument(ExternMemSize, false) << ", ";
    }
    OS << "cgh);";
    StmtWithWarning AccDecl(OS.str());
    if (Dimension > 3) {
      if (DiagnosticsUtils::report(getFilePath(), getOffset(),
                                   Diagnostics::EXCEED_MAX_DIMENSION, false,
                                   false)) {
        AccDecl.Warning = DiagnosticsUtils::getWarningTextAndUpdateUniqueID(
            Diagnostics::EXCEED_MAX_DIMENSION);
      }
    }
    AccList.emplace_back(std::move(AccDecl));
  } else if (DpctGlobalInfo::getUsmLevel() == UsmLevel::restricted &&
             AccMode != Accessor) {
    PtrList.emplace_back(buildString("auto ", getPtrName(), " = ",
                                     getConstVarName(), ".get_ptr();"));
  } else {
    AccList.emplace_back(buildString("auto ", getAccessorName(), " = ",
                                     getConstVarName(), ".get_access(cgh);"));
  }
}

template <class T>
void removeDuplicateVar(GlobalMap<T> &VarMap,
                        std::unordered_set<std::string> &VarNames) {
  auto Itr = VarMap.begin();
  while (Itr != VarMap.end()) {
    if (VarNames.find(Itr->second->getName()) == VarNames.end()) {
      VarNames.insert(Itr->second->getName());
      ++Itr;
    } else {
      Itr = VarMap.erase(Itr);
    }
  }
}
void MemVarMap::removeDuplicateVar() {
  std::unordered_set<std::string> VarNames{DpctGlobalInfo::getItemName(),
                                           DpctGlobalInfo::getStreamName()};
  dpct::removeDuplicateVar(GlobalVarMap, VarNames);
  dpct::removeDuplicateVar(LocalVarMap, VarNames);
  dpct::removeDuplicateVar(ExternVarMap, VarNames);
  dpct::removeDuplicateVar(TextureMap, VarNames);
}

std::string MemVarMap::getExtraCallArguments(bool HasPreParam, bool HasPostParam) const {
  return getArgumentsOrParameters<CallArgument>(HasPreParam, HasPostParam);
}
std::string MemVarMap::getExtraDeclParam(bool HasPreParam, bool HasPostParam,
                                         FormatInfo FormatInformation) const {
  return getArgumentsOrParameters<DeclParameter>(HasPreParam, HasPostParam,
                                                 FormatInformation);
}
std::string MemVarMap::getKernelArguments(bool HasPreParam, bool HasPostParam) const {
  return getArgumentsOrParameters<KernelArgument>(HasPreParam, HasPostParam);
}

CtTypeInfo::CtTypeInfo(const TypeLoc &TL, bool NeedSizeFold)
    : PointerLevel(0), IsTemplate(false) {
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
    ++PointerLevel;
    return setTypeInfo(TYPELOC_CAST(PointerTypeLoc).getPointeeLoc());
  case TypeLoc::LValueReference:
  case TypeLoc::RValueReference:
    IsReference = true;
    return setTypeInfo(TYPELOC_CAST(ReferenceTypeLoc).getPointeeLoc());
  default:
    break;
  }
  setName(TL);
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

void CtTypeInfo::setName(const TypeLoc &TL) {
  ExprAnalysis EA;
  EA.analyze(TL);
  TDSI = EA.getTemplateDependentStringInfo();

  IsTemplate = TL.getTypePtr()->isDependentType();
  updateName();
}

void CtTypeInfo::updateName(){

  BaseNameWithoutQualifiers = TDSI->getSourceString();

  if (isPointer()) {
    BaseNameWithoutQualifiers += ' ';
    BaseNameWithoutQualifiers.append(PointerLevel, '*');
  }

  if (BaseName.empty())
    BaseName = BaseNameWithoutQualifiers;
  else
    BaseName = buildString(BaseName, isPointer() ? "" : " ",
                           BaseNameWithoutQualifiers);
}

std::shared_ptr<CtTypeInfo> CtTypeInfo::applyTemplateArguments(
    const std::vector<TemplateArgumentInfo> &TA) {
  auto NewType = std::make_shared<CtTypeInfo>(*this);
  if (TDSI)
    NewType->TDSI = TDSI->applyTemplateArguments(TA);
  for (auto &R : NewType->Range)
    R.setTemplateList(TA);
  NewType->BaseName.clear();
  NewType->updateName();
  return NewType;
}

void SizeInfo::setTemplateList(
    const std::vector<TemplateArgumentInfo> &TemplateList) {
  if (TDSI)
    TDSI = TDSI->applyTemplateArguments(TemplateList);
}

// In the type migration rule, only location has been recorded. So in this
// function, other info from generator creation API is added.
void RandomEngineInfo::updateEngineType() {
  auto FileInfo = DpctGlobalInfo::getInstance().insertFile(DeclFilePath);
  auto &M = FileInfo->getHostRandomEngineTypeMap();

  auto Iter = M.find(DeclaratorDeclTypeBeginOffset);
  if (Iter != M.end()) {
    Iter->second.EngineType = TypeReplacement;
    Iter->second.HasValue = true;
    Iter->second.IsUnsupportEngine = IsUnsupportEngine;
  } else {
    M.insert(
        std::make_pair(DeclaratorDeclTypeBeginOffset,
                       HostRandomEngineTypeInfo(TypeLength, TypeReplacement,
                                                IsUnsupportEngine)));
  }
}

void RandomEngineInfo::buildInfo() {
  if (IsUnsupportEngine)
    return;

  if (TypeReplacement.empty()) {
    TypeReplacement= "dpct_placeholder/*Fix the engine type manually*/";
    for (unsigned int i = 0; i < CreateAPINum; ++i)
      DiagnosticsUtils::report(CreateCallFilePath[i], CreateAPIBegin[i],
                               Diagnostics::UNDEDUCED_TYPE, true, false, "RNG engine");
  }

  std::string Repl = GeneratorName + " = new " + TypeReplacement + "(" +
                     QueueStr + ", " + (IsQuasiEngine ? DimExpr : SeedExpr) +
                     ")";
  if (IsAssigned) {
    Repl = "(" + Repl + ", 0)";
  }

  for (unsigned int i = 0; i < CreateAPINum; ++i)
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(CreateCallFilePath[i],
                                         CreateAPIBegin[i], CreateAPILength[i],
                                         Repl, nullptr));
}

void DeviceRandomStateTypeInfo::buildInfo(std::string FilePath,
                                          unsigned int Offset) {
  if (DpctGlobalInfo::getDeviceRNGReturnNumSet().size() == 1) {
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(
            FilePath, Offset, Length,
            GeneratorType + "<" +
                std::to_string(
                    *DpctGlobalInfo::getDeviceRNGReturnNumSet().begin()) +
                ">",
            nullptr));
  } else {
    DiagnosticsUtils::report(FilePath, Offset, Diagnostics::UNDEDUCED_TYPE,
                             true, false, "RNG engine");
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(
            FilePath, Offset, Length,
            GeneratorType + "<dpct_placeholder/*Fix the vec_size manually*/>",
            nullptr));
  }
}

void DeviceRandomInitAPIInfo::buildInfo(std::string FilePath,
                                        unsigned int Offset) {
  std::string VecSizeStr;
  bool IsOneNumber = false;
  if (DpctGlobalInfo::getDeviceRNGReturnNumSet().size() == 1) {
    int VecSize = *DpctGlobalInfo::getDeviceRNGReturnNumSet().begin();
    if (VecSize == 1)
      IsOneNumber = true;
    VecSizeStr = std::to_string(VecSize);
  } else {
    DiagnosticsUtils::report(FilePath, Offset, Diagnostics::UNDEDUCED_TYPE,
                             true, false, "RNG engine");
    VecSizeStr = "dpct_placeholder/*Fix the vec_size manually*/";
  }

  std::string FirstOffsetArg, SecondOffsetArg;
  if (IsRNGOffsetLiteral) {
    FirstOffsetArg = RNGOffset + (IsOneNumber ? "" : " * " + VecSizeStr);
  } else {
    FirstOffsetArg = "static_cast<std::uint64_t>(" + RNGOffset +
                     (IsOneNumber ? "" : " * " + VecSizeStr) + ")";
  }
  if (IsRNGSubseqLiteral) {
    SecondOffsetArg = RNGSubseq + " * 8";
  } else {
    SecondOffsetArg = "static_cast<std::uint64_t>(" + RNGSubseq + " * 8)";
  }

  std::string ReplStr = RNGStateName + " = " + GeneratorType + "<" +
                        VecSizeStr + ">(" + RNGSeed + ", {" + FirstOffsetArg +
                        ", " + SecondOffsetArg + "})";

  DpctGlobalInfo::getInstance().addReplacement(std::make_shared<ExtReplacement>(
      FilePath, Offset, Length, ReplStr, nullptr));
}

void DeviceRandomGenerateAPIInfo::buildInfo(std::string FilePath,
                                            unsigned int Offset) {
  std::string ReplStr =
      "oneapi::mkl::rng::device::generate(" + DistrName + ", " + RNGStateName + ")";
  DpctGlobalInfo::getInstance().addReplacement(std::make_shared<ExtReplacement>(
      FilePath, Offset, Length, ReplStr, nullptr));
}

void DeviceRandomDistrInfo::buildInfo(std::string FilePath,
                                      unsigned int Offset) {
  std::string InsertStr = DistrType + "<" + ValueType + "> " + DistrName + ";" +
                          getNL() + IndentStr;
  DpctGlobalInfo::getInstance().addReplacement(std::make_shared<ExtReplacement>(
      FilePath, Offset, 0, InsertStr, nullptr));
}

void HostRandomEngineTypeInfo::buildInfo(std::string FilePath,
                                         unsigned int Offset) {
  // The warning of unsupported engine type is emitted in the genreator creation
  // API migration rule. So do not emit undeduced type warning again.
  if (IsUnsupportEngine)
    return;

  if (HasValue && !EngineType.empty()) {
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(FilePath, Offset, Length,
                                         EngineType + "*", nullptr));
  } else if (DpctGlobalInfo::getHostRNGEngineTypeSet().size() == 1) {
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(
            FilePath, Offset, Length,
            *DpctGlobalInfo::getHostRNGEngineTypeSet().begin() + "*", nullptr));
  } else {
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(
            FilePath, Offset, Length,
            "dpct_placeholder/*Fix the engine type manually*/*", nullptr));
    DiagnosticsUtils::report(FilePath, Offset, Diagnostics::UNDEDUCED_TYPE,
                             true, false, "RNG engine");
  }
}

void HostRandomDistrInfo::buildInfo(std::string FilePath, unsigned int Offset,
                                    std::string DistrType,
                                    std::string ValueType,
                                    std::string DistrArg) {
  std::string InsertStr;
  if (DistrArg.empty())
    InsertStr = DistrType + "<" + ValueType + "> " + DistrName + ";" + getNL() +
                IndentStr;
  else
    InsertStr = DistrType + "<" + ValueType + "> " + DistrName + "(" +
                DistrArg + ");" + getNL() + IndentStr;
  DpctGlobalInfo::getInstance().addReplacement(std::make_shared<ExtReplacement>(
      FilePath, Offset, 0, InsertStr, nullptr));
}

void EventSyncTypeInfo::buildInfo(std::string FilePath, unsigned int Offset) {
  if (NeedReport)
    DiagnosticsUtils::report(FilePath, Offset,
                             Diagnostics::NOERROR_RETURN_COMMA_OP, true, false);

  if (IsAssigned && ReplText.empty()) {
    ReplText = "0";
  }

  DpctGlobalInfo::getInstance().addReplacement(std::make_shared<ExtReplacement>(
      FilePath, Offset, Length, ReplText, nullptr));
}

void BuiltinVarInfo::buildInfo(std::string FilePath, unsigned int Offset,
                                    unsigned int ID) {
  std::string R = Repl + std::to_string(ID) + ")";
  DpctGlobalInfo::getInstance().addReplacement(std::make_shared<ExtReplacement>(
      FilePath, Offset, Len, R, nullptr));
}

bool isInRoot(SourceLocation SL) { return DpctGlobalInfo::isInRoot(SL); }
} // namespace dpct
} // namespace clang
