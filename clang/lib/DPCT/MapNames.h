//===--- MapNames.h -----------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2019 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef DPCT_MAPNAMES_H
#define DPCT_MAPNAMES_H

#include "Utility.h"
#include <map>
#include <set>

const std::string StringLiteralUnsupported{"UNSUPPORTED"};

/// Record mapping between names
class MapNames {
public:
  struct SOLVERFuncReplInfo {
    static SOLVERFuncReplInfo migrateBuffer(std::vector<int> bi,
                                            std::vector<std::string> bt,
                                            std::string s) {
      MapNames::SOLVERFuncReplInfo repl;
      repl.BufferIndexInfo = bi;
      repl.BufferTypeInfo = bt;
      repl.ReplName = s;
      return repl;
    };

    static SOLVERFuncReplInfo
    migrateBufferAndRedundant(std::vector<int> bi, std::vector<std::string> bt,
                              std::vector<int> ri, std::string s) {
      MapNames::SOLVERFuncReplInfo repl;
      repl.BufferIndexInfo = bi;
      repl.BufferTypeInfo = bt;
      repl.RedundantIndexInfo = ri;
      repl.ReplName = s;
      return repl;
    };

    static SOLVERFuncReplInfo
      migrateDeviceAndRedundant(bool q2d, std::vector<int> ri, std::string s) {
      MapNames::SOLVERFuncReplInfo repl;
      repl.ToDevice = true;
      repl.RedundantIndexInfo = ri;
      repl.ReplName = s;
      return repl;
    };

    static SOLVERFuncReplInfo migrateDeviceAndCopy(bool q2d,
                                                   std::vector<int> cfi,
                                                   std::vector<int> cti,
                                                   std::string s) {
      MapNames::SOLVERFuncReplInfo repl;
      repl.ToDevice = q2d;
      repl.CopyFrom = cfi;
      repl.CopyTo = cti;
      repl.ReplName = s;
      return repl;
    };

    static SOLVERFuncReplInfo
    migrateBufferAndMissed(std::vector<int> bi, std::vector<std::string> bt,
                           std::vector<int> mafl, std::vector<int> mai,
                           std::vector<bool> mab, std::vector<std::string> mat,
                           std::vector<std::string> man, std::string s) {
      MapNames::SOLVERFuncReplInfo repl;
      repl.BufferIndexInfo = bi;
      repl.BufferTypeInfo = bt;
      repl.MissedArgumentFinalLocation = mafl;
      repl.MissedArgumentInsertBefore = mai;
      repl.MissedArgumentIsBuffer = mab;
      repl.MissedArgumentType = mat;
      repl.MissedArgumentName = man;
      repl.ReplName = s;
      return repl;
    };

    static SOLVERFuncReplInfo
      migrateDeviceCopyAndMissed(bool q2d, std::vector<int> cfi, std::vector<int> cti,
        std::vector<int> mafl, std::vector<int> mai,
        std::vector<bool> mab, std::vector<std::string> mat,
        std::vector<std::string> man, std::string s) {
      MapNames::SOLVERFuncReplInfo repl;
      repl.ToDevice = q2d;
      repl.CopyFrom = cfi;
      repl.CopyTo = cti;
      repl.MissedArgumentFinalLocation = mafl;
      repl.MissedArgumentInsertBefore = mai;
      repl.MissedArgumentIsBuffer = mab;
      repl.MissedArgumentType = mat;
      repl.MissedArgumentName = man;
      repl.ReplName = s;
      return repl;
    };

    static SOLVERFuncReplInfo
      migrateDeviceRedundantAndMissed(bool q2d,
        std::vector<int> ri, std::vector<int> mafl, std::vector<int> mai,
        std::vector<bool> mab, std::vector<std::string> mat,
        std::vector<std::string> man, std::string s) {
      MapNames::SOLVERFuncReplInfo repl;
      repl.ToDevice = q2d;
      repl.RedundantIndexInfo = ri;
      repl.MissedArgumentFinalLocation = mafl;
      repl.MissedArgumentInsertBefore = mai;
      repl.MissedArgumentIsBuffer = mab;
      repl.MissedArgumentType = mat;
      repl.MissedArgumentName = man;
      repl.ReplName = s;
      return repl;
    };

    static SOLVERFuncReplInfo
      migrateBufferAndCast(std::vector<int> bi, std::vector<std::string> bt,
        std::vector<int> ci, std::vector<std::string> ct, std::string s) {
      MapNames::SOLVERFuncReplInfo repl;
      repl.BufferIndexInfo = bi;
      repl.BufferTypeInfo = bt;
      repl.CastIndexInfo = ci;
      repl.CastTypeInfo = ct;
      repl.ReplName = s;
      return repl;
    };

    static SOLVERFuncReplInfo migrateBufferRedundantAndCast(
        std::vector<int> bi, std::vector<std::string> bt, std::vector<int> ri,
        std::vector<int> ci, std::vector<std::string> ct, std::string s) {
      MapNames::SOLVERFuncReplInfo repl;
      repl.BufferIndexInfo = bi;
      repl.BufferTypeInfo = bt;
      repl.RedundantIndexInfo = ri;
      repl.CastIndexInfo = ci;
      repl.CastTypeInfo = ct;
      repl.ReplName = s;
      return repl;
    };

    static SOLVERFuncReplInfo migrateBufferMissedAndCast(
        std::vector<int> bi, std::vector<std::string> bt, std::vector<int> mafl,
        std::vector<int> mai, std::vector<bool> mab,
        std::vector<std::string> mat, std::vector<std::string> man,
        std::vector<int> ci, std::vector<std::string> ct, std::string s) {
      MapNames::SOLVERFuncReplInfo repl;
      repl.BufferIndexInfo = bi;
      repl.BufferTypeInfo = bt;
      repl.MissedArgumentFinalLocation = mafl;
      repl.MissedArgumentInsertBefore = mai;
      repl.MissedArgumentIsBuffer = mab;
      repl.MissedArgumentType = mat;
      repl.MissedArgumentName = man;
      repl.CastIndexInfo = ci;
      repl.CastTypeInfo = ct;
      repl.ReplName = s;
      return repl;
    };

    static SOLVERFuncReplInfo
    migrateDeviceRedundantAndCast(bool q2d, std::vector<int> ri,
                                  std::vector<int> ci,
                                  std::vector<std::string> ct, std::string s) {
      MapNames::SOLVERFuncReplInfo repl;
      repl.ToDevice = q2d;
      repl.RedundantIndexInfo = ri;
      repl.CastIndexInfo = ci;
      repl.CastTypeInfo = ct;
      repl.ReplName = s;
      return repl;
    };

    std::vector<int> BufferIndexInfo;
    std::vector<std::string> BufferTypeInfo;

    // will be replaced by empty string""
    std::vector<int> RedundantIndexInfo;

    std::vector<int> CastIndexInfo;
    std::vector<std::string> CastTypeInfo;

    std::vector<int> MissedArgumentFinalLocation;
    std::vector<int> MissedArgumentInsertBefore; //index of original argument
    std::vector<bool> MissedArgumentIsBuffer;
    std::vector<std::string> MissedArgumentType;
    std::vector<std::string> MissedArgumentName;

    std::vector<int> CopyFrom;
    std::vector<int> CopyTo;
    bool ToDevice = false;
    std::string ReplName;
  };

  struct BLASFuncReplInfo {
    std::vector<int> BufferIndexInfo;
    std::vector<int> PointerIndexInfo;
    std::vector<std::string> BufferTypeInfo;
    std::vector<int> OperationIndexInfo;
    int FillModeIndexInfo;
    int SideModeIndexInfo;
    int DiagTypeIndexInfo;
    std::string ReplName;
  };

  struct BLASFuncComplexReplInfo {
    std::vector<int> BufferIndexInfo;
    std::vector<int> PointerIndexInfo;
    std::vector<std::string> BufferTypeInfo;
    std::vector<std::string> PointerTypeInfo;
    std::vector<int> OperationIndexInfo;
    int FillModeIndexInfo;
    int SideModeIndexInfo;
    int DiagTypeIndexInfo;
    std::string ReplName;
  };

  struct ThrustFuncReplInfo {
    std::string ReplName;
    std::string ExtraParam;
  };

  using MapTy = std::map<std::string, std::string>;
  using SetTy = std::set<std::string>;
  using ThrustMapTy = std::map<std::string, ThrustFuncReplInfo>;

  static const SetTy SupportedVectorTypes;
  static const MapTy RemovedAPIWarningMessage;
  static const MapTy TypeNamesMap;
  static const MapTy Dim3MemberNamesMap;
  static const MapTy MacrosMap;
  static const MapTy BLASEnumsMap;
  static const std::map<std::string, MapNames::BLASFuncReplInfo>
      BLASFuncReplInfoMap;
  static const std::map<std::string, MapNames::BLASFuncComplexReplInfo>
      BLASFuncComplexReplInfoMap;
  static const SetTy ThrustFileExcludeSet;
  static const ThrustMapTy ThrustFuncNamesMap;
  static const std::map<std::string, MapNames::BLASFuncReplInfo>
      BLASFuncWrapperReplInfoMap;

  static const std::map<std::string, MapNames::BLASFuncComplexReplInfo>
      LegacyBLASFuncReplInfoMap;

  static const MapTy SOLVEREnumsMap;
  static const std::map<std::string, MapNames::SOLVERFuncReplInfo>
      SOLVERFuncReplInfoMap;

  static const MapTy ITFName;

  inline static const std::string &findReplacedName(const MapTy &Map,
                                                    const std::string &Name) {
    static const std::string EmptyString;

    auto Itr = Map.find(Name);
    if (Itr == Map.end())
      return EmptyString;
    return Itr->second;
  }
  static bool replaceName(const MapTy &Map, std::string &Name) {
    auto &Result = findReplacedName(Map, Name);
    if (Result.empty())
      return false;
    Name = Result;
    return true;
  }
  static bool isInSet(const SetTy &Set, std::string &Name) {
    return Set.find(Name) != Set.end();
  }

  static const MapNames::MapTy MemberNamesMap;
};

class MigrationStatistics {
private:
  static std::map<std::string /*API Name*/, bool /*Is Migrated*/>
      MigrationTable;

public:
  static bool IsMigrated(const std::string &APIName);
  static std::vector<std::string> GetAllAPINames(void);
};

#endif
