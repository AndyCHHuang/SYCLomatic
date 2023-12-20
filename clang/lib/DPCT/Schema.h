//===--------------------------- Schema.h ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef DPCT_SCHEMA_H
#define DPCT_SCHEMA_H

#include "AnalysisInfo.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclCXX.h>
#include <clang/Basic/LLVM.h>
#include <iostream>
#include <string>
#include <vector>

namespace clang {
namespace dpct {

enum ValType : int {
  ScalarValue = 0,
  ArrayValue,
  Pointer,
  PointerOfPointer,
};

inline std::ostream &operator<<(std::ostream &strm, ValType tt) {
  const std::string VTStr[] = {"ScalarValue", "ArrayValue", "Pointer",
                               "PointerOfPointer"};
  return strm << VTStr[tt];
}

inline std::string getValTypeStr(ValType tt) {
  const std::string VTStr[] = {"ScalarValue", "ArrayValue", "Pointer",
                               "PointerOfPointer"};
  return VTStr[tt];
}

struct FieldSchema {
  std::string FieldName;
  ValType ValTypeOfField;
  std::string FieldType;
  bool IsBasicType = false;
  int64_t ValSize = 0;
  int64_t Offset = 0;
  std::string Location = "None";
};

struct TypeSchema {
  std::string TypeName;
  int FieldNum = 0;
  int64_t TypeSize = 0;
  bool IsVirtual = false;
  std::string FileName;
  std::vector<FieldSchema> Members;
};

struct VarSchema {
  std::string VarName;
  ValType ValTypeOfVar;
  std::string FileName;
  std::string VarType;
  bool IsBasicType = false;
  int64_t VarSize = 0;
  std::string Location = "None";
};

std::string getFilePathFromDecl(const Decl *D, const SourceManager &SM);

ValType getValType(const clang::QualType &QT);

FieldSchema constructFieldSchema(const clang::FieldDecl *FD,
                                 std::string ClassTypeName);

inline FieldSchema constructFieldSchema(const clang::FieldDecl *FD);

void DFSBaseClass(clang::CXXRecordDecl *RD, TypeSchema &TS);

TypeSchema constructTypeSchema(const clang::RecordType *RT);

TypeSchema registerTypeSchema(const clang::QualType QT);

VarSchema constructVarSchema(const clang::DeclRefExpr *DRE);

extern std::map<std::string, TypeSchema> TypeSchemaMap;

llvm::json::Array
serializeSchemaToJsonArray(const std::map<std::string, TypeSchema> &TSMap);

llvm::json::Array serializeSchemaToJsonArray(const std::vector<TypeSchema> &TSVec);

llvm::json::Object serializeSchemaToJson(const TypeSchema &TS);

void serializeJsonArrayToFile(llvm::json::Array &&Arr,
                              const std::string &FilePath);

std::vector<TypeSchema> getRelatedTypeSchema(const clang::QualType QT);

} // namespace dpct
} // namespace clang

#endif
