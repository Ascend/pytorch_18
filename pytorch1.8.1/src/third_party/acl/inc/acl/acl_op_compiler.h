/**
* @file acl_op_compiler.h
*
* Copyright (C) Huawei Technologies Co., Ltd. 2019-2020. All Rights Reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef INC_EXTERNAL_ACL_ACL_OP_COMPILER_H_
#define INC_EXTERNAL_ACL_ACL_OP_COMPILER_H_

#include "acl_base.h"
#include "acl_op.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum aclCompileType {
    ACL_COMPILE_SYS,
    ACL_COMPILE_UNREGISTERED
} aclopCompileType;

typedef enum aclOpCompileFlag{
    ACL_OPCOMPILE_DEFAULT = 0,
    ACL_OPCOMPILE_FUZZ = 1,
}aclOpCompileFlag;

/**
 * @ingroup AscendCL
 * @brief compile op
 *
 * @param opType [IN]           op type
 * @param numInputs [IN]        number of inputs
 * @param inputDesc [IN]        pointer to array of input tensor descriptions
 * @param numOutputs [IN]       number of outputs
 * @param outputDesc [IN]       pointer to array of output tensor descriptions
 * @param attr [IN]           pointer to instance of aclopAttr.
 *                              may pass nullptr if the op has no attribute
 * @param engineType [IN]       engine type
 * @param compileFlag [IN]      compile flag
 * @param opPath [IN]           path of op
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopCompile(const char *opType,
                                          int numInputs,
                                          const aclTensorDesc *const inputDesc[],
                                          int numOutputs,
                                          const aclTensorDesc *const outputDesc[],
                                          const aclopAttr *attr,
                                          aclopEngineType engineType,
                                          aclopCompileType compileFlag,
                                          const char *opPath);

/**
 * @ingroup AscendCL
 * @brief compile and execute op
 *
 * @param opType [IN]           op type
 * @param numInputs [IN]        number of inputs
 * @param inputDesc [IN]        pointer to array of input tensor descriptions
 * @param inputs [IN]           pointer to array of input buffers
 * @param numOutputs [IN]       number of outputs
 * @param outputDesc [IN]       pointer to array of output tensor descriptions
 * @param outputs [IN]          pointer to array of outputs buffers
 * @param attr [IN]             pointer to instance of aclopAttr.
 *                              may pass nullptr if the op has no attribute
 * @param engineType [IN]       engine type
 * @param compileFlag [IN]      compile flag
 * @param opPath [IN]           path of op
 * @param stream [IN]           stream handle
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopCompileAndExecute(const char *opType,
    int numInputs, const aclTensorDesc *const inputDesc[], const aclDataBuffer *const inputs[],
    int numOutputs, const aclTensorDesc *const outputDesc[], aclDataBuffer *const outputs[],
    const aclopAttr *attr, aclopEngineType engineType, aclopCompileType compileFlag,
    const char *opPath, aclrtStream stream);


/**
 * @ingroup AscendCL
 * @brief an interface set compile flag
 *
 * @param flag [IN]     flag: ACL_OPCOMPILE_DEFAULT represent static compile while ACL_OPCOMPILE_FUZZ represent dynamic compile
 *        
 * @retval ACL_ERROR_NONE The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopSetCompileFlag(aclOpCompileFlag flag);

#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_ACL_OP_COMPILER_H_