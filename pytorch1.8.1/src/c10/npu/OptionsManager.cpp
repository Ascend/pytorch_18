
#include "c10/npu/OptionsManager.h"
#include <string>
#include "c10/npu/register/OptionRegister.h"

namespace c10 {
namespace npu {

using namespace std;

bool OptionsManager::CheckQueueEnable() {
  static int32_t queue_enable = -1;
  if (queue_enable == -1) {
    queue_enable = GetBoolTypeOption("TASK_QUEUE_ENABLE");
  }
  return (queue_enable == 1);
}

bool OptionsManager::CheckPTcopy_Enable() {
  static int32_t PTcopy__enable = -1;
  if (PTcopy__enable == -1) {
    PTcopy__enable = GetBoolTypeOption("PTCOPY_ENABLE");
  }
  return (PTcopy__enable == 1);
}

bool OptionsManager::CheckCombinedOptimizerEnable() {
  static int32_t combined_optimize = -1;
  if (combined_optimize == -1) {
    combined_optimize = GetBoolTypeOption("COMBINED_ENABLE");
  }
  return (combined_optimize == 1);
}

bool OptionsManager::CheckTriCombinedOptimizerEnable() {
  static int32_t tri_combined_optimize = -1;
  if (tri_combined_optimize == -1) {
    tri_combined_optimize = GetBoolTypeOption("TRI_COMBINED_ENABLE");
  }
  return (tri_combined_optimize == 1);
}


bool OptionsManager::CheckAclDumpDateEnable() {
  static int aclDumpDataEnable = -1;
  if (aclDumpDataEnable == -1) {
    aclDumpDataEnable = GetBoolTypeOption("ACL_DUMP_DATA");
  }
  return (aclDumpDataEnable == 1);
}

bool OptionsManager::CheckSwitchMMOutputEnable() {
  static int switchMMOutputEnable = -1;
  if (switchMMOutputEnable == -1) {
    switchMMOutputEnable = GetBoolTypeOption("SWITCH_MM_OUTPUT_ENABLE");
  }
  return (switchMMOutputEnable == 1);
}

int OptionsManager::GetBoolTypeOption(const char* env_str) {
  char* env_val = std::getenv(env_str);
  int64_t envFlag = (env_val != nullptr) ? strtol(env_val, nullptr, 10) : 0;
  return (envFlag != 0) ? 1 : 0;
}

bool OptionsManager::CheckUseNpuLogEnable() {
  static int useNpuLog = -1;
  if (useNpuLog == -1) {
    useNpuLog = GetBoolTypeOption("NPU_LOG_ENABLE");
  }

  return (useNpuLog == 1);
}

bool OptionsManager::CheckDynamicOptimizer(const char* op) {
  static int isGetOps = 0;
  static std::map<std::string, bool> op_map = {{"ADD", false}, {"MUL", false}};
  if (isGetOps == 0) {
    char* dynamicOptimizerEnv = std::getenv("DYNAMIC_OP");
    if (dynamicOptimizerEnv != nullptr) {
      std::string dynamicOptimizerEnvStr = dynamicOptimizerEnv;
      const std::string separator = "#";
      std::string::size_type pos1 = 0;
      std::string::size_type pos2 = dynamicOptimizerEnvStr.find(separator);
      std::string substr;
      while (pos2 != std::string::npos) {
        substr = dynamicOptimizerEnvStr.substr(pos1, pos2 - pos1);
        if (op_map.find(substr) != op_map.end()) {
          op_map[substr] = true;
        }
        pos1 = pos2 + separator.size();
        pos2 = dynamicOptimizerEnvStr.find(separator, pos1);
      }
      if (pos1 != dynamicOptimizerEnvStr.size()) {
        substr = dynamicOptimizerEnvStr.substr(pos1);
        if (op_map.find(substr) != op_map.end()) {
          op_map[substr] = true;
        }
      }
    }
    isGetOps = 1;
  }
  TORCH_CHECK(
      op_map.find(op) != op_map.end(), "This op is not currently optimized.");
  return op_map[op];
}

} // namespace npu
} // namespace c10