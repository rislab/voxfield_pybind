#pragma once

#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <any>

namespace ros_emulator {

    // Emulates key-value param lookup of ros::NodeHandle without ros dependency
    class NodeHandle {

        // Function to parse the yaml
        bool LoadConfig(const std::string& yamlFilePath)
        {
            std::ifstream yamlFile(yamlFilePath);
            YAML::Node yamlNode = YAML::Load(yamlFile);

            if (yamlNode.IsMap())
            {
                for (const auto& entry : yamlNode)
                {
                    params_[entry.first.as<std::string>()] = entry.second;
                }
                return true;
            }
            return false;
        }

        void PrintParams()
        {
            for (const auto& entry : params_)
            {
                std::cout << "Key: " << entry.first << ", Value: " entry.second << std::endl;
            }
        }

        // Assign value from parameter server, with default.
        // https://github.com/strawlab/ros_comm/blob/master/clients/cpp/roscpp/include/ros/node_handle.h#L1562
        template<typename T>
        void param(const std::string& param_name, T& param_val, const T& default_val) const {
            if (params_.count(param_name) > 0) {
                param_val = param_.at(param_name);
                return;
            }
            param_val = default_val;
        }

        protected:
            std::map<std::string, std::any> params_;

    }
}