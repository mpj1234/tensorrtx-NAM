#pragma once
#ifndef ATTENTION
#define ATTENTION

#include "NvInfer.h"
#include "common.hpp"

ILayer* addNAMChannel(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps)
{
    float* gamma = (float*)weightMap[lname + ".weight"].values;
    int len = weightMap[lname + ".running_var"].count;

    // bn_x = self.bn2(x)
    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, input, lname, 1e-5);

    // 求各个通道gamma的值
    // 求权重总值
    // sum = torch.sum(self.bn2.weight.data.abs())
    float sum = 0.0;
    for (int i = 0; i < len; i++)
    {
        sum += abs(gamma[i]);
    }

    // 求每个权重占比
    // self.bn2.weight.data.abs() / sum
    float* weight = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++)
    {
        weight[i] = abs(gamma[i]) / sum;
    }


    Weights scale{ DataType::kFLOAT, weight, len };

    float* shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++)
    {
        shval[i] = 0.0;
    }
    Weights shift{ DataType::kFLOAT, shval, len };

    float* pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++)
    {
        pval[i] = 1.0;
    }
    Weights power{ DataType::kFLOAT, pval, len };

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;  // 0.0
    weightMap[lname + ".power"] = power;  // 1.0
    // x = torch.mul(weight_bn, bn_x)
    IScaleLayer* scale_1 = network->addScale(*bn2->getOutput(0), ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);

    // sigmoid = torch.sigmoid(x)
    auto sigmoid = network->addActivation(*scale_1->getOutput(0), ActivationType::kSIGMOID);
    assert(sigmoid);

    // sigmoid * input
    ILayer* nam = network->addElementWise(input, *sigmoid->getOutput(0), ElementWiseOperation::kPROD);
    assert(nam);

    return nam;
}

#endif // !ATTENTION
