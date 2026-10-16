// Custom detection (bbox) parser (generated).
// Scaffold validates the output layer and exposes the raw tensor(s); the DECODE
// block (per-model) populates `objectList` with NvDsInferObjectDetectionInfo.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool ParseConfigDetPgie(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList);

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(ParseConfigDetPgie);

extern "C" bool ParseConfigDetPgie(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    if (outputLayersInfo.empty()) return false;
    auto const& layer0 = outputLayersInfo[0];

    const unsigned numClasses = detectionParams.numClassesConfigured;
    const float netW = static_cast<float>(networkInfo.width);
    const float netH = static_cast<float>(networkInfo.height);
    const float* data = static_cast<const float*>(layer0.buffer);

    static bool logged = false;
    if (!logged) {
        logged = true;
        fprintf(stderr, "[YOLO11s-det] first call: layer=%s numDims=%d "
                "networkInput=%.0fx%.0f numClasses=%u outputs=%zu\n",
                layer0.layerName ? layer0.layerName : "?",
                layer0.inferDims.numDims, netW, netH, numClasses,
                outputLayersInfo.size());
    }

    // ==== BEGIN GENERATED DECODE ====
    // output0 engine shape [1,84,8400] -> after batch strip: numDims=2, d={84,8400}
    // Layout: data[featureIdx * numProposals + proposalIdx]
    // featureIdx 0..3 = cx, cy, w, h (in network-input pixel coords)
    // featureIdx 4..83 = class scores (post-sigmoid probabilities)

    const unsigned int featureDim = layer0.inferDims.d[0]; // 84
    const unsigned int numProposals = layer0.inferDims.d[1]; // 8400

    if (featureDim < 4u + numClasses) {
        fprintf(stderr, "[YOLO11s-det] ERROR: featureDim %u < 4 + numClasses %u\n",
                featureDim, numClasses);
        return false;
    }

    for (unsigned int i = 0; i < numProposals; ++i) {
        // Find best class
        float maxScore = 0.0f;
        unsigned int bestClassId = 0;
        for (unsigned int c = 0; c < numClasses; ++c) {
            float score = data[(4u + c) * numProposals + i];
            if (score > maxScore) {
                maxScore = score;
                bestClassId = c;
            }
        }

        // Per-class threshold
        float thr = (bestClassId < detectionParams.perClassPreclusterThreshold.size())
                     ? detectionParams.perClassPreclusterThreshold[bestClassId]
                     : 0.3f;
        if (maxScore < thr) continue;

        // Box coords in network-input pixel space
        float cx = data[0u * numProposals + i];
        float cy = data[1u * numProposals + i];
        float bw = data[2u * numProposals + i];
        float bh = data[3u * numProposals + i];

        float left   = cx - bw * 0.5f;
        float top    = cy - bh * 0.5f;

        // Clamp to network input
        left  = std::max<float>(0.0f, std::min<float>(left, netW));
        top   = std::max<float>(0.0f, std::min<float>(top, netH));
        float right  = std::max<float>(0.0f, std::min<float>(cx + bw * 0.5f, netW));
        float bottom = std::max<float>(0.0f, std::min<float>(cy + bh * 0.5f, netH));

        float finalW = right - left;
        float finalH = bottom - top;
        if (finalW <= 1.0f || finalH <= 1.0f) continue;

        NvDsInferObjectDetectionInfo obj;
        obj.classId = bestClassId;
        obj.left = left;
        obj.top = top;
        obj.width = finalW;
        obj.height = finalH;
        obj.detectionConfidence = maxScore;
        objectList.push_back(obj);
    }
    // ==== END GENERATED DECODE ====
    return true;
}