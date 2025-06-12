// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/materials.h>

#include <pbrt/bsdf.h>
#include <pbrt/bssrdf.h>
#include <pbrt/interaction.h>
#include <pbrt/media.h>
#include <pbrt/paramdict.h>
#include <pbrt/textures.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/math.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/print.h>
#include <pbrt/util/spectrum.h>

#include <cmath>
#include <numeric>
#include <string>

namespace pbrt {

std::string MaterialEvalContext::ToString() const {
    return StringPrintf("[ MaterialEvalContext %s wo: %s ns: %s dpdus: %s ]",
                        TextureEvalContext::ToString(), wo, ns, dpdus);
}

std::string NormalBumpEvalContext::ToString() const {
    return StringPrintf(
        "[ NormalBumpEvalContext p: %s uv: %s shading.n: %s shading.dpdu: %s "
        "shading.dpdv: %s shading.dndu: %s shading.dndv: %s dudx: %f "
        "dudy: %f dvdx: %f dvdy: %f dpdx: %s dpdy: %s faceIndex: %d ]",
        p, uv, shading.n, shading.dpdu, shading.dpdv, shading.dndu, shading.dndv, dudx,
        dudy, dvdx, dvdy, dpdx, dpdy, faceIndex);
}

// DielectricMaterial Method Definitions
std::string DielectricMaterial::ToString() const {
    return StringPrintf(
        "[ DielectricMaterial displacement: %s normalMap: %s uRoughness: %s "
        "vRoughness: %s eta: %s remapRoughness: %s ]",
        displacement, normalMap ? normalMap->ToString() : std::string("(nullptr)"),
        uRoughness, vRoughness, eta, remapRoughness);
}

DielectricMaterial *DielectricMaterial::Create(
    const TextureParameterDictionary &parameters, Image *normalMap, const FileLoc *loc,
    Allocator alloc) {
    Spectrum eta;
    if (!parameters.GetFloatArray("eta").empty())
        eta = alloc.new_object<ConstantSpectrum>(parameters.GetFloatArray("eta")[0]);
    else
        eta = parameters.GetOneSpectrum("eta", nullptr, SpectrumType::Unbounded, alloc);
    if (!eta)
        eta = alloc.new_object<ConstantSpectrum>(1.5f);

    FloatTexture uRoughness = parameters.GetFloatTextureOrNull("uroughness", alloc);
    FloatTexture vRoughness = parameters.GetFloatTextureOrNull("vroughness", alloc);
    if (!uRoughness)
        uRoughness = parameters.GetFloatTexture("roughness", 0.f, alloc);
    if (!vRoughness)
        vRoughness = parameters.GetFloatTexture("roughness", 0.f, alloc);

    FloatTexture displacement = parameters.GetFloatTextureOrNull("displacement", alloc);
    bool remapRoughness = parameters.GetOneBool("remaproughness", true);

    return alloc.new_object<DielectricMaterial>(uRoughness, vRoughness, eta, displacement,
                                                normalMap, remapRoughness);
}

// ThinDielectricMaterial Method Definitions
std::string ThinDielectricMaterial::ToString() const {
    return StringPrintf(
        "[ ThinDielectricMaterial displacement: %s normalMap: %s eta: %s ]", displacement,
        normalMap ? normalMap->ToString() : std::string("(nullptr)"), eta);
}

ThinDielectricMaterial *ThinDielectricMaterial::Create(
    const TextureParameterDictionary &parameters, Image *normalMap, const FileLoc *loc,
    Allocator alloc) {
    Spectrum eta;
    if (!parameters.GetFloatArray("eta").empty())
        eta = alloc.new_object<ConstantSpectrum>(parameters.GetFloatArray("eta")[0]);
    else
        eta = parameters.GetOneSpectrum("eta", nullptr, SpectrumType::Unbounded, alloc);
    if (!eta)
        eta = alloc.new_object<ConstantSpectrum>(1.5f);

    FloatTexture displacement = parameters.GetFloatTextureOrNull("displacement", alloc);

    return alloc.new_object<ThinDielectricMaterial>(eta, displacement, normalMap);
}

// MixMaterial Method Definitions
std::string MixMaterial::ToString() const {
    return StringPrintf("[ MixMaterial materials: [ %s %s ] amount: %s ]", materials[0],
                        materials[1], amount);
}

MixMaterial *MixMaterial::Create(Material materials[2],
                                 const TextureParameterDictionary &parameters,
                                 const FileLoc *loc, Allocator alloc) {
    FloatTexture amount = parameters.GetFloatTexture("amount", 0.5f, alloc);

    // Check for this stuff here, where we can include the FileLoc in
    // the error message. Note that both of these limitations could be
    // relaxed if they were problematic; the issue is that we currently
    // resolve MixMaterials in the closest hit shader...
#ifdef PBRT_BUILD_GPU_RENDERER
    if (Options->useGPU && !BasicTextureEvaluator().CanEvaluate({amount}, {}))
        ErrorExit(loc, "The GPU renderer currently only supports basic textures "
                       "for its \"amount\" parameter.");
#else
    if (Options->wavefront && !BasicTextureEvaluator().CanEvaluate({amount}, {}))
        ErrorExit(loc, "The wavefront renderer currently only supports basic textures "
                       "for its \"amount\" parameter.");
#endif

    return alloc.new_object<MixMaterial>(materials, amount);
}

// HairMaterial Method Definitions
std::string HairMaterial::ToString() const {
    return StringPrintf("[ HairMaterial sigma_a: %s color: %s eumelanin: %s "
                        "pheomelanin: %s eta: %s beta_m: %s beta_n: %s alpha: %s ]",
                        sigma_a, color, eumelanin, pheomelanin, eta, beta_m, beta_n,
                        alpha);
}

HairMaterial *HairMaterial::Create(const TextureParameterDictionary &parameters,
                                   const FileLoc *loc, Allocator alloc) {
    SpectrumTexture sigma_a =
        parameters.GetSpectrumTextureOrNull("sigma_a", SpectrumType::Unbounded, alloc);
    SpectrumTexture reflectance =
        parameters.GetSpectrumTextureOrNull("reflectance", SpectrumType::Albedo, alloc);
    if (!reflectance)
        reflectance = parameters.GetSpectrumTextureOrNull("color", SpectrumType::Albedo, alloc);
    FloatTexture eumelanin = parameters.GetFloatTextureOrNull("eumelanin", alloc);
    FloatTexture pheomelanin = parameters.GetFloatTextureOrNull("pheomelanin", alloc);
    if (sigma_a) {
        if (reflectance)
            Warning(loc, R"(Ignoring "reflectance" parameter since "sigma_a" was provided.)");
        if (eumelanin)
            Warning(loc, "Ignoring \"eumelanin\" parameter since \"sigma_a\" was "
                         "provided.");
        if (pheomelanin)
            Warning(loc, "Ignoring \"pheomelanin\" parameter since \"sigma_a\" was "
                         "provided.");
    } else if (reflectance) {
        if (sigma_a)
            Warning(loc, R"(Ignoring "sigma_a" parameter since "reflectance" was provided.)");
        if (eumelanin)
            Warning(loc, "Ignoring \"eumelanin\" parameter since \"reflectance\" was "
                         "provided.");
        if (pheomelanin)
            Warning(loc, "Ignoring \"pheomelanin\" parameter since \"reflectance\" was "
                         "provided.");
    } else if (eumelanin || pheomelanin) {
        if (sigma_a)
            Warning(loc, "Ignoring \"sigma_a\" parameter since "
                         "\"eumelanin\"/\"pheomelanin\" was provided.");
        if (reflectance)
            Warning(loc, "Ignoring \"reflectance\" parameter since "
                         "\"eumelanin\"/\"pheomelanin\" was provided.");
    } else {
        // Default: brown-ish hair.
        sigma_a = alloc.new_object<SpectrumConstantTexture>(
            alloc.new_object<RGBUnboundedSpectrum>(
                HairBxDF::SigmaAFromConcentration(1.3, 0.)));
    }

    FloatTexture eta = parameters.GetFloatTexture("eta", 1.55f, alloc);
    FloatTexture beta_m = parameters.GetFloatTexture("beta_m", 0.3f, alloc);
    FloatTexture beta_n = parameters.GetFloatTexture("beta_n", 0.3f, alloc);
    FloatTexture alpha = parameters.GetFloatTexture("alpha", 2.f, alloc);

    return alloc.new_object<HairMaterial>(sigma_a, reflectance, eumelanin, pheomelanin, eta,
                                          beta_m, beta_n, alpha);
}

// DiffuseMaterial Method Definitions
std::string DiffuseMaterial::ToString() const {
    return StringPrintf(
        "[ DiffuseMaterial displacement: %s normapMap: %s reflectance: %s ]",
        displacement, normalMap ? normalMap->ToString() : std::string("(nullptr)"),
        reflectance);
}

DiffuseMaterial *DiffuseMaterial::Create(const TextureParameterDictionary &parameters,
                                         Image *normalMap, const FileLoc *loc,
                                         Allocator alloc) {
    SpectrumTexture reflectance = parameters.GetSpectrumTexture(
        "reflectance", nullptr, SpectrumType::Albedo, alloc);
    if (!reflectance)
        reflectance = alloc.new_object<SpectrumConstantTexture>(
            alloc.new_object<ConstantSpectrum>(0.5f));
    FloatTexture displacement = parameters.GetFloatTextureOrNull("displacement", alloc);

    return alloc.new_object<DiffuseMaterial>(reflectance, displacement, normalMap);
}

// ConductorMaterial Method Definitions
std::string ConductorMaterial::ToString() const {
    return StringPrintf("[ ConductorMaterial displacement: %s normalMap: %s eta: %s "
                        "k: %s reflectance: %s uRoughness: %s vRoughness: %s "
                        "remapRoughness: %s ]",
                        displacement,
                        normalMap ? normalMap->ToString() : std::string("(nullptr)"), eta,
                        k, reflectance, uRoughness, vRoughness, remapRoughness);
}

ConductorMaterial *ConductorMaterial::Create(const TextureParameterDictionary &parameters,
                                             Image *normalMap, const FileLoc *loc,
                                             Allocator alloc) {
    SpectrumTexture eta =
        parameters.GetSpectrumTextureOrNull("eta", SpectrumType::Unbounded, alloc);
    SpectrumTexture k =
        parameters.GetSpectrumTextureOrNull("k", SpectrumType::Unbounded, alloc);
    SpectrumTexture reflectance =
        parameters.GetSpectrumTextureOrNull("reflectance", SpectrumType::Albedo, alloc);

    if (reflectance && (eta || k))
        ErrorExit(loc, "For the conductor material, both \"reflectance\" "
                       "and \"eta\" and \"k\" can't be provided.");
    if (!reflectance) {
        if (!eta)
            eta = alloc.new_object<SpectrumConstantTexture>(
                GetNamedSpectrum("metal-Cu-eta"));
        if (!k)
            k = alloc.new_object<SpectrumConstantTexture>(GetNamedSpectrum("metal-Cu-k"));
    }

    FloatTexture uRoughness = parameters.GetFloatTextureOrNull("uroughness", alloc);
    FloatTexture vRoughness = parameters.GetFloatTextureOrNull("vroughness", alloc);
    if (!uRoughness)
        uRoughness = parameters.GetFloatTexture("roughness", 0.f, alloc);
    if (!vRoughness)
        vRoughness = parameters.GetFloatTexture("roughness", 0.f, alloc);

    FloatTexture displacement = parameters.GetFloatTextureOrNull("displacement", alloc);
    bool remapRoughness = parameters.GetOneBool("remaproughness", true);

    return alloc.new_object<ConductorMaterial>(eta, k, reflectance, uRoughness,
                                               vRoughness, displacement, normalMap,
                                               remapRoughness);
}

// CoatedDiffuseMaterial Method Definitions
template <typename TextureEvaluator>
CoatedDiffuseBxDF CoatedDiffuseMaterial::GetBxDF(TextureEvaluator texEval,
                                                 const MaterialEvalContext &ctx,
                                                 SampledWavelengths &lambda) const {
    // Initialize diffuse component of plastic material
    SampledSpectrum r = Clamp(texEval(reflectance, ctx, lambda), 0, 1);

    // Create microfacet distribution _distrib_ for coated diffuse material
    Float urough = texEval(uRoughness, ctx);
    Float vrough = texEval(vRoughness, ctx);
    if (remapRoughness) {
        urough = TrowbridgeReitzDistribution::RoughnessToAlpha(urough);
        vrough = TrowbridgeReitzDistribution::RoughnessToAlpha(vrough);
    }
    TrowbridgeReitzDistribution distrib(urough, vrough);

    Float thick = texEval(thickness, ctx);

    Float sampledEta = eta(lambda[0]);
    if (!eta.template Is<ConstantSpectrum>())
        lambda.TerminateSecondary();
    if (sampledEta == 0)
        sampledEta = 1;

    SampledSpectrum a = Clamp(texEval(albedo, ctx, lambda), 0, 1);
    Float gg = Clamp(texEval(g, ctx), -1, 1);

    return CoatedDiffuseBxDF(DielectricBxDF(sampledEta, distrib), DiffuseBxDF(r), thick,
                             a, gg, maxDepth, nSamples);
}

// Explicit template instantiation
template CoatedDiffuseBxDF CoatedDiffuseMaterial::GetBxDF(
    BasicTextureEvaluator, const MaterialEvalContext &ctx,
    SampledWavelengths &lambda) const;
template CoatedDiffuseBxDF CoatedDiffuseMaterial::GetBxDF(
    UniversalTextureEvaluator, const MaterialEvalContext &ctx,
    SampledWavelengths &lambda) const;

std::string CoatedDiffuseMaterial::ToString() const {
    return StringPrintf(
        "[ CoatedDiffuseMaterial displacement: %s normalMap: %s reflectance: %s "
        "uRoughness: %s vRoughness: %s thickness: %s eta: %s remapRoughness: %s ]",
        displacement, normalMap ? normalMap->ToString() : std::string("(nullptr)"),
        reflectance, uRoughness, vRoughness, thickness, eta, remapRoughness);
}

CoatedDiffuseMaterial *CoatedDiffuseMaterial::Create(
    const TextureParameterDictionary &parameters, Image *normalMap, const FileLoc *loc,
    Allocator alloc) {
    SpectrumTexture reflectance = parameters.GetSpectrumTexture(
        "reflectance", nullptr, SpectrumType::Albedo, alloc);
    if (!reflectance)
        reflectance = alloc.new_object<SpectrumConstantTexture>(
            alloc.new_object<ConstantSpectrum>(0.5f));

    FloatTexture uRoughness = parameters.GetFloatTextureOrNull("uroughness", alloc);
    FloatTexture vRoughness = parameters.GetFloatTextureOrNull("vroughness", alloc);
    if (!uRoughness)
        uRoughness = parameters.GetFloatTexture("roughness", 0.f, alloc);
    if (!vRoughness)
        vRoughness = parameters.GetFloatTexture("roughness", 0.f, alloc);

    FloatTexture thickness = parameters.GetFloatTexture("thickness", .01, alloc);

    Spectrum eta;
    if (!parameters.GetFloatArray("eta").empty())
        eta = alloc.new_object<ConstantSpectrum>(parameters.GetFloatArray("eta")[0]);
    else
        eta = parameters.GetOneSpectrum("eta", nullptr, SpectrumType::Unbounded, alloc);
    if (!eta)
        eta = alloc.new_object<ConstantSpectrum>(1.5f);

    int maxDepth = parameters.GetOneInt("maxdepth", 10);
    int nSamples = parameters.GetOneInt("nsamples", 1);

    FloatTexture g = parameters.GetFloatTexture("g", 0.f, alloc);
    SpectrumTexture albedo =
        parameters.GetSpectrumTexture("albedo", nullptr, SpectrumType::Albedo, alloc);
    if (!albedo)
        albedo = alloc.new_object<SpectrumConstantTexture>(
            alloc.new_object<ConstantSpectrum>(0.f));

    FloatTexture displacement = parameters.GetFloatTextureOrNull("displacement", alloc);
    bool remapRoughness = parameters.GetOneBool("remaproughness", true);

    return alloc.new_object<CoatedDiffuseMaterial>(
        reflectance, uRoughness, vRoughness, thickness, albedo, g, eta, displacement,
        normalMap, remapRoughness, maxDepth, nSamples);
}

// *Add RTX-DI material
// DonutMaterial Method Definitions
// TODO:: Use Falcor solution
Float GetPerceivedBrightness(SampledSpectrum color)
{
    return std::sqrt(0.299 * color[0] * color[0] + 0.587 * color[1] * color[1] + 0.114 * color[2] * color[2]);
}

Float SolveMetalness(Float diffuse, Float specular, Float oneMinusSpecularStrength)
{
    if (specular < c_DielectricSpecular)
        return 0;

    Float a = c_DielectricSpecular;
    Float b = diffuse * oneMinusSpecularStrength / (1 - c_DielectricSpecular) + specular - 2 * c_DielectricSpecular;
    Float c = c_DielectricSpecular - specular;
    Float D = std::max<Float>(b * b - 4 * a * c, 0);
    return Clamp((-b + std::sqrt(D)) / (2 * a), 0.0f, 1.0f);
}

#ifndef USE_FALCOR_MATERIAL
template <typename TextureEvaluator>
MetalRoughnessBxDF DonutMaterial::GetBxDF(TextureEvaluator texEval,
                                                 const MaterialEvalContext &ctx,
                                                 SampledWavelengths &lambda) const {
    // return FalcorStandardBSDF(SampledSpectrum(1.0f), SampledSpectrum(0.0f), 1.0f);

    SampledSpectrum resultDiffuseAlbedo;
    SampledSpectrum resultSpecularF0;
    Float resultRoughness;
    bool resultMetalRough = false;
    // Initialize diffuse component of plastic material
    if (useSpecularGlossModel)
    {
        SampledSpectrum resultDiffuseTex = Clamp(texEval(diffuseTex, ctx, lambda), 0.0f, 1.0f);
        SampledSpectrum resultSpecularTex = Clamp(texEval(specularTex, ctx, lambda), 0.0f, 1.0f);
        Float resultGloness = Clamp(texEval(glossTex, ctx), 0.0f, 1.0f);

        SampledSpectrum diffuseColor = diffuse.Sample(lambda) * resultDiffuseTex;
        SampledSpectrum specularColor = specular.Sample(lambda) * resultSpecularTex;
        resultRoughness = 1.0 - resultGloness * (1.0 - roughness.Sample(lambda)[0]);

        const Float epsilon = 1e-6;

        Float oneMinusSpecularStrength = 1.0 - std::max<Float>(specularColor[0], std::max<Float>(specularColor[1], specularColor[2]));
        Float resultMetallic = SolveMetalness(GetPerceivedBrightness(diffuseColor), GetPerceivedBrightness(specularColor), oneMinusSpecularStrength);

        SampledSpectrum baseColorFromDiffuse = diffuseColor * (oneMinusSpecularStrength / (1 - c_DielectricSpecular) / std::max<Float>(1 - resultMetallic, epsilon));
        SampledSpectrum baseColorFromSpecular = specularColor - SampledSpectrum(c_DielectricSpecular * (1 - resultMetallic) / std::max<Float>(resultMetallic, epsilon));
        SampledSpectrum resultBaseColor = Clamp(Lerp(resultMetallic * resultMetallic, baseColorFromDiffuse, baseColorFromSpecular), 0.0f, 1.0f);
        resultMetalRough = true;

        // Compute the BRDF inputs for the specular-gloss model
        // https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_materials_pbrMetalRoughness/README.md#specular---glossiness
        resultDiffuseAlbedo = diffuseColor * (1.0 - std::max<Float>(specularColor[0], std::max<Float>(specularColor[1], specularColor[2])));
        resultSpecularF0 = specularColor;
    }
    else
    {
        SampledSpectrum resultDiffuseTex = Clamp(texEval(diffuseTex, ctx, lambda), 0.0f, 1.0f);
        SampledSpectrum resultSpecularTex = Clamp(texEval(specularTex, ctx, lambda), 0.0f, 1.0f);

        SampledSpectrum resultBaseColor = diffuse.Sample(lambda) * resultDiffuseTex;
        resultRoughness = roughness.Sample(lambda)[0] * resultSpecularTex[1];
        Float resultMetallic = metallic.Sample(lambda)[0] * resultSpecularTex[2];
        resultMetalRough = true;

        // Compute the BRDF inputs for the metal-rough model
        // https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#metal-brdf-and-dielectric-brdf
        // resultDiffuseAlbedo = diffuse.Sample(lambda);
        // resultSpecularF0 = resultDiffuseTex;
        resultDiffuseAlbedo = Lerp(resultMetallic, resultBaseColor * (1.0 - c_DielectricSpecular), SampledSpectrum(.0f));
        resultSpecularF0 = Lerp(resultMetallic, SampledSpectrum(c_DielectricSpecular), resultBaseColor);
    }
    
    
    return MetalRoughnessBxDF(resultDiffuseAlbedo, resultSpecularF0, resultRoughness);
}

// Explicit template instantiation
template MetalRoughnessBxDF DonutMaterial::GetBxDF(
    BasicTextureEvaluator, const MaterialEvalContext &ctx,
    SampledWavelengths &lambda) const;
template MetalRoughnessBxDF DonutMaterial::GetBxDF(
    UniversalTextureEvaluator, const MaterialEvalContext &ctx,
    SampledWavelengths &lambda) const;
#else
// maybe pass N and V too like Falcor solution?
template <typename TextureEvaluator>
FalcorBxDF DonutMaterial::GetBxDF(TextureEvaluator texEval,
                                                 const MaterialEvalContext &ctx,
                                                 SampledWavelengths &lambda) const {
    // return FalcorStandardBSDF(SampledSpectrum(1.0f), SampledSpectrum(0.0f), 1.0f);

    SampledSpectrum resultBaseColor;
    SampledSpectrum resultDiffuseAlbedo;
    SampledSpectrum resultSpecularF0;
    Float resultRoughness, resultMetallic, resultEta;

    bool resultMetalRough = false;
    // Initialize diffuse component of plastic material
    // only use MetalRoughnees currently
    if (useSpecularGlossModel)
    {
        SampledSpectrum resultDiffuseTex = Clamp(texEval(diffuseTex, ctx, lambda), 0.0f, 1.0f);
        SampledSpectrum resultSpecularTex = Clamp(texEval(specularTex, ctx, lambda), 0.0f, 1.0f);
        Float resultGloness = Clamp(texEval(glossTex, ctx), 0.0f, 1.0f);

        SampledSpectrum diffuseColor = diffuse.Sample(lambda) * resultDiffuseTex;
        SampledSpectrum specularColor = specular.Sample(lambda) * resultSpecularTex;
        resultRoughness = 1.0 - resultGloness * (1.0 - roughness.Sample(lambda)[0]);

        const Float epsilon = 1e-6;

        Float oneMinusSpecularStrength = 1.0 - std::max<Float>(specularColor[0], std::max<Float>(specularColor[1], specularColor[2]));
        Float resultMetallic = SolveMetalness(GetPerceivedBrightness(diffuseColor), GetPerceivedBrightness(specularColor), oneMinusSpecularStrength);

        SampledSpectrum baseColorFromDiffuse = diffuseColor * (oneMinusSpecularStrength / (1 - c_DielectricSpecular) / std::max<Float>(1 - resultMetallic, epsilon));
        SampledSpectrum baseColorFromSpecular = specularColor - SampledSpectrum(c_DielectricSpecular * (1 - resultMetallic) / std::max<Float>(resultMetallic, epsilon));
        SampledSpectrum resultBaseColor = Clamp(Lerp(resultMetallic * resultMetallic, baseColorFromDiffuse, baseColorFromSpecular), 0.0f, 1.0f);
        resultMetalRough = true;

        // Compute the BRDF inputs for the specular-gloss model
        // https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_materials_pbrMetalRoughness/README.md#specular---glossiness
        resultDiffuseAlbedo = diffuseColor * (1.0 - std::max<Float>(specularColor[0], std::max<Float>(specularColor[1], specularColor[2])));
        resultSpecularF0 = specularColor;
    }
    else
    {
        SampledSpectrum resultDiffuseTex = Clamp(texEval(diffuseTex, ctx, lambda), 0.0f, 1.0f);
        SampledSpectrum resultSpecularTex = Clamp(texEval(specularTex, ctx, lambda), 0.0f, 1.0f);

        resultBaseColor = diffuse.Sample(lambda) * resultDiffuseTex;
        resultRoughness = roughness.Sample(lambda)[0] * resultSpecularTex[1];
        resultMetallic = metallic.Sample(lambda)[0] * resultSpecularTex[2];
        resultMetalRough = true;

        // Compute the BRDF inputs for the metal-rough model
        // https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#metal-brdf-and-dielectric-brdf
        // resultDiffuseAlbedo = Lerp(resultMetallic, resultBaseColor * (1.0 - c_DielectricSpecular), SampledSpectrum(.0f));
        // resultSpecularF0 = Lerp(resultMetallic, SampledSpectrum(c_DielectricSpecular), resultBaseColor);

        // use the Disney solusion like Falcor
        Float f = (ior.Sample(lambda)[0] - 1.f) / (ior.Sample(lambda)[0] + 1.f);
        Float F0 = f * f;

        resultDiffuseAlbedo = Lerp(resultMetallic, resultBaseColor, SampledSpectrum(.0f));
        resultSpecularF0 = Lerp(resultMetallic, SampledSpectrum(F0), resultBaseColor);
    }

    resultRoughness = std::min(1.0f, resultRoughness);

    Float resultTransmission, resultDiffuseTransmission;
    {
        SampledSpectrum resultTransmissionTex = Clamp(texEval(transmissionTex, ctx, lambda), 0.0f, 1.0f);
        resultTransmission = transmissionFactor.Sample(lambda)[0] * resultTransmissionTex[0];
        resultDiffuseTransmission = diffuseTransmissionFactor.Sample(lambda)[0] * resultTransmissionTex[0];
    }

    resultEta = ior.Sample(lambda)[0];
    
    // TODO? occlusion, opacity, emissive, ior?

    // convert from dount material to falcor material

    // from https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_transmission/README.md#refraction
    // "This microfacet lobe is exactly the same as the specular lobe except sampled along the line of sight through the surface."
    Float resultSpecularTransmission = resultTransmission * (1 - resultMetallic);    // (1 - donutMaterial.metalness) is from https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_transmission/README.md#transparent-metals
    Float diffuseTransmission = resultDiffuseTransmission * (1 - resultMetallic);    // (1 - donutMaterial.metalness) is from https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_transmission/README.md#transparent-metals
    SampledSpectrum transmission = resultBaseColor;

    return FalcorBxDF(resultDiffuseAlbedo, resultSpecularF0,
                      resultRoughness, resultMetallic, resultEta,
                      transmission, diffuseTransmission, resultSpecularTransmission);
}

// Explicit template instantiation
template FalcorBxDF DonutMaterial::GetBxDF(
    BasicTextureEvaluator, const MaterialEvalContext &ctx,
    SampledWavelengths &lambda) const;
template FalcorBxDF DonutMaterial::GetBxDF(
    UniversalTextureEvaluator, const MaterialEvalContext &ctx,
    SampledWavelengths &lambda) const;
#endif
std::string DonutMaterial::ToString() const {
    return StringPrintf(
        "[ DonutMaterial displacement: %s normalMap: %s"
        "diffuseTex: %s specularTex: %s diffuseFactor: %s specularFactor: %s"
        " roughness: %s metallic: %s eta: %s useSpecularGlossModel: %s ]",
        displacement, normalMap ? normalMap->ToString() : std::string("(nullptr)"),
        diffuseTex, specularTex, diffuse, specular, roughness, metallic, ior, useSpecularGlossModel);
}

#ifndef USE_FALCOR_MATERIAL
// GLTF material MetalRoughness or MatelRoughness define
DonutMaterial *DonutMaterial::Create(
    const TextureParameterDictionary &parameters, Image *normalMap, const FileLoc *loc,
    Allocator alloc) {
    SpectrumTexture diffuseTex = parameters.GetSpectrumTexture(
        "diffusetex", nullptr, SpectrumType::Constant, alloc);
    if (!diffuseTex)
        diffuseTex = alloc.new_object<SpectrumConstantTexture>(
            alloc.new_object<ConstantSpectrum>(1.0f));

    SpectrumTexture specularTex = parameters.GetSpectrumTexture(
        "speculartex", nullptr, SpectrumType::Constant, alloc);
    if (!specularTex)
        specularTex = alloc.new_object<SpectrumConstantTexture>(
            alloc.new_object<ConstantSpectrum>(1.0f));

    // specular texture's alpha channel for DonutMaterial
    FloatTexture glossTex = parameters.GetFloatTextureOrNull("glosstex", alloc);

    Spectrum diffuse;
    diffuse = parameters.GetOneSpectrum("diffuse", nullptr, SpectrumType::Constant, alloc);
    if (!diffuse)
        diffuse = alloc.new_object<RGBConstantSpectrum>(*RGBColorSpace::sRGB, RGB(1.0f, 1.0f, 1.0f));

    Spectrum specular;
    specular = parameters.GetOneSpectrum("specular", nullptr, SpectrumType::Constant, alloc);
    if (!specular)
        specular = alloc.new_object<RGBConstantSpectrum>(*RGBColorSpace::sRGB, RGB(1.0f, 1.0f, 1.0f));

    Spectrum metallic;
    if (!parameters.GetFloatArray("metallic").empty())
        metallic = alloc.new_object<ConstantSpectrum>(parameters.GetFloatArray("metallic")[0]);
    else
        metallic = parameters.GetOneSpectrum("metallic", nullptr, SpectrumType::Constant, alloc);
    if (!metallic)
        metallic = alloc.new_object<ConstantSpectrum>(1.0f);

    Spectrum roughness;
    if (!parameters.GetFloatArray("roughness").empty())
        roughness = alloc.new_object<ConstantSpectrum>(parameters.GetFloatArray("roughness")[0]);
    else
        roughness = parameters.GetOneSpectrum("roughness", nullptr, SpectrumType::Constant, alloc);
    if (!roughness)
        roughness = alloc.new_object<ConstantSpectrum>(1.0f);

    Spectrum eta;
    if (!parameters.GetFloatArray("eta").empty())
        eta = alloc.new_object<ConstantSpectrum>(parameters.GetFloatArray("eta")[0]);
    else
        eta = parameters.GetOneSpectrum("eta", nullptr, SpectrumType::Unbounded, alloc);
    if (!eta)
        eta = alloc.new_object<ConstantSpectrum>(1.5f);

    int maxDepth = parameters.GetOneInt("maxdepth", 10);
    int nSamples = parameters.GetOneInt("nsamples", 1);

    FloatTexture displacement = parameters.GetFloatTextureOrNull("displacement", alloc);
    bool useSpecularGlossModel = parameters.GetOneBool("useSpecularGlossModel", false);

    return alloc.new_object<DonutMaterial>(
        diffuseTex, specularTex, glossTex, diffuse, specular, roughness, metallic, eta, displacement,
        normalMap, useSpecularGlossModel, maxDepth, nSamples);
}
#else
// GLTF material to Falcor define
DonutMaterial *DonutMaterial::Create(
    const TextureParameterDictionary &parameters, Image *normalMap, const FileLoc *loc,
    Allocator alloc) {
    SpectrumTexture diffuseTex = parameters.GetSpectrumTexture(
        "diffusetex", nullptr, SpectrumType::Constant, alloc);
    if (!diffuseTex)
        diffuseTex = alloc.new_object<SpectrumConstantTexture>(
            alloc.new_object<ConstantSpectrum>(1.0f));

    SpectrumTexture specularTex = parameters.GetSpectrumTexture(
        "speculartex", nullptr, SpectrumType::Constant, alloc);
    if (!specularTex)
        specularTex = alloc.new_object<SpectrumConstantTexture>(
            alloc.new_object<ConstantSpectrum>(1.0f));

    //TODO:: Assimp Scene convert
    // this is same as texture in Light, maybe share same data?
    // ignore currently
    SpectrumTexture emissiveTex = parameters.GetSpectrumTexture(
        "emissivetex", nullptr, SpectrumType::Constant, alloc);
    if (!emissiveTex)
        emissiveTex = alloc.new_object<SpectrumConstantTexture>(
            alloc.new_object<ConstantSpectrum>(0.0f));

    //TODO:: Assimp Scene convert
    // ignore currently
    SpectrumTexture occlusionTex = parameters.GetSpectrumTexture(
        "occlusionTex", nullptr, SpectrumType::Constant, alloc);
    if (!occlusionTex)
        occlusionTex = alloc.new_object<SpectrumConstantTexture>(
            alloc.new_object<ConstantSpectrum>(0.0f));

    //TODO:: Assimp Scene convert
    // ignore currently
    SpectrumTexture transmissionTex = parameters.GetSpectrumTexture(
        "transmissiontex", nullptr, SpectrumType::Constant, alloc);
    if (!transmissionTex)
        transmissionTex = alloc.new_object<SpectrumConstantTexture>(
            alloc.new_object<ConstantSpectrum>(0.0f));

    FloatTexture glossTex = parameters.GetFloatTextureOrNull("glosstex", alloc);

    Spectrum diffuse;
    diffuse = parameters.GetOneSpectrum("diffuse", nullptr, SpectrumType::Constant, alloc);
    if (!diffuse)
        diffuse = alloc.new_object<RGBConstantSpectrum>(*RGBColorSpace::sRGB, RGB(1.0f, 1.0f, 1.0f));

    Spectrum specular;
    specular = parameters.GetOneSpectrum("specular", nullptr, SpectrumType::Constant, alloc);
    if (!specular)
        specular = alloc.new_object<RGBConstantSpectrum>(*RGBColorSpace::sRGB, RGB(1.0f, 1.0f, 1.0f));

    //TODO:: Assimp Scene convert
    Spectrum emissive;
    emissive = parameters.GetOneSpectrum("emissive", nullptr, SpectrumType::Constant, alloc);
    if (!emissive)
        emissive = alloc.new_object<RGBConstantSpectrum>(*RGBColorSpace::sRGB, RGB(0.0f, 0.0f, 0.0f));

    

    Spectrum metallic;
    if (!parameters.GetFloatArray("metallic").empty())
        metallic = alloc.new_object<ConstantSpectrum>(parameters.GetFloatArray("metallic")[0]);
    else
        metallic = parameters.GetOneSpectrum("metallic", nullptr, SpectrumType::Constant, alloc);
    if (!metallic)
        metallic = alloc.new_object<ConstantSpectrum>(1.0f);

    Spectrum roughness;
    if (!parameters.GetFloatArray("roughness").empty())
        roughness = alloc.new_object<ConstantSpectrum>(parameters.GetFloatArray("roughness")[0]);
    else
        roughness = parameters.GetOneSpectrum("roughness", nullptr, SpectrumType::Constant, alloc);
    if (!roughness)
        roughness = alloc.new_object<ConstantSpectrum>(1.0f);

    //TODO:: Assimp Scene convert
    Spectrum opacity;
    if (!parameters.GetFloatArray("opacity").empty())
        opacity = alloc.new_object<ConstantSpectrum>(parameters.GetFloatArray("opacity")[0]);
    else
        opacity = parameters.GetOneSpectrum("opacity", nullptr, SpectrumType::Constant, alloc);
    if (!opacity)
        opacity = alloc.new_object<ConstantSpectrum>(1.0f);

    //TODO:: Assimp Scene convert
    Spectrum occlusionStrength;
    if (!parameters.GetFloatArray("occlusionstrength").empty())
        occlusionStrength = alloc.new_object<ConstantSpectrum>(parameters.GetFloatArray("occlusionstrength")[0]);
    else
        occlusionStrength = parameters.GetOneSpectrum("occlusionstrength", nullptr, SpectrumType::Constant, alloc);
    if (!occlusionStrength)
        occlusionStrength = alloc.new_object<ConstantSpectrum>(1.0f);

    //TODO:: Assimp Scene convert
    Spectrum transmissionFactor;
    if (!parameters.GetFloatArray("transmissionfactor").empty())
        transmissionFactor = alloc.new_object<ConstantSpectrum>(parameters.GetFloatArray("transmissionfactor")[0]);
    else
        transmissionFactor = parameters.GetOneSpectrum("transmissionfactor", nullptr, SpectrumType::Constant, alloc);
    if (!transmissionFactor)
        transmissionFactor = alloc.new_object<ConstantSpectrum>(0.0f);

    //TODO:: Assimp Scene convert
    Spectrum diffuseTransmissionFactor;
    if (!parameters.GetFloatArray("diffusetransmissionfactor").empty())
        diffuseTransmissionFactor = alloc.new_object<ConstantSpectrum>(parameters.GetFloatArray("diffusetransmissionfactor")[0]);
    else
        diffuseTransmissionFactor = parameters.GetOneSpectrum("diffusetransmissionfactor", nullptr, SpectrumType::Constant, alloc);
    if (!diffuseTransmissionFactor)
        diffuseTransmissionFactor = alloc.new_object<ConstantSpectrum>(0.0f);

    //TODO:: Assimp Scene convert
    Spectrum ior;
    if (!parameters.GetFloatArray("ior").empty())
        ior = alloc.new_object<ConstantSpectrum>(parameters.GetFloatArray("ior")[0]);
    else
        ior = parameters.GetOneSpectrum("ior", nullptr, SpectrumType::Unbounded, alloc);
    if (!ior)
        ior = alloc.new_object<ConstantSpectrum>(1.5f);

    int maxDepth = parameters.GetOneInt("maxdepth", 10);
    int nSamples = parameters.GetOneInt("nsamples", 1);

    FloatTexture displacement = parameters.GetFloatTextureOrNull("displacement", alloc);
    bool useSpecularGlossModel = parameters.GetOneBool("useSpecularGlossModel", false);

    return alloc.new_object<DonutMaterial>(
        diffuseTex, specularTex, emissiveTex,
        occlusionTex, transmissionTex, glossTex,
        diffuse, specular, emissive,
        metallic, roughness,
        opacity, occlusionStrength,
        transmissionFactor, diffuseTransmissionFactor,
        ior,
        displacement, normalMap,
        useSpecularGlossModel, maxDepth, nSamples);
}
#endif
template <typename TextureEvaluator>
CoatedConductorBxDF CoatedConductorMaterial::GetBxDF(TextureEvaluator texEval,
                                                     const MaterialEvalContext &ctx,
                                                     SampledWavelengths &lambda) const {
    Float iurough = texEval(interfaceURoughness, ctx);
    Float ivrough = texEval(interfaceVRoughness, ctx);
    if (remapRoughness) {
        iurough = TrowbridgeReitzDistribution::RoughnessToAlpha(iurough);
        ivrough = TrowbridgeReitzDistribution::RoughnessToAlpha(ivrough);
    }
    TrowbridgeReitzDistribution interfaceDistrib(iurough, ivrough);

    Float thick = texEval(thickness, ctx);

    Float ieta = interfaceEta(lambda[0]);
    if (!interfaceEta.template Is<ConstantSpectrum>())
        lambda.TerminateSecondary();
    if (ieta == 0)
        ieta = 1;

    SampledSpectrum ce, ck;
    if (conductorEta) {
        ce = texEval(conductorEta, ctx, lambda);
        ck = texEval(k, ctx, lambda);
    } else {
        // Avoid r==1 NaN case...
        SampledSpectrum r = Clamp(texEval(reflectance, ctx, lambda), 0, .9999);
        ce = SampledSpectrum(1.f);
        ck = 2 * Sqrt(r) / Sqrt(ClampZero(SampledSpectrum(1) - r));
    }
    ce /= ieta;
    ck /= ieta;

    Float curough = texEval(conductorURoughness, ctx);
    Float cvrough = texEval(conductorVRoughness, ctx);
    if (remapRoughness) {
        curough = TrowbridgeReitzDistribution::RoughnessToAlpha(curough);
        cvrough = TrowbridgeReitzDistribution::RoughnessToAlpha(cvrough);
    }
    TrowbridgeReitzDistribution conductorDistrib(curough, cvrough);

    SampledSpectrum a = Clamp(texEval(albedo, ctx, lambda), 0, 1);
    Float gg = Clamp(texEval(g, ctx), -1, 1);

    return CoatedConductorBxDF(DielectricBxDF(ieta, interfaceDistrib),
                               ConductorBxDF(conductorDistrib, ce, ck), thick, a, gg,
                               maxDepth, nSamples);
}

template CoatedConductorBxDF CoatedConductorMaterial::GetBxDF(
    BasicTextureEvaluator, const MaterialEvalContext &ctx,
    SampledWavelengths &lambda) const;
template CoatedConductorBxDF CoatedConductorMaterial::GetBxDF(
    UniversalTextureEvaluator, const MaterialEvalContext &ctx,
    SampledWavelengths &lambda) const;

std::string CoatedConductorMaterial::ToString() const {
    return StringPrintf(
        "[ CoatedConductorMaterial displacement: %s normalMap: %s "
        "interfaceURoughness: %s interfaceVRoughness: %s thickness: %s "
        "interfaceEta: %s g: %s albedo: %s conductorURoughness: %s "
        "conductorVRoughness: %s conductorEta: %s k: %s "
        "conductorReflectance: %s remapRoughness: %s maxDepth: %d nSamples: %d ]",
        displacement, normalMap ? normalMap->ToString() : std::string("(nullptr)"),
        interfaceURoughness, interfaceVRoughness, thickness, interfaceEta, g, albedo,
        conductorURoughness, conductorVRoughness, conductorEta, k, reflectance,
        remapRoughness, maxDepth, nSamples);
}

CoatedConductorMaterial *CoatedConductorMaterial::Create(
    const TextureParameterDictionary &parameters, Image *normalMap, const FileLoc *loc,
    Allocator alloc) {
    // interface
    FloatTexture interfaceURoughness =
        parameters.GetFloatTextureOrNull("interface.uroughness", alloc);
    FloatTexture interfaceVRoughness =
        parameters.GetFloatTextureOrNull("interface.vroughness", alloc);
    if (!interfaceURoughness)
        interfaceURoughness =
            parameters.GetFloatTexture("interface.roughness", 0.f, alloc);
    if (!interfaceVRoughness)
        interfaceVRoughness =
            parameters.GetFloatTexture("interface.roughness", 0.f, alloc);

    FloatTexture thickness = parameters.GetFloatTexture("thickness", .01, alloc);

    Spectrum interfaceEta;
    if (!parameters.GetFloatArray("interface.eta").empty())
        interfaceEta = alloc.new_object<ConstantSpectrum>(
            parameters.GetFloatArray("interface.eta")[0]);
    else
        interfaceEta = parameters.GetOneSpectrum("interface.eta", nullptr,
                                                 SpectrumType::Unbounded, alloc);
    if (!interfaceEta)
        interfaceEta = alloc.new_object<ConstantSpectrum>(1.5f);

    // conductor
    FloatTexture conductorURoughness =
        parameters.GetFloatTextureOrNull("conductor.uroughness", alloc);
    FloatTexture conductorVRoughness =
        parameters.GetFloatTextureOrNull("conductor.vroughness", alloc);
    if (!conductorURoughness)
        conductorURoughness =
            parameters.GetFloatTexture("conductor.roughness", 0.f, alloc);
    if (!conductorVRoughness)
        conductorVRoughness =
            parameters.GetFloatTexture("conductor.roughness", 0.f, alloc);
    SpectrumTexture conductorEta = parameters.GetSpectrumTextureOrNull(
        "conductor.eta", SpectrumType::Unbounded, alloc);
    SpectrumTexture k = parameters.GetSpectrumTextureOrNull(
        "conductor.k", SpectrumType::Unbounded, alloc);
    SpectrumTexture reflectance =
        parameters.GetSpectrumTextureOrNull("reflectance", SpectrumType::Albedo, alloc);

    if (reflectance && (conductorEta || k))
        ErrorExit(loc, "For the coated conductor material, both \"reflectance\" "
                       "and \"eta\" and \"k\" can't be provided.");
    if (!reflectance) {
        if (!conductorEta)
            conductorEta = alloc.new_object<SpectrumConstantTexture>(
                GetNamedSpectrum("metal-Cu-eta"));
        if (!k)
            k = alloc.new_object<SpectrumConstantTexture>(GetNamedSpectrum("metal-Cu-k"));
    }

    int maxDepth = parameters.GetOneInt("maxdepth", 10);
    int nSamples = parameters.GetOneInt("nsamples", 1);

    FloatTexture g = parameters.GetFloatTexture("g", 0.f, alloc);
    SpectrumTexture albedo =
        parameters.GetSpectrumTexture("albedo", nullptr, SpectrumType::Albedo, alloc);
    if (!albedo)
        albedo = alloc.new_object<SpectrumConstantTexture>(
            alloc.new_object<ConstantSpectrum>(0.f));

    FloatTexture displacement = parameters.GetFloatTextureOrNull("displacement", alloc);
    bool remapRoughness = parameters.GetOneBool("remaproughness", true);

    return alloc.new_object<CoatedConductorMaterial>(
        interfaceURoughness, interfaceVRoughness, thickness, interfaceEta, g, albedo,
        conductorURoughness, conductorVRoughness, conductorEta, k, reflectance,
        displacement, normalMap, remapRoughness, maxDepth, nSamples);
}

// SubsurfaceMaterial Method Definitions
std::string SubsurfaceMaterial::ToString() const {
    return StringPrintf("[ SubsurfaceMaterial displacement: %s normalMap: %s scale: %f "
                        "sigma_a: %s sigma_s: %s reflectance: %s mfp: %s uRoughness: %s "
                        "vRoughness: %s scale: %f eta: %f remapRoughness: %s ]",
                        displacement, normalMap, scale, sigma_a, sigma_s, reflectance, mfp,
                        uRoughness, vRoughness, scale, eta, remapRoughness);
}

SubsurfaceMaterial *SubsurfaceMaterial::Create(
    const TextureParameterDictionary &parameters, Image *normalMap, const FileLoc *loc,
    Allocator alloc) {
    SpectrumTexture sigma_a, sigma_s, reflectance, mfp;

    Float g = parameters.GetOneFloat("g", 0.0f);

    // 4, mutually-exclusive, ways to specify the subsurface properties...
    std::string name = parameters.GetOneString("name", "");
    if (!name.empty()) {
        // 1. By name
        Spectrum sig_a, sig_s;
        if (!GetMediumScatteringProperties(name, &sig_a, &sig_s, alloc))
            ErrorExit(loc, "%s: named medium not found.", name);
        if (g != 0)
            Warning(loc, "Non-zero \"g\" ignored with named scattering coefficients.");
        g = 0; /* Enforce g=0 (the database specifies reduced scattering
                  coefficients) */
        sigma_a = alloc.new_object<SpectrumConstantTexture>(sig_a);
        sigma_s = alloc.new_object<SpectrumConstantTexture>(sig_s);
    } else {
        // 2. sigma_a and sigma_s directly specified
        sigma_a = parameters.GetSpectrumTextureOrNull("sigma_a", SpectrumType::Unbounded,
                                                      alloc);
        sigma_s = parameters.GetSpectrumTextureOrNull("sigma_s", SpectrumType::Unbounded,
                                                      alloc);
        if (sigma_a && !sigma_s)
            ErrorExit(loc, "Provided \"sigma_a\" parameter without \"sigma_s\".");
        if (sigma_s && !sigma_a)
            ErrorExit(loc, "Provided \"sigma_s\" parameter without \"sigma_a\".");

        if (!sigma_a && !sigma_s) {
            // 3. RGB/Spectrum, reflectance
            reflectance = parameters.GetSpectrumTextureOrNull(
                "reflectance", SpectrumType::Albedo, alloc);
            if (reflectance) {
                Spectrum one = alloc.new_object<ConstantSpectrum>(1.);
                mfp = parameters.GetSpectrumTexture("mfp", one, SpectrumType::Unbounded,
                                                    alloc);
            } else {
                // 4. nothing specified -- use defaults
                RGBUnboundedSpectrum *defaultSigma_a =
                    alloc.new_object<RGBUnboundedSpectrum>(*RGBColorSpace::sRGB,
                                                           RGB(.0011f, .0024f, .014f));
                RGBUnboundedSpectrum *defaultSigma_s =
                    alloc.new_object<RGBUnboundedSpectrum>(*RGBColorSpace::sRGB,
                                                           RGB(2.55f, 3.21f, 3.77f));
                sigma_a = alloc.new_object<SpectrumConstantTexture>(defaultSigma_a);
                sigma_s = alloc.new_object<SpectrumConstantTexture>(defaultSigma_s);
            }
        }
    }

    Float scale = parameters.GetOneFloat("scale", 1.f);
    Float eta = parameters.GetOneFloat("eta", 1.33f);

    FloatTexture uRoughness = parameters.GetFloatTextureOrNull("uroughness", alloc);
    FloatTexture vRoughness = parameters.GetFloatTextureOrNull("vroughness", alloc);
    if (!uRoughness)
        uRoughness = parameters.GetFloatTexture("roughness", 0.f, alloc);
    if (!vRoughness)
        vRoughness = parameters.GetFloatTexture("roughness", 0.f, alloc);

    FloatTexture displacement = parameters.GetFloatTextureOrNull("displacement", alloc);
    bool remapRoughness = parameters.GetOneBool("remaproughness", true);

    return alloc.new_object<SubsurfaceMaterial>(
        scale, sigma_a, sigma_s, reflectance, mfp, g, eta, uRoughness, vRoughness,
        displacement, normalMap, remapRoughness, alloc);
}

// DiffuseTransmissionMaterial Method Definitions
std::string DiffuseTransmissionMaterial::ToString() const {
    return StringPrintf("[ DiffuseTransmissionMaterial displacement: %s reflectance: %s "
                        "transmittance: %s scale: %f ]",
                        displacement, reflectance, transmittance, scale);
}

DiffuseTransmissionMaterial *DiffuseTransmissionMaterial::Create(
    const TextureParameterDictionary &parameters, Image *normalMap, const FileLoc *loc,
    Allocator alloc) {
    SpectrumTexture reflectance = parameters.GetSpectrumTexture(
        "reflectance", nullptr, SpectrumType::Albedo, alloc);
    if (!reflectance)
        reflectance = alloc.new_object<SpectrumConstantTexture>(
            alloc.new_object<ConstantSpectrum>(0.25f));

    SpectrumTexture transmittance = parameters.GetSpectrumTexture(
        "transmittance", nullptr, SpectrumType::Albedo, alloc);
    if (!transmittance)
        transmittance = alloc.new_object<SpectrumConstantTexture>(
            alloc.new_object<ConstantSpectrum>(0.25f));

    FloatTexture displacement = parameters.GetFloatTextureOrNull("displacement", alloc);
    Float scale = parameters.GetOneFloat("scale", 1.f);

    return alloc.new_object<DiffuseTransmissionMaterial>(reflectance, transmittance,
                                                         displacement, normalMap, scale);
}

MeasuredMaterial::MeasuredMaterial(const std::string &filename, FloatTexture displacement,
                                   Image *normalMap, Allocator alloc)
    : displacement(displacement), normalMap(normalMap) {
    brdf = MeasuredBxDF::BRDFDataFromFile(filename, alloc);
}

std::string MeasuredMaterial::ToString() const {
    return StringPrintf("[ MeasuredMaterial displacement: %s normalMap: %s ]",
                        displacement,
                        normalMap ? normalMap->ToString() : std::string("(nullptr)"));
}

MeasuredMaterial *MeasuredMaterial::Create(const TextureParameterDictionary &parameters,
                                           Image *normalMap, const FileLoc *loc,
                                           Allocator alloc) {
    std::string filename = ResolveFilename(parameters.GetOneString("filename", ""));
    if (filename.empty()) {
        Error("Filename must be provided for MeasuredMaterial");
        return nullptr;
    }
    FloatTexture displacement = parameters.GetFloatTextureOrNull("displacement", alloc);

    return alloc.new_object<MeasuredMaterial>(filename, displacement, normalMap, alloc);
}

std::string Material::ToString() const {
    if (!ptr())
        return "(nullptr)";

    auto toStr = [](auto ptr) { return ptr->ToString(); };
    return DispatchCPU(toStr);
}

STAT_COUNTER("Scene/Materials", nMaterialsCreated);

Material Material::Create(const std::string &name,
                          const TextureParameterDictionary &parameters, Image *normalMap,
                          /*const */ std::map<std::string, Material> &namedMaterials,
                          const FileLoc *loc, Allocator alloc) {
    Material material;
    if (name.empty() || name == "none") {
        Warning(loc, "Material \"%s\" is deprecated; use \"interface\" instead.",
                name.c_str());
        return nullptr;
    } else if (name == "interface")
        return nullptr;
    else if (name == "diffuse")
        material = DiffuseMaterial::Create(parameters, normalMap, loc, alloc);
    else if (name == "coateddiffuse")
        material = CoatedDiffuseMaterial::Create(parameters, normalMap, loc, alloc);
    else if (name == "coatedconductor")
        material = CoatedConductorMaterial::Create(parameters, normalMap, loc, alloc);
    else if (name == "diffusetransmission")
        material = DiffuseTransmissionMaterial::Create(parameters, normalMap, loc, alloc);
    else if (name == "dielectric")
        material = DielectricMaterial::Create(parameters, normalMap, loc, alloc);
    else if (name == "thindielectric")
        material = ThinDielectricMaterial::Create(parameters, normalMap, loc, alloc);
    else if (name == "hair")
        material = HairMaterial::Create(parameters, loc, alloc);
    else if (name == "conductor")
        material = ConductorMaterial::Create(parameters, normalMap, loc, alloc);
    else if (name == "measured")
        material = MeasuredMaterial::Create(parameters, normalMap, loc, alloc);
    else if (name == "subsurface")
        material = SubsurfaceMaterial::Create(parameters, normalMap, loc, alloc);
    else if (name == "mix") {
        std::vector<std::string> materialNames = parameters.GetStringArray("materials");
        if (materialNames.size() != 2)
            ErrorExit(
                "Must provide two values for \"string materials\" for mix material.");

        Material materials[2];
        for (int i = 0; i < 2; ++i) {
            auto iter = namedMaterials.find(materialNames[i]);
            if (iter == namedMaterials.end())
                ErrorExit("%s: named material not found.", materialNames[i]);
            materials[i] = iter->second;

            if (materials[i] == nullptr)
                ErrorExit("%s: an \"interface\" material cannot be used as an element of "
                          "the \"mix\" material.", materialNames[i]);
        }
        material = MixMaterial::Create(materials, parameters, loc, alloc);
    }
#ifndef USE_FALCOR_MATERIAL
    // *Add metalroughness material from RTX-DI
    else if (name == "metalroughness")
        material = DonutMaterial::Create(parameters, normalMap, loc, alloc);
#else
    // *Add Falcor material from RTX-PT
    else if (name == "falcor")
        material = DonutMaterial::Create(parameters, normalMap, loc, alloc);
#endif
    else
        ErrorExit(loc, "%s: material type unknown.", name);

    if (!material)
        ErrorExit(loc, "%s: unable to create material.", name);

    parameters.ReportUnused();
    ++nMaterialsCreated;
    return material;
}

}  // namespace pbrt
