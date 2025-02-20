// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/base/bxdf.h>
#include <pbrt/bxdfs.h>
#include <pbrt/cameras.h>
#include <pbrt/interaction.h>
#include <pbrt/materials.h>
#include <pbrt/options.h>
#include <pbrt/textures.h>
#include <pbrt/util/check.h>
#include <pbrt/util/containers.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/vecmath.h>
#include <pbrt/wavefront/integrator.h>

#include <type_traits>

namespace pbrt {

//// EvaluateMaterialCallback Definition
//struct EvaluateMaterialCallback {
//    int wavefrontDepth;
//    WavefrontPathIntegrator *integrator;
//    Transform movingFromCamera;
//    // EvaluateMaterialCallback Public Methods
//    template <typename ConcreteMaterial>
//    void operator()() {
//        if constexpr (!std::is_same_v<ConcreteMaterial, MixMaterial>)
//            integrator->EvaluateMaterialAndBSDF<ConcreteMaterial>(wavefrontDepth,
//                                                                  movingFromCamera);
//    }
//};

template <typename IntegratorT>
struct EvaluateMaterialCallback {
    int wavefrontDepth;
    IntegratorT *integrator;
    Transform movingFromCamera;
    // EvaluateMaterialCallback Public Methods
    template <typename ConcreteMaterial>
    void operator()() {
        if constexpr (!std::is_same_v<ConcreteMaterial, MixMaterial>)
            integrator->template EvaluateMaterialAndBSDF<ConcreteMaterial>(
                wavefrontDepth, movingFromCamera);
    }
};

// WavefrontPathIntegrator Surface Scattering Methods
void GWavefrontPathIntegrator::EvaluateMaterialsAndBSDFs(int wavefrontDepth,
                                                        Transform movingFromCamera) {
    ForEachType(EvaluateMaterialCallback<GWavefrontPathIntegrator>{wavefrontDepth, this,
                                                                   movingFromCamera},
                Material::Types());
}

template <typename ConcreteMaterial>
void GWavefrontPathIntegrator::EvaluateMaterialAndBSDF(int wavefrontDepth,
                                                      Transform movingFromCamera) {
    int index = Material::TypeIndex<ConcreteMaterial>();
    if (haveBasicEvalMaterial[index])
        EvaluateMaterialAndBSDF<ConcreteMaterial, BasicTextureEvaluator>(
            basicEvalMaterialQueue, movingFromCamera, wavefrontDepth);
    if (haveUniversalEvalMaterial[index])
        EvaluateMaterialAndBSDF<ConcreteMaterial, UniversalTextureEvaluator>(
            universalEvalMaterialQueue, movingFromCamera, wavefrontDepth);
}

template <typename ConcreteMaterial, typename TextureEvaluator>
void GWavefrontPathIntegrator::EvaluateMaterialAndBSDF(MaterialEvalQueue *evalQueue,
                                                      Transform movingFromCamera,
                                                      int wavefrontDepth) {
    // Get BSDF for items in _evalQueue_ and sample illumination
    // Construct _desc_ for material/texture evaluation kernel
    std::string desc = StringPrintf(
        "%s + BxDF eval (%s tex)", ConcreteMaterial::Name(),
        std::is_same_v<TextureEvaluator, BasicTextureEvaluator> ? "Basic" : "Universal");

    RayQueue *nextRayQueue = NextRayQueue(wavefrontDepth);
    auto queue = evalQueue->Get<MaterialEvalWorkItem<ConcreteMaterial>>();
    ForAllQueued(
        desc.c_str(), queue, maxQueueSize,
        PBRT_CPU_GPU_LAMBDA(const MaterialEvalWorkItem<ConcreteMaterial> w) {
            // Evaluate material and BSDF for ray intersection
            TextureEvaluator texEval;
            // Compute differentials for position and $(u,v)$ at intersection point
            Vector3f dpdx, dpdy;
            Float dudx = 0, dudy = 0, dvdx = 0, dvdy = 0;
            if (!GetOptions().disableTextureFiltering) {
                Point3f pc = movingFromCamera.ApplyInverse(Point3f(w.pi));
                Normal3f nc = movingFromCamera.ApplyInverse(w.n);
                camera.Approximate_dp_dxy(pc, nc, w.time, samplesPerPixel, &dpdx, &dpdy);
                Vector3f dpdu = w.dpdu, dpdv = w.dpdv;
                // Estimate screen-space change in $(u,v)$
                // Compute $\transpose{\XFORM{A}} \XFORM{A}$ and its determinant
                Float ata00 = Dot(dpdu, dpdu), ata01 = Dot(dpdu, dpdv);
                Float ata11 = Dot(dpdv, dpdv);
                Float invDet = 1 / DifferenceOfProducts(ata00, ata11, ata01, ata01);
                invDet = IsFinite(invDet) ? invDet : 0.f;

                // Compute $\transpose{\XFORM{A}} \VEC{b}$ for $x$ and $y$
                Float atb0x = Dot(dpdu, dpdx), atb1x = Dot(dpdv, dpdx);
                Float atb0y = Dot(dpdu, dpdy), atb1y = Dot(dpdv, dpdy);

                // Compute $u$ and $v$ derivatives with respect to $x$ and $y$
                dudx = DifferenceOfProducts(ata11, atb0x, ata01, atb1x) * invDet;
                dvdx = DifferenceOfProducts(ata00, atb1x, ata01, atb0x) * invDet;
                dudy = DifferenceOfProducts(ata11, atb0y, ata01, atb1y) * invDet;
                dvdy = DifferenceOfProducts(ata00, atb1y, ata01, atb0y) * invDet;

                // Clamp derivatives of $u$ and $v$ to reasonable values
                dudx = IsFinite(dudx) ? Clamp(dudx, -1e8f, 1e8f) : 0.f;
                dvdx = IsFinite(dvdx) ? Clamp(dvdx, -1e8f, 1e8f) : 0.f;
                dudy = IsFinite(dudy) ? Clamp(dudy, -1e8f, 1e8f) : 0.f;
                dvdy = IsFinite(dvdy) ? Clamp(dvdy, -1e8f, 1e8f) : 0.f;
            }

            // Compute shading normal if bump or normal mapping is being used
            Normal3f ns = w.ns;
            Vector3f dpdus = w.dpdus;
            FloatTexture displacement = w.material->GetDisplacement();
            const Image *normalMap = w.material->GetNormalMap();
            if (normalMap) {
                // Call _NormalMap()_ to find shading geometry
                NormalBumpEvalContext bctx =
                    w.GetNormalBumpEvalContext(dudx, dudy, dvdx, dvdy);
                Vector3f dpdvs;
                NormalMap(*normalMap, bctx, &dpdus, &dpdvs);
                ns = Normal3f(Normalize(Cross(dpdus, dpdvs)));
                ns = FaceForward(ns, w.n);

            } else if (displacement) {
                // Call _BumpMap()_ to find shading geometry
                if (displacement)
                    DCHECK(texEval.CanEvaluate({displacement}, {}));
                NormalBumpEvalContext bctx =
                    w.GetNormalBumpEvalContext(dudx, dudy, dvdx, dvdy);
                Vector3f dpdvs;
                BumpMap(texEval, displacement, bctx, &dpdus, &dpdvs);
                ns = Normal3f(Normalize(Cross(dpdus, dpdvs)));
                ns = FaceForward(ns, w.n);
            }

            // Get BSDF at intersection point
            SampledWavelengths lambda = w.lambda;
            MaterialEvalContext ctx =
                w.GetMaterialEvalContext(dudx, dudy, dvdx, dvdy, ns, dpdus);
            using ConcreteBxDF = typename ConcreteMaterial::BxDF;
            ConcreteBxDF bxdf = w.material->GetBxDF(texEval, ctx, lambda);
            BSDF bsdf(ctx.ns, ctx.dpdus, &bxdf);
            // Handle terminated secondary wavelengths after BSDF creation
            if (lambda.SecondaryTerminated())
                pixelSampleState.lambda[w.pixelIndex] = lambda;

            // Regularize BSDF, if appropriate
            if (regularize && w.anyNonSpecularBounces)
                bsdf.Regularize();

            // Initialize _VisibleSurface_ at first intersection if necessary
            if (w.depth == 0 && initializeVisibleSurface) {
                SurfaceInteraction isect;
                isect.pi = w.pi;
                isect.n = w.n;
                isect.shading.n = ns;
                isect.uv = w.uv;
                isect.wo = w.wo;
                isect.time = w.time;
                isect.dpdx = dpdx;
                isect.dpdy = dpdy;

                // Estimate BSDF's albedo
                // Define sample arrays _ucRho_ and _uRho_ for reflectance estimate
                constexpr int nRhoSamples = 16;
                const Float ucRho[nRhoSamples] = {
                    0.75741637, 0.37870818, 0.7083487, 0.18935409, 0.9149363, 0.35417435,
                    0.5990858,  0.09467703, 0.8578725, 0.45746812, 0.686759,  0.17708716,
                    0.9674518,  0.2995429,  0.5083201, 0.047338516};
                const Point2f uRho[nRhoSamples] = {
                    Point2f(0.855985, 0.570367), Point2f(0.381823, 0.851844),
                    Point2f(0.285328, 0.764262), Point2f(0.733380, 0.114073),
                    Point2f(0.542663, 0.344465), Point2f(0.127274, 0.414848),
                    Point2f(0.964700, 0.947162), Point2f(0.594089, 0.643463),
                    Point2f(0.095109, 0.170369), Point2f(0.825444, 0.263359),
                    Point2f(0.429467, 0.454469), Point2f(0.244460, 0.816459),
                    Point2f(0.756135, 0.731258), Point2f(0.516165, 0.152852),
                    Point2f(0.180888, 0.214174), Point2f(0.898579, 0.503897)};

                SampledSpectrum albedo = bsdf.rho(isect.wo, ucRho, uRho);

                pixelSampleState.visibleSurface[w.pixelIndex] =
                    VisibleSurface(isect, albedo, lambda);
            }

            // Sample BSDF and enqueue indirect ray at intersection point
            Vector3f wo = w.wo;
            RaySamples raySamples = pixelSampleState.samples[w.pixelIndex];
            pstd::optional<BSDFSample> bsdfSample = bsdf.Sample_f<ConcreteBxDF>(
                wo, raySamples.indirect.uc, raySamples.indirect.u);
            if (bsdfSample) {
                // Compute updated path throughput and PDFs and enqueue indirect ray
                Vector3f wi = bsdfSample->wi;
                SampledSpectrum beta =
                    w.beta * bsdfSample->f * AbsDot(wi, ns) / bsdfSample->pdf;
                SampledSpectrum r_u = w.r_u, r_l;

                PBRT_DBG("%s f*cos[0] %f bsdfSample->pdf %f f*cos/pdf %f\n",
                         ConcreteBxDF::Name(), bsdfSample->f[0] * AbsDot(wi, ns),
                         bsdfSample->pdf,
                         bsdfSample->f[0] * AbsDot(wi, ns) / bsdfSample->pdf);

                // Update _r_u_ based on BSDF sample PDF
                if (bsdfSample->pdfIsProportional)
                    r_l = r_u / bsdf.PDF<ConcreteBxDF>(wo, bsdfSample->wi);
                else
                    r_l = r_u / bsdfSample->pdf;

                // Update _etaScale_ accounting for BSDF scattering
                Float etaScale = w.etaScale;
                if (bsdfSample->IsTransmission())
                    etaScale *= Sqr(bsdfSample->eta);

                // Apply Russian roulette to indirect ray based on weighted path
                // throughput
                SampledSpectrum rrBeta = beta * etaScale / r_u.Average();
                // Note: depth >= 1 here to match VolPathIntegrator (which increments
                // depth earlier).
                if (rrBeta.MaxComponentValue() < 1 && w.depth >= 1) {
                    Float q = std::max<Float>(0, 1 - rrBeta.MaxComponentValue());
                    if (raySamples.indirect.rr < q) {
                        beta = SampledSpectrum(0.f);
                        PBRT_DBG("Path terminated with RR\n");
                    } else
                        beta /= 1 - q;
                }

                if (beta) {
                    // Initialize spawned ray and enqueue for next ray depth
                    if (bsdfSample->IsTransmission() &&
                        w.material->HasSubsurfaceScattering()) {
                        bssrdfEvalQueue->Push(w.material, lambda, beta, r_u,
                                              Point3f(w.pi), wo, w.n, ns, dpdus, w.uv,
                                              w.depth, w.mediumInterface, etaScale,
                                              w.pixelIndex);
                    } else {
                        Ray ray = SpawnRay(w.pi, w.n, w.time, wi);
                        // Initialize _ray_ medium if media are present
                        if (haveMedia)
                            ray.medium = Dot(ray.d, w.n) > 0 ? w.mediumInterface.outside
                                                             : w.mediumInterface.inside;

                        bool anyNonSpecularBounces =
                            !bsdfSample->IsSpecular() || w.anyNonSpecularBounces;
                        // NOTE: slightly different than context below. Problem?
                        LightSampleContext ctx(w.pi, w.n, ns);
                        nextRayQueue->PushIndirectRay(
                            ray, w.depth + 1, ctx, beta, r_u, r_l, lambda, etaScale,
                            bsdfSample->IsSpecular(), anyNonSpecularBounces,
                            w.pixelIndex);

                        PBRT_DBG("Spawned indirect ray at depth %d from w.index %d. "
                                 "Specular %d beta %f %f %f %f r_u %f %f %f %f r_l %f "
                                 "%f %f %f beta/r_u %f %f %f %f\n",
                                 w.depth + 1, w.pixelIndex, int(bsdfSample->IsSpecular()),
                                 beta[0], beta[1], beta[2], beta[3], r_u[0], r_u[1],
                                 r_u[2], r_u[3], r_l[0], r_l[1], r_l[2], r_l[3],
                                 SafeDiv(beta, r_u)[0], SafeDiv(beta, r_u)[1],
                                 SafeDiv(beta, r_u)[2], SafeDiv(beta, r_u)[3]);
                    }
                }
            }

            // Sample light and enqueue shadow ray at intersection point
            BxDFFlags flags = bsdf.Flags();
            if (IsNonSpecular(flags)) {
                // Choose a light source using the _LightSampler_
                LightSampleContext ctx(w.pi, w.n, ns);
                if (IsReflective(flags) && !IsTransmissive(flags))
                    ctx.pi = OffsetRayOrigin(ctx.pi, w.n, wo);
                else if (IsTransmissive(flags) && IsReflective(flags))
                    ctx.pi = OffsetRayOrigin(ctx.pi, w.n, -wo);
                pstd::optional<SampledLight> sampledLight =
                    lightSampler.Sample(ctx, raySamples.direct.uc);
                if (!sampledLight)
                    return;
                Light light = sampledLight->light;

                // Sample light source and evaluate BSDF for direct lighting
                pstd::optional<LightLiSample> ls =
                    light.SampleLi(ctx, raySamples.direct.u, lambda, true);
                if (!ls || !ls->L || ls->pdf == 0)
                    return;
                Vector3f wi = ls->wi;
                SampledSpectrum f = bsdf.f<ConcreteBxDF>(wo, wi);
                if (!f)
                    return;

                // Compute path throughput and path PDFs for light sample
                SampledSpectrum beta = w.beta * f * AbsDot(wi, ns);
                PBRT_DBG("w.beta %f %f %f %f f %f %f %f %f dot %f\n", w.beta[0],
                         w.beta[1], w.beta[2], w.beta[3], f[0], f[1], f[2], f[3],
                         AbsDot(wi, ns));

                PBRT_DBG(
                    "me index %d depth %d beta %f %f %f %f f %f %f %f %f ls.L %f %f %f "
                    "%f ls.pdf %f\n",
                    w.pixelIndex, w.depth, beta[0], beta[1], beta[2], beta[3], f[0], f[1],
                    f[2], f[3], ls->L[0], ls->L[1], ls->L[2], ls->L[3], ls->pdf);

                Float lightPDF = ls->pdf * sampledLight->p;
                // This causes r_u to be zero for the shadow ray, so that
                // part of MIS just becomes a no-op.
                Float bsdfPDF =
                    IsDeltaLight(light.Type()) ? 0.f : bsdf.PDF<ConcreteBxDF>(wo, wi);
                SampledSpectrum r_u = w.r_u * bsdfPDF;
                SampledSpectrum r_l = w.r_u * lightPDF;

                // Enqueue shadow ray with tentative radiance contribution
                SampledSpectrum Ld = beta * ls->L;
                Ray ray = SpawnRayTo(w.pi, w.n, w.time, ls->pLight.pi, ls->pLight.n);
                // Initialize _ray_ medium if media are present
                if (haveMedia)
                    ray.medium = Dot(ray.d, w.n) > 0 ? w.mediumInterface.outside
                                                     : w.mediumInterface.inside;

                shadowRayQueue->Push(ShadowRayWorkItem{ray, 1 - ShadowEpsilon, lambda, Ld,
                                                       r_u, r_l, w.pixelIndex});

                PBRT_DBG("w.index %d spawned shadow ray depth %d Ld %f %f %f %f "
                         "new beta %f %f %f %f beta/uni %f %f %f %f Ld/uni %f %f %f %f\n",
                         w.pixelIndex, w.depth, Ld[0], Ld[1], Ld[2], Ld[3], beta[0],
                         beta[1], beta[2], beta[3], SafeDiv(beta, r_u)[0],
                         SafeDiv(beta, r_u)[1], SafeDiv(beta, r_u)[2],
                         SafeDiv(beta, r_u)[3], SafeDiv(Ld, r_u)[0], SafeDiv(Ld, r_u)[1],
                         SafeDiv(Ld, r_u)[2], SafeDiv(Ld, r_u)[3]);
            }
        });
}

// WavefrontPathIntegrator Surface Scattering Methods
void ReSTIRDIWavefrontPathIntegrator::EvaluateMaterialsAndBSDFs(
    int wavefrontDepth,
                                                        Transform movingFromCamera) {
    ForEachType(EvaluateMaterialCallback<ReSTIRDIWavefrontPathIntegrator>{wavefrontDepth, this,
                                                                  movingFromCamera},
                Material::Types());
}

template <typename ConcreteMaterial>
void ReSTIRDIWavefrontPathIntegrator::EvaluateMaterialAndBSDF(
    int wavefrontDepth,
                                                      Transform movingFromCamera) {
    int index = Material::TypeIndex<ConcreteMaterial>();
    if (haveBasicEvalMaterial[index])
        EvaluateMaterialAndBSDF<ConcreteMaterial, BasicTextureEvaluator>(
            basicEvalMaterialQueue, movingFromCamera, wavefrontDepth);
    if (haveUniversalEvalMaterial[index])
        EvaluateMaterialAndBSDF<ConcreteMaterial, UniversalTextureEvaluator>(
            universalEvalMaterialQueue, movingFromCamera, wavefrontDepth);
}

template <typename ConcreteMaterial, typename TextureEvaluator>
void ReSTIRDIWavefrontPathIntegrator::EvaluateMaterialAndBSDF(
    MaterialEvalQueue *evalQueue,
                                                      Transform movingFromCamera,
                                                      int wavefrontDepth) {
    LOG_VERBOSE(
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
    // Get BSDF for items in _evalQueue_ and sample illumination
    // Construct _desc_ for material/texture evaluation kernel
    std::string desc = StringPrintf(
        "%s + BxDF eval (%s tex)", ConcreteMaterial::Name(),
        std::is_same_v<TextureEvaluator, BasicTextureEvaluator> ? "Basic" : "Universal");

    RayQueue *nextRayQueue = NextRayQueue(wavefrontDepth);
    auto queue = evalQueue->Get<MaterialEvalWorkItem<ConcreteMaterial>>();
    ForAllQueued(
        desc.c_str(), queue, maxQueueSize,
        PBRT_CPU_GPU_LAMBDA(const MaterialEvalWorkItem<ConcreteMaterial> w) {
            Vector2i resolution = film.PixelBounds().Diagonal();
            Point2i pixel = pixelSampleState.pPixel[w.pixelIndex];
            int curPixelNumber = pixel.y * resolution.x + pixel.x;

            // Evaluate material and BSDF for ray intersection
            TextureEvaluator texEval;
            // Compute differentials for position and $(u,v)$ at intersection point
            Vector3f dpdx, dpdy;
            Float dudx = 0, dudy = 0, dvdx = 0, dvdy = 0;
            if (!GetOptions().disableTextureFiltering) {
                Point3f pc = movingFromCamera.ApplyInverse(Point3f(w.pi));
                Normal3f nc = movingFromCamera.ApplyInverse(w.n);
                camera.Approximate_dp_dxy(pc, nc, w.time, samplesPerPixel, &dpdx, &dpdy);
                Vector3f dpdu = w.dpdu, dpdv = w.dpdv;
                // Estimate screen-space change in $(u,v)$
                // Compute $\transpose{\XFORM{A}} \XFORM{A}$ and its determinant
                Float ata00 = Dot(dpdu, dpdu), ata01 = Dot(dpdu, dpdv);
                Float ata11 = Dot(dpdv, dpdv);
                Float invDet = 1 / DifferenceOfProducts(ata00, ata11, ata01, ata01);
                invDet = IsFinite(invDet) ? invDet : 0.f;

                // Compute $\transpose{\XFORM{A}} \VEC{b}$ for $x$ and $y$
                Float atb0x = Dot(dpdu, dpdx), atb1x = Dot(dpdv, dpdx);
                Float atb0y = Dot(dpdu, dpdy), atb1y = Dot(dpdv, dpdy);

                // Compute $u$ and $v$ derivatives with respect to $x$ and $y$
                dudx = DifferenceOfProducts(ata11, atb0x, ata01, atb1x) * invDet;
                dvdx = DifferenceOfProducts(ata00, atb1x, ata01, atb0x) * invDet;
                dudy = DifferenceOfProducts(ata11, atb0y, ata01, atb1y) * invDet;
                dvdy = DifferenceOfProducts(ata00, atb1y, ata01, atb0y) * invDet;

                // Clamp derivatives of $u$ and $v$ to reasonable values
                dudx = IsFinite(dudx) ? Clamp(dudx, -1e8f, 1e8f) : 0.f;
                dvdx = IsFinite(dvdx) ? Clamp(dvdx, -1e8f, 1e8f) : 0.f;
                dudy = IsFinite(dudy) ? Clamp(dudy, -1e8f, 1e8f) : 0.f;
                dvdy = IsFinite(dvdy) ? Clamp(dvdy, -1e8f, 1e8f) : 0.f;
            }

            // Compute shading normal if bump or normal mapping is being used
            Normal3f ns = w.ns;
            Vector3f dpdus = w.dpdus;
            FloatTexture displacement = w.material->GetDisplacement();
            const Image *normalMap = w.material->GetNormalMap();
            if (normalMap) {
                // Call _NormalMap()_ to find shading geometry
                NormalBumpEvalContext bctx =
                    w.GetNormalBumpEvalContext(dudx, dudy, dvdx, dvdy);
                Vector3f dpdvs;
                NormalMap(*normalMap, bctx, &dpdus, &dpdvs);
                ns = Normal3f(Normalize(Cross(dpdus, dpdvs)));
                ns = FaceForward(ns, w.n);

            } else if (displacement) {
                // Call _BumpMap()_ to find shading geometry
                if (displacement)
                    DCHECK(texEval.CanEvaluate({displacement}, {}));
                NormalBumpEvalContext bctx =
                    w.GetNormalBumpEvalContext(dudx, dudy, dvdx, dvdy);
                Vector3f dpdvs;
                BumpMap(texEval, displacement, bctx, &dpdus, &dpdvs);
                ns = Normal3f(Normalize(Cross(dpdus, dpdvs)));
                ns = FaceForward(ns, w.n);
            }

            // Get BSDF at intersection point
            SampledWavelengths lambda = w.lambda;
            MaterialEvalContext ctx =
                w.GetMaterialEvalContext(dudx, dudy, dvdx, dvdy, ns, dpdus);
            using ConcreteBxDF = typename ConcreteMaterial::BxDF;
            ConcreteBxDF bxdf = w.material->GetBxDF(texEval, ctx, lambda);
            BSDF bsdf(ctx.ns, ctx.dpdus, &bxdf);
            // Handle terminated secondary wavelengths after BSDF creation
            if (lambda.SecondaryTerminated())
                pixelSampleState.lambda[w.pixelIndex] = lambda;

            // Regularize BSDF, if appropriate
            if (regularize && w.anyNonSpecularBounces)
                bsdf.Regularize();

            // Initialize _VisibleSurface_ at first intersection if necessary
            if (w.depth == 0 && initializeVisibleSurface) {
                SurfaceInteraction isect;
                isect.pi = w.pi;
                isect.n = w.n;
                isect.shading.n = ns;
                isect.uv = w.uv;
                isect.wo = w.wo;
                isect.time = w.time;
                isect.dpdx = dpdx;
                isect.dpdy = dpdy;

                // Estimate BSDF's albedo
                // Define sample arrays _ucRho_ and _uRho_ for reflectance estimate
                constexpr int nRhoSamples = 16;
                const Float ucRho[nRhoSamples] = {
                    0.75741637, 0.37870818, 0.7083487, 0.18935409, 0.9149363, 0.35417435,
                    0.5990858,  0.09467703, 0.8578725, 0.45746812, 0.686759,  0.17708716,
                    0.9674518,  0.2995429,  0.5083201, 0.047338516};
                const Point2f uRho[nRhoSamples] = {
                    Point2f(0.855985, 0.570367), Point2f(0.381823, 0.851844),
                    Point2f(0.285328, 0.764262), Point2f(0.733380, 0.114073),
                    Point2f(0.542663, 0.344465), Point2f(0.127274, 0.414848),
                    Point2f(0.964700, 0.947162), Point2f(0.594089, 0.643463),
                    Point2f(0.095109, 0.170369), Point2f(0.825444, 0.263359),
                    Point2f(0.429467, 0.454469), Point2f(0.244460, 0.816459),
                    Point2f(0.756135, 0.731258), Point2f(0.516165, 0.152852),
                    Point2f(0.180888, 0.214174), Point2f(0.898579, 0.503897)};

                SampledSpectrum albedo = bsdf.rho(isect.wo, ucRho, uRho);

                pixelSampleState.visibleSurface[w.pixelIndex] =
                    VisibleSurface(isect, albedo, lambda);
            }

            // Sample BSDF and enqueue indirect ray at intersection point
            Vector3f wo = w.wo;
            RaySamples raySamples = pixelSampleState.samples[w.pixelIndex];
            pstd::optional<BSDFSample> bsdfSample = bsdf.Sample_f<ConcreteBxDF>(
                wo, raySamples.indirect.uc, raySamples.indirect.u);
            if (bsdfSample) {
                // Compute updated path throughput and PDFs and enqueue indirect ray
                Vector3f wi = bsdfSample->wi;
                SampledSpectrum beta =
                    w.beta * bsdfSample->f * AbsDot(wi, ns) / bsdfSample->pdf;
                SampledSpectrum r_u = w.r_u, r_l;

                PBRT_DBG("%s f*cos[0] %f bsdfSample->pdf %f f*cos/pdf %f\n",
                         ConcreteBxDF::Name(), bsdfSample->f[0] * AbsDot(wi, ns),
                         bsdfSample->pdf,
                         bsdfSample->f[0] * AbsDot(wi, ns) / bsdfSample->pdf);

                // Update _r_u_ based on BSDF sample PDF
                if (bsdfSample->pdfIsProportional)
                    r_l = r_u / bsdf.PDF<ConcreteBxDF>(wo, bsdfSample->wi);
                else
                    r_l = r_u / bsdfSample->pdf;

                // Update _etaScale_ accounting for BSDF scattering
                Float etaScale = w.etaScale;
                if (bsdfSample->IsTransmission())
                    etaScale *= Sqr(bsdfSample->eta);

                // Apply Russian roulette to indirect ray based on weighted path
                // throughput
                SampledSpectrum rrBeta = beta * etaScale / r_u.Average();
                // Note: depth >= 1 here to match VolPathIntegrator (which increments
                // depth earlier).
                if (rrBeta.MaxComponentValue() < 1 && w.depth >= 1) {
                    Float q = std::max<Float>(0, 1 - rrBeta.MaxComponentValue());
                    if (raySamples.indirect.rr < q) {
                        beta = SampledSpectrum(0.f);
                        PBRT_DBG("Path terminated with RR\n");
                    } else
                        beta /= 1 - q;
                }

                if (beta) {
                    // Initialize spawned ray and enqueue for next ray depth
                    if (bsdfSample->IsTransmission() &&
                        w.material->HasSubsurfaceScattering()) {
                        bssrdfEvalQueue->Push(w.material, lambda, beta, r_u,
                                              Point3f(w.pi), wo, w.n, ns, dpdus, w.uv,
                                              w.depth, w.mediumInterface, etaScale,
                                              w.pixelIndex);
                    } else {
                        Ray ray = SpawnRay(w.pi, w.n, w.time, wi);
                        // Initialize _ray_ medium if media are present
                        if (haveMedia)
                            ray.medium = Dot(ray.d, w.n) > 0 ? w.mediumInterface.outside
                                                             : w.mediumInterface.inside;

                        bool anyNonSpecularBounces =
                            !bsdfSample->IsSpecular() || w.anyNonSpecularBounces;
                        // NOTE: slightly different than context below. Problem?
                        LightSampleContext ctx(w.pi, w.n, ns);
                        nextRayQueue->PushIndirectRay(
                            ray, w.depth + 1, ctx, beta, r_u, r_l, lambda, etaScale,
                            bsdfSample->IsSpecular(), anyNonSpecularBounces,
                            w.pixelIndex);

                        PBRT_DBG("Spawned indirect ray at depth %d from w.index %d. "
                                 "Specular %d beta %f %f %f %f r_u %f %f %f %f r_l %f "
                                 "%f %f %f beta/r_u %f %f %f %f\n",
                                 w.depth + 1, w.pixelIndex, int(bsdfSample->IsSpecular()),
                                 beta[0], beta[1], beta[2], beta[3], r_u[0], r_u[1],
                                 r_u[2], r_u[3], r_l[0], r_l[1], r_l[2], r_l[3],
                                 SafeDiv(beta, r_u)[0], SafeDiv(beta, r_u)[1],
                                 SafeDiv(beta, r_u)[2], SafeDiv(beta, r_u)[3]);
                    }
                }
            }

            auto spectrumToLuminance = [](SampledSpectrum ss, SampledWavelengths lambda) {
                XYZ xyz = ss.ToXYZ(lambda);
                constexpr double matrix[3][3] = {{3.2406, -1.5372, -0.4986},
                                                 {-0.9689, 1.8758, 0.0415},
                                                 {0.0557, -0.2040, 1.0570}};
                double linear_r =
                    matrix[0][0] * xyz.X + matrix[0][1] * xyz.Y + matrix[0][2] * xyz.Z;
                double linear_g =
                    matrix[1][0] * xyz.X + matrix[1][1] * xyz.Y + matrix[1][2] * xyz.Z;
                double linear_b =
                    matrix[2][0] * xyz.X + matrix[2][1] * xyz.Y + matrix[2][2] * xyz.Z;
                RGB rgb(linear_r, linear_g, linear_b);
                // RGB rgb = RGBColorSpace_sRGB->ToRGB(xyz);
                rgb = ClampZero(rgb);
                float l = rgb.r * 0.299f + rgb.g * 0.587f + rgb.b * 0.114f;
                return l;
            };

            auto rng1D = [](Float seed) {
                // 將浮點數轉換為整數表示
                union {
                    float f;
                    uint32_t i;
                } u;
                u.f = seed;

                // 簡單的整數雜湊函數
                uint32_t hash = u.i;
                hash = ((hash >> 16) ^ hash) * 0x45d9f3b;
                hash = ((hash >> 16) ^ hash) * 0x45d9f3b;
                hash = (hash >> 16) ^ hash;

                // 轉換到 0-1 範圍
                Float rng = static_cast<float>(hash) / static_cast<float>(UINT32_MAX);
                return rng;
            };

            auto addSampleToShadowRayReservoir = [&](DIReservoir reservoir, int sampleNumber) {
                // Sample light and enqueue shadow ray at intersection point
                BxDFFlags flags = bsdf.Flags();
                if (IsNonSpecular(flags)) {
                    // Choose a light source using the _LightSampler_
                    LightSampleContext ctx(w.pi, w.n, ns);
                    if (IsReflective(flags) && !IsTransmissive(flags))
                        ctx.pi = OffsetRayOrigin(ctx.pi, w.n, wo);
                    else if (IsTransmissive(flags) && IsReflective(flags))
                        ctx.pi = OffsetRayOrigin(ctx.pi, w.n, -wo);

                    for (int risCount = 0; risCount < sampleNumber; ++risCount)
                    {
                        Float seed1 = w.time + w.pixelIndex + w.depth + w.n.x + w.n.y +
                                      w.n.z + w.wo.x + w.wo.y + w.wo.z + risCount + raySamples.direct.uc;
                        Float seed2 = w.time + w.pixelIndex + w.depth + w.n.x + w.n.y +
                                      w.n.z + w.wo.x + w.wo.y + w.wo.z + risCount + raySamples.direct.u.x;
                        Float seed3 = w.time + w.pixelIndex + w.depth + w.n.x + w.n.y +
                                      w.n.z + w.wo.x + w.wo.y + w.wo.z + risCount + raySamples.direct.u.y;
                        pstd::optional<SampledLight> sampledLight =
                            lightSampler.Sample(ctx, rng1D(seed1));
                        if (!sampledLight) {
                            continue;
                        }
                        Light light = sampledLight->light;

                        // Sample light source and evaluate BSDF for direct lighting
                        pstd::optional<LightLiSample> ls =
                            light.SampleLi(ctx, Point2f(rng1D(seed2), rng1D(seed3)), lambda, true);

                        if (!ls || !ls->L || ls->pdf == 0) {
                            continue;
                        }
                        Vector3f wi = ls->wi;
                        SampledSpectrum f = bsdf.f<ConcreteBxDF>(wo, wi);
                        if (!f) {
                            continue;
                        }

                        // Compute path throughput and path PDFs for light sample
                        SampledSpectrum beta = w.beta * f * AbsDot(wi, ns);
                        PBRT_DBG("w.beta %f %f %f %f f %f %f %f %f dot %f\n", w.beta[0],
                                 w.beta[1], w.beta[2], w.beta[3], f[0], f[1], f[2], f[3],
                                 AbsDot(wi, ns));

                        PBRT_DBG(
                            "me index %d depth %d beta %f %f %f %f f %f %f %f %f ls.L "
                            "%f %f %f "
                            "%f ls.pdf %f\n",
                            w.pixelIndex, w.depth, beta[0], beta[1], beta[2], beta[3],
                            f[0], f[1], f[2], f[3], ls->L[0], ls->L[1], ls->L[2],
                            ls->L[3], ls->pdf);

                        Float lightPDF = ls->pdf * sampledLight->p;
                        // This causes r_u to be zero for the shadow ray, so that
                        // part of MIS just becomes a no-op.
                        Float bsdfPDF = IsDeltaLight(light.Type())
                                            ? 0.f
                                            : bsdf.PDF<ConcreteBxDF>(wo, wi);
                        SampledSpectrum r_u = w.r_u * bsdfPDF;
                        SampledSpectrum r_l = w.r_u * lightPDF;

                        // Enqueue shadow ray with tentative radiance contribution
                        SampledSpectrum Ld = beta * ls->L;

                        // Direct lighting and restir di
                        // RIS source PDF
                        Float source_p = lightPDF;

                        ////// RIS target PDF
                        Float target_p = 0.0f;

                        ////// RIS f(x)
                        SampledSpectrum w_ld = ClampZero(ls->L) * f;

                        target_p = spectrumToLuminance(w_ld, lambda);

                        float curM = reservoir.M;
                        float curWeight = spectrumToLuminance(Ld, lambda);
                        float curWeightSum = reservoir.weightSum;
                        float chooseRate = curWeight / curWeightSum;

                        Float seed4 = w.time + w.pixelIndex + w.depth + w.n.x + w.n.y +
                                      w.n.z + w.wo.x + w.wo.y + w.wo.z;
                        Float rng = rng1D(seed4);
                        reservoir.update(ls, rng, curWeight, 1, target_p, sampledLight->p,
                                         w.n, w.depth);
                    }
                    reservoir.updateWeight();
                }
                return reservoir;
            };

            auto temporalReuse = [&](DIReservoir reservoir) {
                // Sample light and enqueue shadow ray at intersection point
                BxDFFlags flags = bsdf.Flags();
                if (IsNonSpecular(flags)) {
                    DIReservoir lastFrameReservoir = imageState.diReservoir[curPixelNumber];

                    if (lastFrameReservoir.M == 0)
                        return reservoir;

                    pstd::optional<LightLiSample> ls = lastFrameReservoir.ls;
                    if (!ls || !ls->L || ls->pdf == 0)
                        return reservoir;

                    Float sampledLightP = lastFrameReservoir.sampledLightP;
                    Vector3f wi = ls->wi;
                    Vector3f wo = w.wo;
                    Normal3f normal = lastFrameReservoir.normal;
                    Float depth = lastFrameReservoir.depth;

                    // 檢查深度差異
                    float depth_diff = std::abs(depth - w.depth);
                    if (depth_diff > 0.001 * depth) {
                        Float exitAt1 = imageState.exitAt1[curPixelNumber];
                        exitAt1++;
                        imageState.exitAt1[curPixelNumber] = exitAt1;
                        return reservoir;
                    }

                    // 檢查法線差異
                    float normal_similarity = pbrt::Dot(normal, w.n);
                    if (normal_similarity < 0.9f) {
                        Float exitAt2 = imageState.exitAt2[curPixelNumber];
                        exitAt2++;
                        imageState.exitAt2[curPixelNumber] = exitAt2;
                        return reservoir;
                    }

                    SampledSpectrum f = bsdf.f<ConcreteBxDF>(wo, wi);
                    if (!f) {
                        Float exitAt3 = imageState.exitAt3[curPixelNumber];
                        exitAt3++;
                        imageState.exitAt3[curPixelNumber] = exitAt3;
                        return reservoir;
                    }

                    // Compute path throughput and path PDFs for light sample
                    SampledSpectrum beta = w.beta * f * AbsDot(wi, ns);
                    PBRT_DBG("w.beta %f %f %f %f f %f %f %f %f dot %f\n",
                                w.beta[0], w.beta[1], w.beta[2], w.beta[3], f[0],
                                f[1], f[2], f[3], AbsDot(wi, ns));

                    PBRT_DBG("me index %d depth %d beta %f %f %f %f f %f %f %f "
                                "%f ls.L "
                                "%f %f %f "
                                "%f ls.pdf %f\n",
                                w.pixelIndex, w.depth, beta[0], beta[1], beta[2],
                                beta[3], f[0], f[1], f[2], f[3], ls->L[0], ls->L[1],
                                ls->L[2], ls->L[3], ls->pdf);

                    Float lightPDF = ls->pdf * sampledLightP;
                    // This causes r_u to be zero for the shadow ray, so that
                    // part of MIS just becomes a no-op.

                    // TODO  use real light type
                    // Float bsdfPDF = IsDeltaLight(light.Type()) ? 0.f :
                    // bsdf.PDF<ConcreteBxDF>(wo, wi);
                    Float bsdfPDF = bsdf.PDF<ConcreteBxDF>(wo, wi);
                    SampledSpectrum r_u = w.r_u * bsdfPDF;
                    SampledSpectrum r_l = w.r_u * lightPDF;

                    // Enqueue shadow ray with tentative radiance contribution
                    SampledSpectrum Ld = beta * ls->L;

                    // Direct lighting and restir di
                    // RIS source PDF
                    Float source_p = lightPDF;

                    ////// RIS target PDF
                    Float target_p = 0.0f;

                    ////// RIS f(x)
                    SampledSpectrum w_ld = ClampZero(ls->L) * f;

                    target_p = spectrumToLuminance(w_ld, lambda);

                    Float seed1 = w.time + w.pixelIndex + w.depth + w.n.x + w.n.y +
                                  w.n.z + w.wo.x + w.wo.y + w.wo.z;
                    Float rng = rng1D(seed1);
                    float lastFrameWeight = spectrumToLuminance(Ld, lambda);

                    reservoir.update(ls, rng, lastFrameWeight, lastFrameReservoir.M,
                                        target_p, sampledLightP, normal, depth);
                        
                    reservoir.updateWeight();
                }
                return reservoir;
            };
            
            auto spatialReuse = [&](DIReservoir reservoir, int radius) {
                // Sample light and enqueue shadow ray at intersection point
                BxDFFlags flags = bsdf.Flags();
                if (IsNonSpecular(flags)) {
                    int minX = film.PixelBounds().pMin.x;
                    int minY = film.PixelBounds().pMin.y;
                    int maxX = film.PixelBounds().pMax.x;
                    int maxY = film.PixelBounds().pMax.y;

                    int spatialSampleCount = 0;
                    for (int curOffY = 0; curOffY <= radius*2; ++ curOffY){
                        int dy = 0;
                        if (spatialSampleCount >= 8)
                            break;
                        if (curOffY == 0)
                        {
                            dy = 0;
                        } else
                        {
                            if (curOffY % 2)
                                dy = curOffY * -0.5;
                            else
                                dy = curOffY * 0.5;
                        }
                        for (int curOffX = 0; curOffX <= radius * 2; ++curOffX){
                            int dx = 0;
                            if (spatialSampleCount >= 8)
                                break;
                            if (curOffX == 0) {
                                dx = 0;
                            } else {
                                if (curOffX % 2)
                                    dx = curOffX *-0.5;
                                else
                                    dx = curOffX * 0.5;
                            }
                            int nx = pixel.x + dx;
                            int ny = pixel.y + dy;

                            float distanceWeight =
                                1 - (sqrtf(dx * dx + dy * dy) / (radius * sqrtf(2)));

                            if (nx < minX || nx >= maxX || ny < minY || ny >= maxY) {
                                continue;
                            }

                            Point2i neighborPixel(nx, ny);
                            int neighborPixelNumber =
                                neighborPixel.y * resolution.x + neighborPixel.x;
                            DIReservoir neighborReservoir =
                                imageState.diReservoir[neighborPixelNumber];
                            
                            if (neighborReservoir.M == 0)
                                continue;

                            pstd::optional<LightLiSample> ls = neighborReservoir.ls;
                            if (!ls || !ls->L || ls->pdf == 0)
                                continue;
                            
                            Float sampledLightP = neighborReservoir.sampledLightP;
                            Vector3f wi = ls->wi;
                            Vector3f wo = w.wo;
                            Normal3f normal = neighborReservoir.normal;
                            Float depth = neighborReservoir.depth;

                            // 檢查深度差異
                            float depth_diff = std::abs(depth - w.depth);
                            if (depth_diff > 0.001 * depth) {
                                Float exitAt1 = imageState.exitAt1[curPixelNumber];
                                exitAt1++;
                                imageState.exitAt1[curPixelNumber] = exitAt1;
                                continue;
                            }

                            // 檢查法線差異
                            float normal_similarity = pbrt::Dot(normal, w.n);
                            if (normal_similarity < 0.9f) {
                                Float exitAt2 = imageState.exitAt2[curPixelNumber];
                                exitAt2++;
                                imageState.exitAt2[curPixelNumber] = exitAt2;
                                continue;
                            }

                            SampledSpectrum f = bsdf.f<ConcreteBxDF>(wo, wi);
                            if (!f) {
                                Float exitAt3 = imageState.exitAt3[curPixelNumber];
                                exitAt3++;
                                imageState.exitAt3[curPixelNumber] = exitAt3;
                                continue;
                            }

                            // Compute path throughput and path PDFs for light sample
                            SampledSpectrum beta = w.beta * f * AbsDot(wi, ns);
                            PBRT_DBG("w.beta %f %f %f %f f %f %f %f %f dot %f\n",
                                     w.beta[0], w.beta[1], w.beta[2], w.beta[3], f[0],
                                     f[1], f[2], f[3], AbsDot(wi, ns));

                            PBRT_DBG("me index %d depth %d beta %f %f %f %f f %f %f %f "
                                     "%f ls.L "
                                     "%f %f %f "
                                     "%f ls.pdf %f\n",
                                     w.pixelIndex, w.depth, beta[0], beta[1], beta[2],
                                     beta[3], f[0], f[1], f[2], f[3], ls->L[0], ls->L[1],
                                     ls->L[2], ls->L[3], ls->pdf);

                            Float lightPDF = ls->pdf * sampledLightP;
                            // This causes r_u to be zero for the shadow ray, so that
                            // part of MIS just becomes a no-op.

                            // TODO  use real light type
                            // Float bsdfPDF = IsDeltaLight(light.Type()) ? 0.f :
                            // bsdf.PDF<ConcreteBxDF>(wo, wi);
                            Float bsdfPDF = bsdf.PDF<ConcreteBxDF>(wo, wi);
                            SampledSpectrum r_u = w.r_u * bsdfPDF;
                            SampledSpectrum r_l = w.r_u * lightPDF;

                            // Enqueue shadow ray with tentative radiance contribution
                            SampledSpectrum Ld = beta * ls->L;

                            // Direct lighting and restir di
                            // RIS source PDF
                            Float source_p = lightPDF;

                            ////// RIS target PDF
                            Float target_p = 0.0f;

                            ////// RIS f(x)
                            SampledSpectrum w_ld = ClampZero(ls->L) * f;

                            target_p = spectrumToLuminance(w_ld, lambda);

                            Float seed1 = w.time + w.pixelIndex + w.depth + w.n.x +
                                          w.n.y + w.n.z + w.wo.x + w.wo.y + w.wo.z + dx + dy;
                            Float rng = rng1D(seed1);
                            float neighborWeight = distanceWeight * spectrumToLuminance(Ld, lambda);

                            reservoir.update(ls, rng, neighborWeight, neighborReservoir.M, target_p,
                                                sampledLightP, normal, depth);
                            spatialSampleCount++;
                        }
                    }
                    reservoir.updateWeight();
                }
                return reservoir;
            };

            auto addShadingRayFromReservoir = [&](DIReservoir reservoir) {
                pstd::optional<LightLiSample> ls = reservoir.ls;
                if (ls) {
                    Ray ray = SpawnRayTo(w.pi, w.n, w.time, ls->pLight.pi, ls->pLight.n);
                    // Initialize _ray_ medium if media are present
                    if (haveMedia)
                        ray.medium = Dot(ray.d, w.n) > 0 ? w.mediumInterface.outside
                                                         : w.mediumInterface.inside;

                    Vector3f wi = ls->wi;
                    Vector3f wo = w.wo;
                    SampledSpectrum f = bsdf.f<ConcreteBxDF>(wo, wi);
                    if (!f) {
                        return;
                    }

                    // Compute path throughput and path PDFs for light sample
                    SampledSpectrum beta = w.beta * f * AbsDot(wi, ns);
                    Float lightPDF = ls->pdf * reservoir.sampledLightP;
                    // This causes r_u to be zero for the shadow ray, so that
                    // part of MIS just becomes a no-op.

                    // TODO shall use real light
                    // Float bsdfPDF = IsDeltaLight(light.Type()) ? 0.f :
                    // bsdf.PDF<ConcreteBxDF>(wo, wi);
                    Float bsdfPDF = bsdf.PDF<ConcreteBxDF>(wo, wi);
                    SampledSpectrum r_u = w.r_u * bsdfPDF;
                    SampledSpectrum r_l = w.r_u * lightPDF;

                    // Enqueue shadow ray with tentative radiance contribution
                    SampledSpectrum Ld = beta * ls->L;

                    shadowRayQueue->Push(ShadowRayWorkItem{
                        ray, 1 - ShadowEpsilon, w.lambda, Ld, r_u, r_l, w.pixelIndex});
                    Float shadowrayCount = imageState.shadowRayCount[curPixelNumber];
                    shadowrayCount++;
                    imageState.shadowRayCount[curPixelNumber] = shadowrayCount;

                    PBRT_DBG(
                        "w.index %d spawned shadow ray depth %d Ld %f %f %f %f "
                        "new beta %f %f %f %f beta/uni %f %f %f %f Ld/uni %f %f %f %f\n",
                        w.pixelIndex, w.depth, Ld[0], Ld[1], Ld[2], Ld[3], beta[0],
                        beta[1], beta[2], beta[3], SafeDiv(beta, r_u)[0],
                        SafeDiv(beta, r_u)[1], SafeDiv(beta, r_u)[2],
                        SafeDiv(beta, r_u)[3], SafeDiv(Ld, r_u)[0], SafeDiv(Ld, r_u)[1],
                        SafeDiv(Ld, r_u)[2], SafeDiv(Ld, r_u)[3]);
                }
            };

            DIReservoir curFrameReservoir = DIReservoir();
            curFrameReservoir = addSampleToShadowRayReservoir(curFrameReservoir, 16);
            curFrameReservoir = temporalReuse(curFrameReservoir);
            imageState.diReservoir[curPixelNumber] = curFrameReservoir;
            curFrameReservoir = spatialReuse(curFrameReservoir, 5);
            addShadingRayFromReservoir(curFrameReservoir);
        });
}

}  // namespace pbrt
