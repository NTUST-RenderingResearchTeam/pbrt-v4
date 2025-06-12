// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_CPU_INTEGRATORS_H
#define PBRT_CPU_INTEGRATORS_H

#include <pbrt/pbrt.h>

#include <pbrt/base/camera.h>
#include <pbrt/base/sampler.h>
#include <pbrt/bsdf.h>
#include <pbrt/cameras.h>
#include <pbrt/cpu/primitive.h>
#include <pbrt/film.h>
#include <pbrt/interaction.h>
#include <pbrt/lights.h>
#include <pbrt/lightsamplers.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/print.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/sampling.h>

#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace pbrt {

// Integrator Definition
class Integrator {
  public:
    // Integrator Public Methods
    virtual ~Integrator();

    static std::unique_ptr<Integrator> Create(
        const std::string &name, const ParameterDictionary &parameters, Camera camera,
        Sampler sampler, Primitive aggregate, std::vector<Light> lights,
        const RGBColorSpace *colorSpace, const FileLoc *loc);

    virtual std::string ToString() const = 0;

    virtual void Render() = 0;

    pstd::optional<ShapeIntersection> Intersect(const Ray &ray,
                                                Float tMax = Infinity) const;
    bool IntersectP(const Ray &ray, Float tMax = Infinity) const;

    bool Unoccluded(const Interaction &p0, const Interaction &p1) const {
        return !IntersectP(p0.SpawnRayTo(p1), 1 - ShadowEpsilon);
    }

    SampledSpectrum Tr(const Interaction &p0, const Interaction &p1,
                       const SampledWavelengths &lambda) const;

    // Integrator Public Members
    Primitive aggregate;
    std::vector<Light> lights;
    std::vector<Light> infiniteLights;

  protected:
    // Integrator Protected Methods
    Integrator(Primitive aggregate, std::vector<Light> lights)
        : aggregate(aggregate), lights(lights) {
        // Integrator constructor implementation
        Bounds3f sceneBounds = aggregate ? aggregate.Bounds() : Bounds3f();
        LOG_VERBOSE("Scene bounds %s", sceneBounds);
        for (auto &light : lights) {
            light.Preprocess(sceneBounds);
            if (light.Type() == LightType::Infinite)
                infiniteLights.push_back(light);
        }
    }
};

// ImageTileIntegrator Definition
class ImageTileIntegrator : public Integrator {
  public:
    // ImageTileIntegrator Public Methods
    ImageTileIntegrator(Camera camera, Sampler sampler, Primitive aggregate,
                        std::vector<Light> lights)
        : Integrator(aggregate, lights), camera(camera), samplerPrototype(sampler) {}

    void Render();

    virtual void EvaluatePixelSample(Point2i pPixel, int sampleIndex, Sampler sampler,
                                     ScratchBuffer &scratchBuffer) = 0;

  protected:
    // ImageTileIntegrator Protected Members
    Camera camera;
    Sampler samplerPrototype;
};

// RayIntegrator Definition
class RayIntegrator : public ImageTileIntegrator {
  public:
    // RayIntegrator Public Methods
    RayIntegrator(Camera camera, Sampler sampler, Primitive aggregate,
                  std::vector<Light> lights)
        : ImageTileIntegrator(camera, sampler, aggregate, lights) {}

    void EvaluatePixelSample(Point2i pPixel, int sampleIndex, Sampler sampler,
                             ScratchBuffer &scratchBuffer) final;

    virtual SampledSpectrum Li(RayDifferential ray, SampledWavelengths &lambda,
                               Sampler sampler, ScratchBuffer &scratchBuffer,
                               VisibleSurface *visibleSurface) const = 0;
};

// *Add
// Wavefront-Style Interator for test GPU (different from GPGPU) style raytracing.
// PBRT use ThreadPool which is MIMD, and wavefront is optimizate for SIMT though.
// WavefrontIntegrator Definition
class WavefrontIntegrator : public Integrator {
  protected:

    // This will be replace by reservoir & GBuffer swapchain in ReSTIR Integrator, need rearchitecture.
    // Buffer for passing data between stage
    struct RayStageBuffer {
        RayStageBuffer() = default;
        pstd::optional<ShapeIntersection> isect = {};
        pstd::optional<BSDF> bsdf = {};
        
        // Save by light sample or random seed?
        pstd::optional<SampledLight> sampledLight = {};
        pstd::optional<LightLiSample> ls = {};

        Float u;      // Sampled Light
        Point2f uv;   // Light sample
    };

    // Buffer for passing data between Bounce
    struct RayBounceBuffer/*RayDepthBuffer*/ {
        RayBounceBuffer() = default;
        pstd::optional<RayDifferential> ray = {};
        Point2i pPixel = Point2i(0,0);
        Float etaScale = 1;
        // current bounce's pdf by Sample_f(), for MIS calculate
        Float ray_pdf = 1.0f;
        // is current bounce spawn by full specular surface
        bool specularBounce = false;
        bool anyNonSpecularBounces = false;
        bool rayFinished = false;
        pstd::optional<ShapeIntersection> prevIsect = {};
        pstd::optional<RayDifferential> prevRay = {};
        // pstd::optional<BSDF> prevBsdf;
        int depth = 0;
        SampledSpectrum beta = SampledSpectrum(1.f);

        // for ReSTIR-PT
        // beta when reconnect vertex find
        SampledSpectrum rc_beta = SampledSpectrum(1.f);

        pstd::optional<ShapeIntersection> prev_prevIsect = {};
        pstd::optional<RayDifferential> prev_prevRay = {};
        // pstd::optional<BSDF> prev_prevBsdf;

        SampledWavelengths lambda;
        SampledSpectrum weight = SampledSpectrum(1.f);
        Float filterWeight = 1;
        // save for sampler
        int dim = 0;
        // pbrt ParallelJob2D divide job first by image tile, 
        // maybe record pPixel mapping to skip pixel that already finished 
        // Point2i pPixel;
        // buffer for output to image
        // * PT ReSTIR don't use this
        SampledSpectrum L = SampledSpectrum(0.f);
    };
  public:
    // ImageTileIntegrator Public Methods
    // TODO:: maybe create a light only aggregate to separete BRDF light sample
    WavefrontIntegrator(int fps, int maxDepth, Camera camera, Sampler sampler, Primitive aggregate, /*Primitive lightAggregate,*/
                        std::vector<Light> lights, const std::string &lightSampleStrategy, bool regularize);
    virtual void SpawnFirstRays(Point2i pPixel, RayBounceBuffer &buffer, Sampler &sampler, Float aTime);
    virtual void IntersectSurfaces(RayBounceBuffer &rbBuffer, RayStageBuffer &rsBuffer);
    virtual void HitEmittedLight(RayBounceBuffer &rbBuffer, RayStageBuffer &rsBuffer);
    // get bsdfs and skip medium surfaces. 
    // maybe merge into IntersectSurfaces?
    virtual void GetBSDF(RayBounceBuffer &rbBuffer, RayStageBuffer &rsBuffer, ScratchBuffer &scratchBuffer, Sampler &sampler);
    virtual void SampleLights(RayBounceBuffer &rbBuffer, RayStageBuffer &rsBuffer, Sampler &sampler);
    // check light occluded and shading
    virtual void Shading(RayBounceBuffer &rbBuffer, RayStageBuffer &rsBuffer);
    virtual void SpawnBrdfRays(RayBounceBuffer &rbBuffer, RayStageBuffer &rsBuffer, Sampler &sampler);

  protected:

    int fps;

    // RayBuffer for wavefront
    Array2D<RayStageBuffer> rayStageBuffers;  // ReSTIR Integrator not using
    Array2D<RayBounceBuffer> rayBounceBuffers;

    Camera camera;
    Sampler samplerPrototype;
    bool regularize = false;
    int maxDepth;
    LightSampler lightSampler;

    bool disableBSDFLightSample = false;
    bool disableSampleFromLightSample = false;

    bool disableFilmGBuffer = true;

    //for motion vector
    Point3f currentPCamera;
    Point3f prevPCamera;

    bool allFinished = true;
};

// RayIntegrator Definition
class WavefrontPathIntegrator : public WavefrontIntegrator {
  public:
    // RayIntegrator Public Methods
    WavefrontPathIntegrator(int fps, int maxDepth, Camera camera, Sampler sampler, Primitive aggregate,
                  std::vector<Light> lights, const std::string &lightSampleStrategy, bool regularize);

    static std::unique_ptr<WavefrontPathIntegrator> Create(const ParameterDictionary &parameters,
                                                  Camera camera, Sampler sampler,
                                                  Primitive aggregate,
                                                  std::vector<Light> lights,
                                                  const FileLoc *loc);

    std::string ToString() const;

    void Render();
};

class ReSTIRIntegrator : public WavefrontIntegrator {
  public:

    struct RestirDIParameter {
      RestirDIParameter() = default;

      // local light DI RIS sample count
      int numLocalLightSample = 8;

      ////////////////////ReSTIR DI's parameter////////////////////

      // normal similarity threshold
      Float Nthreshold = 0.5;
      // depth similarity threshold
      Float Dthreshold = 0.1;

      //spatial parameter
      bool isSpatial = false;
      Float spatialRadius = 32.0f;
      int numSpatialSamples = 1;
      Float maxSpatialDistance = 32.f;

      //temporal parameter
      bool isTemporal = false;
      int maxAge = 20;
      int historyLimit = 20;

      bool isSpatiotemporal = false;

      // unbiased vis setting: 000000001
      bool risCheckVis = false;
      bool risDiscardInVis = false;
      bool spatialCheckVis = false;
      bool spatialForceCheckVis = false;
      bool spatialDiscardInVis = false;
      bool temporalCheckVis = false;
      bool temporalForceCheckVis = false;
      bool temporalDiscardInVis = false;
      bool finalVisCheck = true;

      ////////////////////ReSTIR DI's parameter////////////////////



      
    };

    struct RestirPTParameter {
      RestirPTParameter() = default;

      // local light DI RIS sample count
      int numLocalLightSample = 8;

      ////////////////////ReSTIR PT's parameter////////////////////

      // normal similarity threshold
      Float Nthreshold = 0.5;
      // depth similarity threshold
      Float Dthreshold = 0.1;

      //spatial parameter
      bool isSpatial = false;
      Float spatialRadius = 20.0f;
      int numSpatialSamples = 3;
      // Float maxSpatialDistance = 32.f;

      //temporal parameter
      bool isTemporal = false;
      int maxAge = 20;
      int historyLimit = 5;

      bool isSpatiotemporal = false;

      ////////////////////ReSTIR PT's parameter////////////////////

      // shift mapping mode

      bool rcVisibility = true;
      float jacobianThreshold = 10.f;

      
    };

    bool enable_SS_ReSTIR_DI = false;  // enable screen space ReSTIR DI
    bool enableNEE = true;
    bool enableRR = false;
    // RayIntegrator Public Methods
    ReSTIRIntegrator(int fps, int maxDepth, RestirDIParameter restirDISetting, RestirPTParameter restirPTSetting, Camera camera, Sampler sampler, Primitive aggregate,
                  std::vector<Light> lights, const std::string &lightSampleStrategy, bool regularize);

    static std::unique_ptr<ReSTIRIntegrator> Create(const ParameterDictionary &parameters,
                                                  Camera camera, Sampler sampler,
                                                  Primitive aggregate,
                                                  std::vector<Light> lights,
                                                  const FileLoc *loc);

    // Buffer for passing data between frame
    struct GBuffer {
        GBuffer() = default;

        pstd::optional<ShapeIntersection> isect = {};
        pstd::optional<BSDF> bsdf = {};

        pstd::optional<RayDifferential> ray = {};
        bool specularBounce = false;

        Vector3f test_wi;
    };

    struct DIReservoir {
        DIReservoir() = default;

        // Can save light sample or random seed
        pstd::optional<SampledLight> sampledLight = {};
        pstd::optional<LightLiSample> ls = {};
        // Float u;
        Point2f uv;

        // (optinal)
        // unnormalize target PDF (DI * BSDF), we can save f when sample light to improve preformance
        SampledSpectrum w_ld = SampledSpectrum(0.0f);   


        bool visibility = true;
        bool isVisCheck = false;
        Float targetPdf = 0.f;
        Float weightSum = 0.f;
        Float W = 0.f;          // (optinal) this can merge into weightSum
        int M = 0;

        Point2i spatialDistance = Point2i(0,0);
        int age = 0;
    };

    // connect Case 1: like ReSTIR-DI
    // x_k-1(*) -> x_k(light)
    // rcVertex: x_k(light)

    // connect Case 2: like ReSTIR-GI
    // x_k-1(rough) -> x_k(rough) -> x_k+1(light)
    // rcVertex: x_k(rough)

    // connect Case 3:
    // x_k-1(rough) -> x_k(rough) -> x_k+1(*)
    // rcVertex: x_k(rough)
    struct ReconnectData{
      ReconnectData() = default;
      // random seed
      // Point2i p;
      // int sampleIndex = 0;
      // int dimension = 0;
      // for case 1: like ReSTIR-DI
      SampledLight sampledLight;
      ShapeIntersection rc_isect;
      // pstd::optional<BSDF> rc_bsdf = {};
      RayDifferential ray;
      ShapeIntersection rc_Le_isect;

      Point2f uLight;
      Vector3f rc_wi;
      Float lightPdf = 0.0f;
      SampledSpectrum reconnectIrradiance;

      int pathLength = 0;
      int rcType = 0;

      bool isHitEmissive = false;
      bool isNEE = false;
    };

    struct PTReservoir {
        PTReservoir() = default;

        //////////////// necessary data for RIS ////////////////
        SampledSpectrum target = SampledSpectrum(0.0f);

        // unbiased contribution weight W or resampling weights w
        // we can convert between W and w with finalResampling() or prepareMerge()
        Float W = 0.f;
        //////////////// necessary data for RIS ////////////////

        ReconnectData rc;
        Vector3f test_wi;

        int M = 0;

        int age = 0;
    };

    

    struct RandomSeedCache{
      Point2i p;
      int sampleIndex = 0;
      int dimension = 0;
    };

    bool checkNormalSimilar(Normal3f n1, Normal3f n2, Float threshold);
    bool checkDepthSimilar(Float d1, Float d2, Float threshold);
    bool checkMaterialSimilar(BSDF m1, BSDF m2, Float threshold);

    // update reservoir weight and return if it should do swap
    bool streamReservoir(DIReservoir &reservoir, Float targetPdf, Float sourcePdf, Sampler &sampler);

    bool streamReservoir(PTReservoir &reservoir, Float targetPdf, Float sourcePdf, Sampler &sampler);

    // combine reservoir and return if it should do swap
    bool combineReservoir(DIReservoir &dst, const DIReservoir &src, Float targetPdf, Sampler &sampler, bool resetVis = true);

    bool combineReservoir(PTReservoir &dst, const PTReservoir &src, Float targetPdf, Float misWeight, Float jacobian, Sampler &sampler, bool resetVis = true);

    void finalResampling(DIReservoir &reservoir, const GBuffer &gBuffer, bool checkVis, bool forceVisCheck, bool discardIfInvisible);

    // convert weightSum to W in reservoir.weight
    void finalResampling(PTReservoir &reservoir);

    void prepareMerge(PTReservoir &reservoir);

    void storeVisibility(DIReservoir &reservoir, bool visibility, bool discardIfInvisible);

    void IntersectSurfaces(RayBounceBuffer &rbBuffer, GBuffer &gBuffer);

    void HitEmittedLight(RayBounceBuffer &rbBuffer, ReconnectData &rc, GBuffer &gBuffer, DIReservoir& diReservoir, PTReservoir &ptReservoir, ScratchBuffer &scratchBuffer, Sampler &sampler, RandomSeedCache &rand);

    void GetBSDF(RayBounceBuffer &rbBuffer, GBuffer &gBuffer, ScratchBuffer &scratchBuffer, Sampler &sampler);

    // sample light with WRS
    void SampleLights(RayBounceBuffer &rbBuffer, GBuffer &gBuffer, Sampler &sampler, DIReservoir &diReservoir);

    // combine DI reservoir temporal
    void TemporalResample(Point2i pPixel, Array2D<DIReservoir> &prevReservoirs, const Array2D<GBuffer> &prevGBuffers, DIReservoir &dstReservoir, const GBuffer &dstGbuffer, const RayBounceBuffer &dstRb, Sampler &sampler);
    
    // combine DI reservoir spatial
    void SpatialResample(Point2i pPixel, const Array2D<DIReservoir> &srcReservoir, const Array2D<GBuffer> &srcGBuffers, const Array2D<RayBounceBuffer> &srcRb, DIReservoir &dstReservoir, Sampler &sampler);

    // combine DI reservoir spatiotemporal
    void SpatialtemporalResample(Point2i pPixel, Array2D<DIReservoir> &prevReservoirs, const Array2D<GBuffer> &prevGBuffers, DIReservoir &dstReservoir, GBuffer &dstGBuffer, RayBounceBuffer &dstRb, Sampler &sampler);

    // check light occluded and shading for WRS
    void Shading(RayBounceBuffer &rbBuffer, ReconnectData &rc, GBuffer &gBuffer, DIReservoir& diReservoir, PTReservoir &ptReservoir, ScratchBuffer &scratchBuffer, Sampler &sampler, RandomSeedCache &rand);

    void SpawnBrdfRays(RayBounceBuffer &rbBuffer, ReconnectData &rc, GBuffer &gBuffer, ScratchBuffer &scratchBuffer, Sampler &sampler);

    bool ShiftReservoir(PTReservoir &dstReservoir, const PTReservoir &srcReservoir, const GBuffer &dstGBuffer, const GBuffer &srcGBuffer, const RayBounceBuffer &dstRbBuffer, SampledSpectrum &w_ld, Float &jacobian, ScratchBuffer &scratchBuffer, Sampler &sampler);
    // combine PT reservoir temporal
    void ReconnectTemporalResample(Point2i pPixel, Array2D<PTReservoir> &prevReservoirs, const Array2D<GBuffer> &prevGBuffers, PTReservoir &dstReservoir, const GBuffer &dstGBuffer, const RayBounceBuffer &dstRb, ScratchBuffer &scratchBuffer, Sampler &sampler);
    // combine PT reservoir spatial
    void ReconnectSpatialResample(Point2i pPixel, const Array2D<PTReservoir> &srcReservoirs, const Array2D<GBuffer> &srcGBuffers, const Array2D<RayBounceBuffer> &srcRb, PTReservoir &dstReservoir, ScratchBuffer &scratchBuffer, Sampler &sampler);

    

    Array2D<GBuffer> pingGBuffers;
    Array2D<GBuffer> pongGBuffers;

    Array2D<GBuffer> ptGBuffers;

    Array2D<DIReservoir> pingDIReservoirBuffers;
    Array2D<DIReservoir> pongDIReservoirBuffers;

    Array2D<PTReservoir> pingPTReservoirBuffers;
    Array2D<PTReservoir> pongPTReservoirBuffers;

    Array2D<ReconnectData> rcBuffers;

    RestirDIParameter restirDISetting;
    RestirPTParameter restirPTSetting;

    int currentSampleIndex = 0;
    Float currentTime = 0.f;
    int totalFrame = 0;
    int currentFrame = 0;

    bool isFirstRay = true;

    std::string ToString() const;

    void Render();
};

// RandomWalkIntegrator Definition
class RandomWalkIntegrator : public RayIntegrator {
  public:
    // RandomWalkIntegrator Public Methods
    RandomWalkIntegrator(int maxDepth, Camera camera, Sampler sampler,
                         Primitive aggregate, std::vector<Light> lights)
        : RayIntegrator(camera, sampler, aggregate, lights), maxDepth(maxDepth) {}

    static std::unique_ptr<RandomWalkIntegrator> Create(
        const ParameterDictionary &parameters, Camera camera, Sampler sampler,
        Primitive aggregate, std::vector<Light> lights, const FileLoc *loc);

    std::string ToString() const;

    SampledSpectrum Li(RayDifferential ray, SampledWavelengths &lambda, Sampler sampler,
                       ScratchBuffer &scratchBuffer,
                       VisibleSurface *visibleSurface) const {
        return LiRandomWalk(ray, lambda, sampler, scratchBuffer, 0);
    }

  private:
    // RandomWalkIntegrator Private Methods
    SampledSpectrum LiRandomWalk(RayDifferential ray, SampledWavelengths &lambda,
                                 Sampler sampler, ScratchBuffer &scratchBuffer,
                                 int depth) const {
        // Intersect ray with scene and return if no intersection
        pstd::optional<ShapeIntersection> si = Intersect(ray);
        if (!si) {
            // Return emitted light from infinite light sources
            SampledSpectrum Le(0.f);
            for (Light light : infiniteLights)
                Le += light.Le(ray, lambda);
            return Le;
        }
        SurfaceInteraction &isect = si->intr;

        // Get emitted radiance at surface intersection
        Vector3f wo = -ray.d;
        SampledSpectrum Le = isect.Le(wo, lambda);

        // Terminate random walk if maximum depth has been reached
        if (depth == maxDepth)
            return Le;

        // Compute BSDF at random walk intersection point
        BSDF bsdf = isect.GetBSDF(ray, lambda, camera, scratchBuffer, sampler);
        if (!bsdf)
            return Le;

        // Randomly sample direction leaving surface for random walk
        Point2f u = sampler.Get2D();
        Vector3f wp = SampleUniformSphere(u);

        // Evaluate BSDF at surface for sampled direction
        SampledSpectrum fcos = bsdf.f(wo, wp) * AbsDot(wp, isect.shading.n);
        if (!fcos)
            return Le;

        // Recursively trace ray to estimate incident radiance at surface
        ray = isect.SpawnRay(wp);
        return Le + fcos * LiRandomWalk(ray, lambda, sampler, scratchBuffer, depth + 1) /
                        (1 / (4 * Pi));
    }

    // RandomWalkIntegrator Private Members
    int maxDepth;
};

// SimplePathIntegrator Definition
class SimplePathIntegrator : public RayIntegrator {
  public:
    // SimplePathIntegrator Public Methods
    SimplePathIntegrator(int maxDepth, bool sampleLights, bool sampleBSDF, Camera camera,
                         Sampler sampler, Primitive aggregate, std::vector<Light> lights);

    SampledSpectrum Li(RayDifferential ray, SampledWavelengths &lambda, Sampler sampler,
                       ScratchBuffer &scratchBuffer,
                       VisibleSurface *visibleSurface) const;

    static std::unique_ptr<SimplePathIntegrator> Create(
        const ParameterDictionary &parameters, Camera camera, Sampler sampler,
        Primitive aggregate, std::vector<Light> lights, const FileLoc *loc);

    std::string ToString() const;

  private:
    // SimplePathIntegrator Private Members
    int maxDepth;
    bool sampleLights, sampleBSDF;
    UniformLightSampler lightSampler;
};

// PathIntegrator Definition
class PathIntegrator : public RayIntegrator {
  public:
    // PathIntegrator Public Methods
    PathIntegrator(int maxDepth, Camera camera, Sampler sampler, Primitive aggregate,
                   std::vector<Light> lights,
                   const std::string &lightSampleStrategy = "bvh",
                   bool regularize = false);

    SampledSpectrum Li(RayDifferential ray, SampledWavelengths &lambda, Sampler sampler,
                       ScratchBuffer &scratchBuffer,
                       VisibleSurface *visibleSurface) const;

    static std::unique_ptr<PathIntegrator> Create(const ParameterDictionary &parameters,
                                                  Camera camera, Sampler sampler,
                                                  Primitive aggregate,
                                                  std::vector<Light> lights,
                                                  const FileLoc *loc);

    std::string ToString() const;

  private:
    // PathIntegrator Private Methods
    SampledSpectrum SampleLd(const SurfaceInteraction &intr, const BSDF *bsdf,
                             SampledWavelengths &lambda, Sampler sampler) const;

    SampledSpectrum SampleAllLd(const SurfaceInteraction &intr, const BSDF *bsdf,
                             SampledWavelengths &lambda, Sampler sampler) const;

    // PathIntegrator Private Members
    int maxDepth;
    LightSampler lightSampler;
    bool regularize;
};

////////////////////////////////////////RIS Implement////////////////////////////////////////////////////

// RIS PathIntegrator Definition
class RisPathIntegrator : public RayIntegrator {
  public:
    // PathIntegrator Public Methods
    RisPathIntegrator(int maxDepth, int numRisSampleLd, Camera camera, Sampler sampler, Primitive aggregate,
                   std::vector<Light> lights,
                   const std::string &lightSampleStrategy = "uniform",
                   bool regularize = false);

    SampledSpectrum Li(RayDifferential ray, SampledWavelengths &lambda, Sampler sampler,
                       ScratchBuffer &scratchBuffer,
                       VisibleSurface *visibleSurface) const;

    static std::unique_ptr<RisPathIntegrator> Create(const ParameterDictionary &parameters,
                                                  Camera camera, Sampler sampler,
                                                  Primitive aggregate,
                                                  std::vector<Light> lights,
                                                  const FileLoc *loc);

    std::string ToString() const;

  private:
    // PathIntegrator Private Methods
    SampledSpectrum SampleLd(const SurfaceInteraction &intr, const BSDF *bsdf,
                             SampledWavelengths &lambda, Sampler sampler) const;

    // PathIntegrator Private Members
    int maxDepth;
    LightSampler lightSampler;
    bool regularize;

    int numRisSampleLd;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////

// SimpleVolPathIntegrator Definition
class SimpleVolPathIntegrator : public RayIntegrator {
  public:
    // SimpleVolPathIntegrator Public Methods
    SimpleVolPathIntegrator(int maxDepth, Camera camera, Sampler sampler,
                            Primitive aggregate, std::vector<Light> lights);

    SampledSpectrum Li(RayDifferential ray, SampledWavelengths &lambda, Sampler sampler,
                       ScratchBuffer &scratchBuffer,
                       VisibleSurface *visibleSurface) const;

    static std::unique_ptr<SimpleVolPathIntegrator> Create(
        const ParameterDictionary &parameters, Camera camera, Sampler sampler,
        Primitive aggregate, std::vector<Light> lights, const FileLoc *loc);

    std::string ToString() const;

  private:
    // SimpleVolPathIntegrator Private Members
    int maxDepth;
};

// VolPathIntegrator Definition
class VolPathIntegrator : public RayIntegrator {
  public:
    // VolPathIntegrator Public Methods
    VolPathIntegrator(int maxDepth, Camera camera, Sampler sampler, Primitive aggregate,
                      std::vector<Light> lights,
                      const std::string &lightSampleStrategy = "bvh",
                      bool regularize = false)
        : RayIntegrator(camera, sampler, aggregate, lights),
          maxDepth(maxDepth),
          lightSampler(LightSampler::Create(lightSampleStrategy, lights, Allocator())),
          regularize(regularize) {}

    SampledSpectrum Li(RayDifferential ray, SampledWavelengths &lambda, Sampler sampler,
                       ScratchBuffer &scratchBuffer,
                       VisibleSurface *visibleSurface) const;

    static std::unique_ptr<VolPathIntegrator> Create(
        const ParameterDictionary &parameters, Camera camera, Sampler sampler,
        Primitive aggregate, std::vector<Light> lights, const FileLoc *loc);

    std::string ToString() const;

  private:
    // VolPathIntegrator Private Methods
    SampledSpectrum SampleLd(const Interaction &intr, const BSDF *bsdf,
                             SampledWavelengths &lambda, Sampler sampler,
                             SampledSpectrum beta, SampledSpectrum inv_w_u) const;

    // VolPathIntegrator Private Members
    int maxDepth;
    LightSampler lightSampler;
    bool regularize;
};

// AOIntegrator Definition
class AOIntegrator : public RayIntegrator {
  public:
    // AOIntegrator Public Methods
    AOIntegrator(bool cosSample, Float maxDist, Camera camera, Sampler sampler,
                 Primitive aggregate, std::vector<Light> lights, Spectrum illuminant);

    SampledSpectrum Li(RayDifferential ray, SampledWavelengths &lambda, Sampler sampler,
                       ScratchBuffer &scratchBuffer,
                       VisibleSurface *visibleSurface) const;

    static std::unique_ptr<AOIntegrator> Create(const ParameterDictionary &parameters,
                                                Spectrum illuminant, Camera camera,
                                                Sampler sampler, Primitive aggregate,
                                                std::vector<Light> lights,
                                                const FileLoc *loc);

    std::string ToString() const;

  private:
    bool cosSample;
    Float maxDist;
    Spectrum illuminant;
    Float illumScale;
};

// LightPathIntegrator Definition
class LightPathIntegrator : public ImageTileIntegrator {
  public:
    // LightPathIntegrator Public Methods
    LightPathIntegrator(int maxDepth, Camera camera, Sampler sampler, Primitive aggregate,
                        std::vector<Light> lights);

    void EvaluatePixelSample(Point2i pPixel, int sampleIndex, Sampler sampler,
                             ScratchBuffer &scratchBuffer);

    static std::unique_ptr<LightPathIntegrator> Create(
        const ParameterDictionary &parameters, Camera camera, Sampler sampler,
        Primitive aggregate, std::vector<Light> lights, const FileLoc *loc);

    std::string ToString() const;

  private:
    // LightPathIntegrator Private Members
    int maxDepth;
    PowerLightSampler lightSampler;
};

// BDPTIntegrator Definition
struct Vertex;
class BDPTIntegrator : public RayIntegrator {
  public:
    // BDPTIntegrator Public Methods
    BDPTIntegrator(Camera camera, Sampler sampler, Primitive aggregate,
                   std::vector<Light> lights, int maxDepth, bool visualizeStrategies,
                   bool visualizeWeights, bool regularize = false)
        : RayIntegrator(camera, sampler, aggregate, lights),
          maxDepth(maxDepth),
          regularize(regularize),
          lightSampler(new PowerLightSampler(lights, Allocator())),
          visualizeStrategies(visualizeStrategies),
          visualizeWeights(visualizeWeights) {}

    SampledSpectrum Li(RayDifferential ray, SampledWavelengths &lambda, Sampler sampler,
                       ScratchBuffer &scratchBuffer,
                       VisibleSurface *visibleSurface) const;

    static std::unique_ptr<BDPTIntegrator> Create(const ParameterDictionary &parameters,
                                                  Camera camera, Sampler sampler,
                                                  Primitive aggregate,
                                                  std::vector<Light> lights,
                                                  const FileLoc *loc);

    std::string ToString() const;

    void Render();

  private:
    // BDPTIntegrator Private Members
    int maxDepth;
    bool regularize;
    LightSampler lightSampler;
    bool visualizeStrategies, visualizeWeights;
    mutable std::vector<Film> weightFilms;
};

// MLTIntegrator Definition
class MLTSampler;

class MLTIntegrator : public Integrator {
  public:
    // MLTIntegrator Public Methods
    MLTIntegrator(Camera camera, Primitive aggregate, std::vector<Light> lights,
                  int maxDepth, int nBootstrap, int nChains, int mutationsPerPixel,
                  Float sigma, Float largeStepProbability, bool regularize)
        : Integrator(aggregate, lights),
          lightSampler(new PowerLightSampler(lights, Allocator())),
          camera(camera),
          maxDepth(maxDepth),
          nBootstrap(nBootstrap),
          nChains(nChains),
          mutationsPerPixel(mutationsPerPixel),
          sigma(sigma),
          largeStepProbability(largeStepProbability),
          regularize(regularize) {}

    void Render();

    static std::unique_ptr<MLTIntegrator> Create(const ParameterDictionary &parameters,
                                                 Camera camera, Primitive aggregate,
                                                 std::vector<Light> lights,
                                                 const FileLoc *loc);

    std::string ToString() const;

  private:
    // MLTIntegrator Constants
    static constexpr int cameraStreamIndex = 0;
    static constexpr int lightStreamIndex = 1;
    static constexpr int connectionStreamIndex = 2;
    static constexpr int nSampleStreams = 3;

    // MLTIntegrator Private Methods
    SampledSpectrum L(ScratchBuffer &scratchBuffer, MLTSampler &sampler, int k,
                      Point2f *pRaster, SampledWavelengths *lambda);

    static Float c(const SampledSpectrum &L, const SampledWavelengths &lambda) {
        return L.y(lambda);
    }

    // MLTIntegrator Private Members
    Camera camera;
    bool regularize;
    LightSampler lightSampler;
    int maxDepth, nBootstrap;
    int mutationsPerPixel;
    Float sigma, largeStepProbability;
    int nChains;
};

// SPPMIntegrator Definition
class SPPMIntegrator : public Integrator {
  public:
    // SPPMIntegrator Public Methods
    SPPMIntegrator(Camera camera, Sampler sampler, Primitive aggregate,
                   std::vector<Light> lights, int photonsPerIteration, int maxDepth,
                   Float initialSearchRadius, int seed, const RGBColorSpace *colorSpace)
        : Integrator(aggregate, lights),
          camera(camera),
          samplerPrototype(sampler),
          initialSearchRadius(initialSearchRadius),
          maxDepth(maxDepth),
          photonsPerIteration(photonsPerIteration > 0
                                  ? photonsPerIteration
                                  : camera.GetFilm().PixelBounds().Area()),
          colorSpace(colorSpace),
          digitPermutationsSeed(seed) {}

    static std::unique_ptr<SPPMIntegrator> Create(const ParameterDictionary &parameters,
                                                  const RGBColorSpace *colorSpace,
                                                  Camera camera, Sampler sampler,
                                                  Primitive aggregate,
                                                  std::vector<Light> lights,
                                                  const FileLoc *loc);

    std::string ToString() const;

    void Render();

  private:
    // SPPMIntegrator Private Methods
    SampledSpectrum SampleLd(const SurfaceInteraction &intr, const BSDF &bsdf,
                             SampledWavelengths &lambda, Sampler sampler,
                             LightSampler lightSampler) const;

    // SPPMIntegrator Private Members
    Camera camera;
    Float initialSearchRadius;
    Sampler samplerPrototype;
    int digitPermutationsSeed;
    int maxDepth;
    int photonsPerIteration;
    const RGBColorSpace *colorSpace;
};

// FunctionIntegrator Definition
class FunctionIntegrator : public Integrator {
  public:
    FunctionIntegrator(std::function<double(Point2f)> func,
                       const std::string &outputFilename, Camera camera, Sampler sampler,
                       bool skipBad, std::string imageFilename);

    static std::unique_ptr<FunctionIntegrator> Create(
        const ParameterDictionary &parameters, Camera camera, Sampler sampler,
        const FileLoc *loc);

    void Render();

    std::string ToString() const;

  private:
    std::function<double(Point2f)> func;
    std::string outputFilename;
    Camera camera;
    Sampler baseSampler;
    bool skipBad;
    std::string imageFilename;
};

}  // namespace pbrt

#endif  // PBRT_CPU_INTEGRATORS_H
