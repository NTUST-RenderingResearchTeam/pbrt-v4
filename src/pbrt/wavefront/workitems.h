// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_WAVEFRONT_WORKITEMS_H
#define PBRT_WAVEFRONT_WORKITEMS_H

#include <pbrt/pbrt.h>

#include <pbrt/base/sampler.h>
#include <pbrt/film.h>
#include <pbrt/lightsamplers.h>
#include <pbrt/materials.h>
#include <pbrt/ray.h>
#include <pbrt/util/containers.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/soa.h>
#include <pbrt/wavefront/workqueue.h>

namespace pbrt {

struct DIReservoir {
    DIReservoir() = default;

    LightSampleContext ctx;
    Float lightRng;
    Point2f lightSampleRng;
    Float weightSum = 0.f;
    Float W = 0.f;
    int M = 0;
    // Check reuse correleation
    Normal3f normal;
    Float depth;
    Float targetPdf = 0.f;

    PBRT_CPU_GPU
    void update(Float rng, LightSampleContext _ctx, Float _lightRng, Point2f _lightSampleRng, Float weight,
                Float M,
                Float targetPDF, Normal3f normal, Float depth);

    PBRT_CPU_GPU
    void updateWeight();
};

inline void DIReservoir::update(Float rng, LightSampleContext _ctx, Float _lightRng,
                                Point2f _lightSampleRng, Float weight, Float _M,
                                Float _targetPdf,
                                Normal3f _normal, Float _depth) {
    weightSum += weight;
    M += _M;
    /*if (M > 30)
        M = 30;*/

    if (rng * weightSum <= weight) {
        ctx = _ctx;
        lightRng = _lightRng;
        lightSampleRng = _lightSampleRng;
        normal = _normal;
        depth = _depth;
        targetPdf = _targetPdf;
    }
}

inline void DIReservoir::updateWeight() {
    if (M > 0 && targetPdf > 0.f) {
        W = (weightSum / M) / targetPdf;
    } else {
        W = 0.f;
    }
}
//template <>
//struct SOA<DIReservoir> {
//  public:
//    SOA() = default;
//    SOA(int size, Allocator alloc) {
//        // Basic Float members
//        targetPdf = alloc.allocate_object<Float>(size);
//        weightSum = alloc.allocate_object<Float>(size);
//        sampledLightP = alloc.allocate_object<Float>(size);
//        W = alloc.allocate_object<Float>(size);
//        M = alloc.allocate_object<Float>(size);           // Store int as Float
//        normal = alloc.allocate_object<Float4>(size);      
//        depth = alloc.allocate_object<Float>(size);           // Store int as Float
//        age = alloc.allocate_object<Float>(size);         // Store int as Float
//        visibility = alloc.allocate_object<Float>(size);  // Store bool as Float
//        isVisCheck = alloc.allocate_object<Float>(size);  // Store bool as Float
//
//        // uv and spatialDistance packed into Float4
//        uvAndFlags = alloc.allocate_object<Float4>(size);
//
//        // LightLiSample members using Float4
//        lightL = alloc.allocate_object<Float4>(size);
//        lightWi = alloc.allocate_object<Float4>(size);
//        lightPdf = alloc.allocate_object<Float4>(size);
//
//        // Interaction members using Float4
//        lightPi = alloc.allocate_object<Float4>(size);
//        lightWo = alloc.allocate_object<Float4>(size);
//        lightN = alloc.allocate_object<Float4>(size);
//        lightUV = alloc.allocate_object<Float4>(size);
//    }
//
//    PBRT_CPU_GPU
//    DIReservoir operator[](int i) const {
//        DIReservoir res;
//
//        // Basic members
//        res.targetPdf = targetPdf[i];
//        res.weightSum = weightSum[i];
//        res.sampledLightP = sampledLightP[i];
//        Float4 hitNormal = Load4(normal + i);
//        res.normal = Normal3f(hitNormal.v[0], hitNormal.v[1], hitNormal.v[2]);
//        res.depth = depth[i];
//        res.W = W[i];
//        res.M = int(M[i]);
//        res.age = int(age[i]);
//        res.visibility = visibility[i] != 0;
//        res.isVisCheck = isVisCheck[i] != 0;
//
//        // Get uv and spatialDistance from Float4
//        Float4 uvFlags = Load4(uvAndFlags + i);
//        res.uv = Point2f(uvFlags.v[0], uvFlags.v[1]);
//        res.spatialDistance = Point2i(int(uvFlags.v[2]), int(uvFlags.v[3]));
//
//        // LightLiSample
//        Float4 lpdf = Load4(lightPdf + i);
//        if (lpdf.v[3] != 0) {  // Use w component as has_value flag
//            Float4 L = Load4(lightL + i);
//            Float4 wi = Load4(lightWi + i);
//
//            // Construct Interaction
//            Interaction pLight;
//            Float4 pi = Load4(lightPi + i);
//            pLight.pi = Point3fi(pi.v[0], pi.v[1], pi.v[2]);
//            pLight.time = pi.v[3];
//
//            Float4 wo = Load4(lightWo + i);
//            pLight.wo = Vector3f(wo.v[0], wo.v[1], wo.v[2]);
//
//            Float4 n = Load4(lightN + i);
//            pLight.n = Normal3f(n.v[0], n.v[1], n.v[2]);
//
//            Float4 uv = Load4(lightUV + i);
//            pLight.uv = Point2f(uv.v[0], uv.v[1]);
//
//            SampledSpectrum ssl;
//            ssl[0] = L.v[0];
//            ssl[1] = L.v[1];
//            ssl[2] = L.v[2];
//            ssl[3] = L.v[3];
//            res.ls = LightLiSample(ssl,
//                           Vector3f(wi.v[0], wi.v[1], wi.v[2]), lpdf.v[0], pLight,
//                           LightType(int(lpdf.v[1])));
//        }
//
//        return res;
//    }
//
//    struct GetSetIndirector {
//        PBRT_CPU_GPU
//        operator DIReservoir() const { return (*(const SOA *)soa)[index]; }
//
//        PBRT_CPU_GPU
//        void operator=(DIReservoir res) {
//            // Basic members
//            soa->targetPdf[index] = res.targetPdf;
//            soa->weightSum[index] = res.weightSum;
//            soa->sampledLightP[index] = res.sampledLightP;
//            soa->normal[index] = Float4{res.normal.x, res.normal.y, res.normal.z, 0};
//            soa->depth[index] = res.depth;
//            soa->W[index] = res.W;
//            soa->M[index] = Float(res.M);
//            soa->age[index] = Float(res.age);
//            soa->visibility[index] = res.visibility ? 1.f : 0.f;
//            soa->isVisCheck[index] = res.isVisCheck ? 1.f : 0.f;
//
//            // Pack uv and spatialDistance into Float4
//            soa->uvAndFlags[index] =
//                Float4{res.uv.x, res.uv.y, Float(res.spatialDistance.x),
//                       Float(res.spatialDistance.y)};
//
//            // LightLiSample
//            if (res.ls) {
//                const auto &ls = *res.ls;
//                soa->lightL[index] = Float4{ls.L[0], ls.L[1], ls.L[2], 0};
//                soa->lightWi[index] = Float4{ls.wi.x, ls.wi.y, ls.wi.z, 0};
//                soa->lightPdf[index] =
//                    Float4{ls.pdf, Float(ls.type), 0, 1};  // w=1 indicates has_value
//
//                // Interaction
//                const auto &pLight = ls.pLight;
//                soa->lightPi[index] =
//                    Float4{pLight.pi.x.Midpoint(), pLight.pi.y.Midpoint(),
//                           pLight.pi.z.Midpoint(),
//                           pLight.time};
//                soa->lightWo[index] = Float4{pLight.wo.x, pLight.wo.y, pLight.wo.z, 0};
//                soa->lightN[index] = Float4{pLight.n.x, pLight.n.y, pLight.n.z, 0};
//                soa->lightUV[index] = Float4{pLight.uv.x, pLight.uv.y, 0, 0};
//            } else {
//                // Set has_value flag to false
//                soa->lightPdf[index] = Float4{0, 0, 0, 0};
//            }
//        }
//
//        SOA *soa;
//        int index;
//    };
//
//    PBRT_CPU_GPU
//    GetSetIndirector operator[](int i) { return GetSetIndirector{this, i}; }
//
//  private:
//    // Basic members
//    Float *PBRT_RESTRICT targetPdf;
//    Float *PBRT_RESTRICT weightSum;
//    Float *PBRT_RESTRICT sampledLightP;
//    Float4 *PBRT_RESTRICT normal;
//    Float *PBRT_RESTRICT depth; 
//    Float *PBRT_RESTRICT W;
//    Float *PBRT_RESTRICT M;
//    Float *PBRT_RESTRICT age;
//    Float *PBRT_RESTRICT visibility;
//    Float *PBRT_RESTRICT isVisCheck;
//
//    // Packed uv and spatialDistance
//    Float4 *PBRT_RESTRICT uvAndFlags;
//
//    // LightLiSample members
//    Float4 *PBRT_RESTRICT lightL;
//    Float4 *PBRT_RESTRICT lightWi;
//    Float4 *PBRT_RESTRICT lightPdf;  // {pdf, type, unused, has_value}
//
//    // Interaction members
//    Float4 *PBRT_RESTRICT lightPi;
//    Float4 *PBRT_RESTRICT lightWo;
//    Float4 *PBRT_RESTRICT lightN;
//    Float4 *PBRT_RESTRICT lightUV;
//};

// RaySamples Definition
struct RaySamples {
    // RaySamples Public Members
    struct {
        Point2f u;
        Float uc;
    } direct;
    struct {
        Point2f u;
        Float uc;
    } direct1;
    struct {
        Point2f u;
        Float uc;
    } direct2;
    struct {
        Point2f u;
        Float uc;
    } direct3;
    struct {
        Point2f u;
        Float uc;
    } direct4;
    struct {
        Float uc, rr;
        Point2f u;
    } indirect;
    bool haveSubsurface;
    struct {
        Float uc;
        Point2f u;
    } subsurface;
};

template <>
struct SOA<RaySamples> {
  public:
    SOA() = default;

    SOA(int size, Allocator alloc) {
        direct = alloc.allocate_object<Float4>(size);
        direct1 = alloc.allocate_object<Float4>(size);
        direct2 = alloc.allocate_object<Float4>(size);
        direct3 = alloc.allocate_object<Float4>(size);
        direct4 = alloc.allocate_object<Float4>(size);
        indirect = alloc.allocate_object<Float4>(size);
        subsurface = alloc.allocate_object<Float4>(size);
        mediaDist = alloc.allocate_object<Float>(size);
        mediaMode = alloc.allocate_object<Float>(size);
    }

    PBRT_CPU_GPU
    RaySamples operator[](int i) const {
        RaySamples rs;
        Float4 dir = Load4(direct + i);
        rs.direct.u = Point2f(dir.v[0], dir.v[1]);
        rs.direct.uc = dir.v[2];

        rs.haveSubsurface = int(dir.v[3]) & 1;

        Float4 dir1 = Load4(direct1 + i);
        rs.direct1.u = Point2f(dir1.v[0], dir1.v[1]);
        rs.direct1.uc = dir1.v[2];
        Float4 dir2 = Load4(direct2 + i);
        rs.direct2.u = Point2f(dir2.v[0], dir2.v[1]);
        rs.direct2.uc = dir2.v[2];
        Float4 dir3 = Load4(direct3 + i);
        rs.direct3.u = Point2f(dir3.v[0], dir3.v[1]);
        rs.direct3.uc = dir3.v[2];
        Float4 dir4 = Load4(direct4 + i);
        rs.direct4.u = Point2f(dir4.v[0], dir4.v[1]);
        rs.direct4.uc = dir4.v[2];

        Float4 ind = Load4(indirect + i);
        rs.indirect.uc = ind.v[0];
        rs.indirect.rr = ind.v[1];
        rs.indirect.u = Point2f(ind.v[2], ind.v[3]);

        if (rs.haveSubsurface) {
            Float4 ss = Load4(subsurface + i);
            rs.subsurface.uc = ss.v[0];
            rs.subsurface.u = Point2f(ss.v[1], ss.v[2]);
        }

        return rs;
    }

    struct GetSetIndirector {
        PBRT_CPU_GPU
        operator RaySamples() const { return (*(const SOA *)soa)[index]; }

        PBRT_CPU_GPU
        void operator=(RaySamples rs) {
            int flags = rs.haveSubsurface ? 1 : 0;
            soa->direct[index] =
                Float4{rs.direct.u[0], rs.direct.u[1], rs.direct.uc, Float(flags)};
            soa->direct1[index] =
                Float4{rs.direct1.u[0], rs.direct1.u[1], rs.direct1.uc, 0};
            soa->direct2[index] =
                Float4{rs.direct2.u[0], rs.direct2.u[1], rs.direct2.uc, 0};
            soa->direct3[index] =
                Float4{rs.direct3.u[0], rs.direct3.u[1], rs.direct3.uc, 0};
            soa->direct4[index] =
                Float4{rs.direct4.u[0], rs.direct4.u[1], rs.direct4.uc, 0};
            soa->indirect[index] = Float4{rs.indirect.uc, rs.indirect.rr,
                                          rs.indirect.u[0], rs.indirect.u[1]};
            if (rs.haveSubsurface)
                soa->subsurface[index] =
                    Float4{rs.subsurface.uc, rs.subsurface.u.x, rs.subsurface.u.y, 0.f};
        }

        SOA *soa;
        int index;
    };

    PBRT_CPU_GPU
    GetSetIndirector operator[](int i) { return GetSetIndirector{this, i}; }

  private:
    Float4 *PBRT_RESTRICT direct;
    Float4 *PBRT_RESTRICT direct1;
    Float4 *PBRT_RESTRICT direct2;
    Float4 *PBRT_RESTRICT direct3;
    Float4 *PBRT_RESTRICT direct4;
    Float4 *PBRT_RESTRICT indirect;
    Float4 *PBRT_RESTRICT subsurface;
    Float *PBRT_RESTRICT mediaDist, *PBRT_RESTRICT mediaMode;
};

struct ImageState {
    DIReservoir diReservoirA;
    DIReservoir diReservoirB;
    Float shadowRayCount;
    Float exitAt1;
    Float exitAt2;
    Float exitAt3;
};

// PixelSampleState Definition
struct PixelSampleState {
    // PixelSampleState Public Members
    Point2i pPixel;
    SampledSpectrum L;
    SampledWavelengths lambda;
    Float filterWeight;
    VisibleSurface visibleSurface;
    SampledSpectrum cameraRayWeight;
    RaySamples samples;
    SampledSpectrum DirectL;
};

// RayWorkItem Definition
struct RayWorkItem {
    // RayWorkItem Public Members
    Ray ray;
    int depth;
    SampledWavelengths lambda;
    int pixelIndex;
    SampledSpectrum beta, r_u, r_l;
    LightSampleContext prevIntrCtx;
    Float etaScale;
    int specularBounce;
    int anyNonSpecularBounces;
};

// EscapedRayWorkItem Definition
struct EscapedRayWorkItem {
    // EscapedRayWorkItem Public Members
    Point3f rayo;
    Vector3f rayd;
    int depth;
    SampledWavelengths lambda;
    int pixelIndex;
    SampledSpectrum beta;
    int specularBounce;
    SampledSpectrum r_u, r_l;
    LightSampleContext prevIntrCtx;
};

// HitAreaLightWorkItem Definition
struct HitAreaLightWorkItem {
    // HitAreaLightWorkItem Public Members
    Light areaLight;
    Point3f p;
    Normal3f n;
    Point2f uv;
    Vector3f wo;
    SampledWavelengths lambda;
    int depth;
    SampledSpectrum beta, r_u, r_l;
    LightSampleContext prevIntrCtx;
    int specularBounce;
    int pixelIndex;
};

// HitAreaLightQueue Definition
using HitAreaLightQueue = WorkQueue<HitAreaLightWorkItem>;

// ShadowRayWorkItem Definition
struct ShadowRayWorkItem {
    Ray ray;
    Float tMax;
    SampledWavelengths lambda;
    SampledSpectrum Ld, r_u, r_l;
    int pixelIndex;
};

// GetBSSRDFAndProbeRayWorkItem Definition
struct GetBSSRDFAndProbeRayWorkItem {
    PBRT_CPU_GPU
    MaterialEvalContext GetMaterialEvalContext() const {
        MaterialEvalContext ctx;
        ctx.wo = wo;
        ctx.n = n;
        ctx.ns = ns;
        ctx.dpdus = dpdus;
        ctx.p = p;
        ctx.uv = uv;
        return ctx;
    }

    Material material;
    SampledWavelengths lambda;
    SampledSpectrum beta, r_u;
    Point3f p;
    Vector3f wo;
    Normal3f n, ns;
    Vector3f dpdus;
    Point2f uv;
    int depth;
    MediumInterface mediumInterface;
    Float etaScale;
    int pixelIndex;
};

// SubsurfaceScatterWorkItem Definition
struct SubsurfaceScatterWorkItem {
    Point3f p0, p1;
    int depth;
    Material material;
    TabulatedBSSRDF bssrdf;
    SampledWavelengths lambda;
    SampledSpectrum beta, r_u;
    Float reservoirPDF;
    Float uLight;
    SubsurfaceInteraction ssi;
    MediumInterface mediumInterface;
    Float etaScale;
    int pixelIndex;
};

// MediumSampleWorkItem Definition
struct MediumSampleWorkItem {
    // Both enqueue types (have mtl and no hit)
    Ray ray;
    int depth;
    Float tMax;
    SampledWavelengths lambda;
    SampledSpectrum beta;
    SampledSpectrum r_u;
    SampledSpectrum r_l;
    int pixelIndex;
    LightSampleContext prevIntrCtx;
    int specularBounce;
    int anyNonSpecularBounces;
    Float etaScale;

    // Have a hit material as well
    Light areaLight;
    Point3fi pi;
    Normal3f n;
    Vector3f dpdu, dpdv;
    Vector3f wo;
    Point2f uv;
    Material material;
    Normal3f ns;
    Vector3f dpdus, dpdvs;
    Normal3f dndus, dndvs;
    int faceIndex;
    MediumInterface mediumInterface;
};

// MediumScatterWorkItem Definition
template <typename PhaseFunction>
struct MediumScatterWorkItem {
    Point3f p;
    int depth;
    SampledWavelengths lambda;
    SampledSpectrum beta, r_u;
    const PhaseFunction *phase;
    Vector3f wo;
    Float time;
    Float etaScale;
    Medium medium;
    int pixelIndex;
};

// MaterialEvalWorkItem Definition
template <typename ConcreteMaterial>
struct MaterialEvalWorkItem {
    // MaterialEvalWorkItem Public Methods
    PBRT_CPU_GPU
    NormalBumpEvalContext GetNormalBumpEvalContext(Float dudx, Float dudy, Float dvdx,
                                                   Float dvdy) const {
        NormalBumpEvalContext ctx;
        ctx.p = Point3f(pi);
        ctx.uv = uv;
        ctx.dudx = dudx;
        ctx.dudy = dudy;
        ctx.dvdx = dvdx;
        ctx.dvdy = dvdy;
        ctx.shading.n = ns;
        ctx.shading.dpdu = dpdus;
        ctx.shading.dpdv = dpdvs;
        ctx.shading.dndu = dndus;
        ctx.shading.dndv = dndvs;
        ctx.faceIndex = faceIndex;
        return ctx;
    }

    PBRT_CPU_GPU
    MaterialEvalContext GetMaterialEvalContext(Float dudx, Float dudy, Float dvdx,
                                               Float dvdy, Normal3f ns,
                                               Vector3f dpdus) const {
        MaterialEvalContext ctx;
        ctx.wo = wo;
        ctx.n = n;
        ctx.ns = ns;
        ctx.dpdus = dpdus;
        ctx.p = Point3f(pi);
        ctx.uv = uv;
        ctx.dudx = dudx;
        ctx.dudy = dudy;
        ctx.dvdx = dvdx;
        ctx.dvdy = dvdy;
        ctx.faceIndex = faceIndex;
        return ctx;
    }

    // MaterialEvalWorkItem Public Members
    const ConcreteMaterial *material;
    Point3fi pi;
    Normal3f n;
    Vector3f dpdu, dpdv;
    Float time;
    int depth;
    Normal3f ns;
    Vector3f dpdus, dpdvs;
    Normal3f dndus, dndvs;
    Point2f uv;
    int faceIndex;
    SampledWavelengths lambda;
    int pixelIndex;
    int anyNonSpecularBounces;
    Vector3f wo;
    SampledSpectrum beta, r_u;
    Float etaScale;
    MediumInterface mediumInterface;
};

#include "wavefront_workitems_soa.h"

// RayQueue Definition
class RayQueue : public WorkQueue<RayWorkItem> {
  public:
    using WorkQueue::WorkQueue;
    // RayQueue Public Methods
    PBRT_CPU_GPU
    int PushCameraRay(const Ray &ray, const SampledWavelengths &lambda, int pixelIndex);

    PBRT_CPU_GPU
    int PushIndirectRay(const Ray &ray, int depth, const LightSampleContext &prevIntrCtx,
                        const SampledSpectrum &beta, const SampledSpectrum &r_u,
                        const SampledSpectrum &r_l, const SampledWavelengths &lambda,
                        Float etaScale, bool specularBounce, bool anyNonSpecularBounces,
                        int pixelIndex);
};

// RayQueue Inline Methods
inline int RayQueue::PushCameraRay(const Ray &ray, const SampledWavelengths &lambda,
                                   int pixelIndex) {
    int index = AllocateEntry();
    DCHECK(!ray.HasNaN());
    this->ray[index] = ray;
    this->depth[index] = 0;
    this->pixelIndex[index] = pixelIndex;
    this->lambda[index] = lambda;
    this->beta[index] = SampledSpectrum(1.f);
    this->etaScale[index] = 1.f;
    this->anyNonSpecularBounces[index] = false;
    this->r_u[index] = SampledSpectrum(1.f);
    this->r_l[index] = SampledSpectrum(1.f);
    this->specularBounce[index] = false;
    return index;
}

PBRT_CPU_GPU
inline int RayQueue::PushIndirectRay(
    const Ray &ray, int depth, const LightSampleContext &prevIntrCtx,
    const SampledSpectrum &beta, const SampledSpectrum &r_u,
    const SampledSpectrum &r_l, const SampledWavelengths &lambda, Float etaScale,
    bool specularBounce, bool anyNonSpecularBounces, int pixelIndex) {
    int index = AllocateEntry();
    DCHECK(!ray.HasNaN());
    this->ray[index] = ray;
    this->depth[index] = depth;
    this->pixelIndex[index] = pixelIndex;
    this->prevIntrCtx[index] = prevIntrCtx;
    this->beta[index] = beta;
    this->r_u[index] = r_u;
    this->r_l[index] = r_l;
    this->lambda[index] = lambda;
    this->anyNonSpecularBounces[index] = anyNonSpecularBounces;
    this->specularBounce[index] = specularBounce;
    this->etaScale[index] = etaScale;
    return index;
}

// ShadowRayQueue Definition
using ShadowRayQueue = WorkQueue<ShadowRayWorkItem>;

// EscapedRayQueue Definition
class EscapedRayQueue : public WorkQueue<EscapedRayWorkItem> {
  public:
    // EscapedRayQueue Public Methods
    PBRT_CPU_GPU
    int Push(RayWorkItem r);

    using WorkQueue::WorkQueue;

    using WorkQueue::Push;
};

inline int EscapedRayQueue::Push(RayWorkItem r) {
    return Push(EscapedRayWorkItem{r.ray.o, r.ray.d, r.depth, r.lambda, r.pixelIndex,
                                   r.beta, (int)r.specularBounce, r.r_u, r.r_l,
                                   r.prevIntrCtx});
}

// GetBSSRDFAndProbeRayQueue Definition
class GetBSSRDFAndProbeRayQueue : public WorkQueue<GetBSSRDFAndProbeRayWorkItem> {
  public:
    using WorkQueue::WorkQueue;

    PBRT_CPU_GPU
    int Push(Material material, SampledWavelengths lambda, SampledSpectrum beta,
             SampledSpectrum r_u, Point3f p, Vector3f wo, Normal3f n, Normal3f ns,
             Vector3f dpdus, Point2f uv, int depth, MediumInterface mediumInterface,
             Float etaScale, int pixelIndex) {
        int index = AllocateEntry();
        this->material[index] = material;
        this->lambda[index] = lambda;
        this->beta[index] = beta;
        this->r_u[index] = r_u;
        this->p[index] = p;
        this->wo[index] = wo;
        this->n[index] = n;
        this->ns[index] = ns;
        this->dpdus[index] = dpdus;
        this->uv[index] = uv;
        this->depth[index] = depth;
        this->mediumInterface[index] = mediumInterface;
        this->etaScale[index] = etaScale;
        this->pixelIndex[index] = pixelIndex;
        return index;
    }
};

// SubsurfaceScatterQueue Definition
class SubsurfaceScatterQueue : public WorkQueue<SubsurfaceScatterWorkItem> {
  public:
    using WorkQueue::WorkQueue;

    PBRT_CPU_GPU
    int Push(Point3f p0, Point3f p1, int depth, Material material, TabulatedBSSRDF bssrdf,
             SampledWavelengths lambda, SampledSpectrum beta, SampledSpectrum r_u,
             MediumInterface mediumInterface, Float etaScale, int pixelIndex) {
        int index = AllocateEntry();
        this->p0[index] = p0;
        this->p1[index] = p1;
        this->depth[index] = depth;
        this->material[index] = material;
        this->bssrdf[index] = bssrdf;
        this->lambda[index] = lambda;
        this->beta[index] = beta;
        this->r_u[index] = r_u;
        this->mediumInterface[index] = mediumInterface;
        this->etaScale[index] = etaScale;
        this->pixelIndex[index] = pixelIndex;
        return index;
    }
};

// MediumSampleQueue Definition
class MediumSampleQueue : public WorkQueue<MediumSampleWorkItem> {
  public:
    using WorkQueue::WorkQueue;

    using WorkQueue::Push;

    PBRT_CPU_GPU
    int Push(Ray ray, Float tMax, SampledWavelengths lambda, SampledSpectrum beta,
             SampledSpectrum r_u, SampledSpectrum r_l, int pixelIndex,
             LightSampleContext prevIntrCtx, int specularBounce,
             int anyNonSpecularBounces, Float etaScale) {
        int index = AllocateEntry();
        this->ray[index] = ray;
        this->tMax[index] = tMax;
        this->lambda[index] = lambda;
        this->beta[index] = beta;
        this->r_u[index] = r_u;
        this->r_l[index] = r_l;
        this->pixelIndex[index] = pixelIndex;
        this->prevIntrCtx[index] = prevIntrCtx;
        this->specularBounce[index] = specularBounce;
        this->anyNonSpecularBounces[index] = anyNonSpecularBounces;
        this->etaScale[index] = etaScale;
        return index;
    }

    PBRT_CPU_GPU
    int Push(RayWorkItem r, Float tMax) {
        return Push(r.ray, tMax, r.lambda, r.beta, r.r_u, r.r_l, r.pixelIndex,
                    r.prevIntrCtx, r.specularBounce, r.anyNonSpecularBounces, r.etaScale);
    }
};

// MediumScatterQueue Definition
using MediumScatterQueue = MultiWorkQueue<
    typename MapType<MediumScatterWorkItem, typename PhaseFunction::Types>::type>;

// MaterialEvalQueue Definition
using MaterialEvalQueue = MultiWorkQueue<
    typename MapType<MaterialEvalWorkItem, typename Material::Types>::type>;

}  // namespace pbrt

#endif  // PBRT_WAVEFRONT_WORKITEMS_H
