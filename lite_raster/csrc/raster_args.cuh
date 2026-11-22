#pragma once

enum Feature {
    Inverse = 1, Hard = 2, Expected = 4, Median = 8, Normal = 16, Rade = 32, HardDepth = 64,
    ReturnAlpha = 128, ReturnHardAlpha = 256, ReturnHit = 512, RawNormal = 1024
};
enum Input { XY, Conic, Opacity, Color, CenterZ, Plane, Normals, Background, K, Offsets, Ids, NInputs };
enum Output { RGB, Alpha, InvZ, HardInvZ, HardAlpha, ExpZ, MedZ, Nrm, HardZ, NOutputs };
enum Cache { Last, LastHard, MedId, NormLength, Trans, HardTrans, Hit, NCache };
struct Args {
    void *in[NInputs], *out[NOutputs], *cache[NCache], *dout[NOutputs], *din[8], *absgrad, *moments;
    const bool *absgrad_candidates;
    int width, height, count;
};
