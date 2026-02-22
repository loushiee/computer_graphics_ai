/////////////////////////////////////////////////////
//// CS 8803/4803 CGAI: Computer Graphics in AI Era
//// Assignment 3A: Volumetric Ray Tracing
//// Cloud renderer adapted from https://www.shadertoy.com/view/7sSGRV
/////////////////////////////////////////////////////

precision highp float;

uniform vec2 iResolution;            //// screen resolution
uniform float iTime;                 //// time elapsed
uniform sampler2D uNoise;            //// noise texture (2D)
uniform sampler2D uBlueNoise;        //// blue noise texture (2D)
uniform int iFrame;                  //// frame index
uniform vec4 iMouse;                 //// mouse input (x, y, clickX, clickY)

#define PI 3.14159265359

#define BBOX vec3(2.0, 2.0, 2.0)
#define SUN_DIR normalize(vec3(1.0, 0.8, 0.2))

/////////////////////////////////////////////////////
//// tone mapping
//// This function maps original HDR color to displayable LDR color.
//// Adapted from https://www.shadertoy.com/view/7sSGR
/////////////////////////////////////////////////////

vec3 aces_tonemap(vec3 color) {
    mat3 m1 = mat3(0.59719, 0.07600, 0.02840, 0.35458, 0.90834, 0.13383, 0.04823, 0.01566, 0.83777);
    mat3 m2 = mat3(1.60475, -0.10208, -0.00327, -0.53108, 1.10813, -0.07276, -0.07367, -0.00605, 1.07602);
    vec3 v = m1 * color;
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return pow(clamp(m2 * (a / b), 0.0, 1.0), vec3(1.0 / 2.2));
}

/////////////////////////////////////////////////////
//// density-to-color conversion
/////////////////////////////////////////////////////

//// Inigo Quilez - https://iquilezles.org/articles/palettes/
//// This function converts a scalar value t to a color
vec3 palette(in float t) {
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(1.0, 1.0, 1.0);
    vec3 d = vec3(0.0, 0.10, 0.20);

    return a + b * cos(6.28318 * (c * t + d));
}

/////////////////////////////////////////////////////
//// utilities
/////////////////////////////////////////////////////

float opSmoothUnion(float d1, float d2, float k) {
    float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}

/////////////////////////////////////////////////////
//// Step 1: Ray–AABB intersection
//// You are asked to compute the intersection interval [tNear, tFar]
//// between a ray and an axis-aligned bounding box (AABB).
////
//// The purpose of this function is to restrict volumetric sampling
//// to a finite region of interest, which significantly accelerates
//// volumetric rendering by avoiding unnecessary sampling outside the volume.
////
//// The ray is defined as:
////     r(t) = rayOrigin + t * rayDir
////
//// The AABB is defined by its minimum corner boxMin
//// and maximum corner boxMax.
/////////////////////////////////////////////////////

vec2 intersectAABB(vec3 rayOrigin, vec3 rayDir, vec3 boxMin, vec3 boxMax) {
    float tNear = 0.0;
    float tFar = 1.0;

    //// TODO Step 1: Ray–AABB intersection
    //// your implementation starts
    vec3 t1 = (boxMin - rayOrigin) / rayDir;
    vec3 t2 = (boxMax - rayOrigin) / rayDir;
    vec3 tmin = min(t1, t2);
    vec3 tmax = max(t1, t2);
    tNear = max(tmin.x, max(tmin.y, tmin.z));
    tFar = min(tmax.x, min(tmax.y, tmax.z));
    return vec2(tNear, tFar);
    //// your implementation ends

    return vec2(tNear, tFar);
}

// 3D value noise using a 2D noise texture.
float noise3(in vec3 x) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    vec2 uv = (p.xy + vec2(37.0, 239.0) * p.z) + f.xy;
    vec2 tex = textureLod(uNoise, (uv + 0.5) / 256.0, 0.0).yx;
    return mix(tex.x, tex.y, f.z) * 2.0 - 1.0;
}

// Blue-noise with per-frame rotation.
//// Adapted from https://www.shadertoy.com/view/7sSGR
float blueNoise(vec2 pixelPos) {
    const float c_goldenRatioConjugate = 0.61803398875;
    float val = texture2D(uBlueNoise, (pixelPos) / 1024.0).r;
    int frame = iFrame % 64;
    val = fract(val + float(frame) * c_goldenRatioConjugate);
    return val;
}

/////////////////////////////////////////////////////
//// background sky
//// Adapted from https://www.shadertoy.com/view/7sSGR
/////////////////////////////////////////////////////

vec3 skyColor(vec3 viewDir, vec3 camPos) {
    float horizonNoise = smoothstep(-0.2, 0.9, noise3(viewDir * vec3(8.0, 36.0, 8.0)) +
        noise3(viewDir * vec3(28.0, 56.0, 28.0)) * 0.5);

    float skyExtinction1 = exp(-max(0.0, abs(viewDir.y) * 28.0 - horizonNoise * 1.8));
    float skyExtinction2 = exp(-max(0.0, abs(viewDir.y) * 12.0 - horizonNoise * 0.5));

    vec3 sky = vec3(7.5, 7.2, 7.2) *
        mix(vec3(0.12, 0.85, 1.0), vec3(1.0), skyExtinction1) *
        mix(vec3(0.5, 0.25, 0.52) * 0.5, vec3(1.0), skyExtinction2);

    sky += vec3(3.0, 2.0, 0.5) * pow(clamp(dot(viewDir, SUN_DIR), 0.0, 1.0), 4.0);
    sky += vec3(50.0, 40.0, 2.0) * smoothstep(0.991, 0.997, dot(viewDir, SUN_DIR)) * 0.3;

    return sky * 0.65;
}

/////////////////////////////////////////////////////
//// density field
//// Adapted from https://www.shadertoy.com/view/7sSGR
//// For density, we assume it is alway 1.0 inside the object (sdf < 0) and 0.0 outside the object (sdf >= 0).
/////////////////////////////////////////////////////

float density_cloud(vec3 pos) {
    vec3 normDist = pos / (vec3(1.0, 0.4, 1.0));

    float topClip = smoothstep(-0.0, 1.0, normDist.y);
    float bottomClip = smoothstep(-0.5, -1.0, normDist.y);

    float sideClipX = smoothstep(0.60 - topClip * 0.6, 1.0, abs(normDist.x));
    float sideClipZ = smoothstep(0.60 - topClip * 0.6, 1.0, abs(normDist.z));

    float freq = 4.9;
    float amp = 0.5;

    float d1 = distance(normDist * vec3(1.0, 0.9, 1.0), vec3(0.0, -0.65, 0.0) * vec3(1.0, 0.8, 1.0));
    float d2 = distance(normDist * vec3(1.6, 1.0, 1.6), vec3(-0.3, 0.2, 0.3));

    float dens = 1.0 - opSmoothUnion(pow(d1 * 1.1, 1.2), pow(d2 * 1.1, 1.4), 0.5);

    pos += vec3(1.0, 0.0, 2.0);

    for(int i = 1; i < 4; i++) {
        float timeOffs = iTime * (1.0 + float(i) * 1.0) * 0.4;
        if(i == 1)
            timeOffs = 0.0;

        vec3 sineCoeff = vec3(pos.z * 2.4 * freq, pos.x * 2.6 * freq + timeOffs, pos.y * 2.8 * freq);
        vec3 randOffs = sin(sineCoeff);

        dens += (abs(noise3(pos * freq + vec3(float(i)) * 167.0 + randOffs * 0.022 * freq)) - 0.48) * amp;

        freq *= 2.1;
        amp *= 0.5;
    }

    dens = dens * 0.8 - pow(sideClipX, 10.5) * 0.4 - pow(sideClipZ, 10.5) * 0.4 - pow(topClip, 13.5) * 0.3 - pow(bottomClip, 25.0) * 0.5;

    return max(0.0, dens);
}

/////////////////////////////////////////////////////
//// Creative Expression: Ring Nebula
//// Inspired by images in https://www.forbes.com/sites/startswithabang/2019/06/03/the-ring-nebula-is-much-much-more-than-a-ring/
/////////////////////////////////////////////////////

float nebulaCoreRadius = 0.4;
float nebulaOuterRadius = 0.2;
float cloudRadius = 0.4;

float density_nebula(vec3 pos)
{
    float baseRadius = nebulaCoreRadius;
    float variation  = 0.1;

    float r = length(pos);
    vec3 dir = normalize(pos);

    float dirNoise = noise3(dir * 3.0);
    float coreRadius = baseRadius + dirNoise * variation;

    float shellRadius = coreRadius + nebulaOuterRadius;

    // Inner core
    float core = 1.0 - smoothstep(coreRadius - 0.2,
                                  coreRadius,
                                  r);

    // Outer core
    float shell = 1.0 - smoothstep(shellRadius - 0.25,
                                   shellRadius,
                                   r);

    shell *= (1.0 - core);

    // Outer halo (cloud)
    float haloMask = smoothstep(shellRadius,
                                shellRadius + cloudRadius,
                                r);
    float halo = 0.0;
    if (r > shellRadius * 0.8) {
        halo = density_cloud(pos * 0.65);
    }

    float dens = core * 1.5
               + shell * 0.35
               + halo * 0.7;

    return max(dens, 0.0);
}

vec4 sampleVolumeNebula(vec3 p, vec2 fragCoord)
{
    float dens = density_nebula(p);
    if (dens <= 0.0)
        return vec4(0.0);

    float r = length(p);

    float baseRadius = nebulaCoreRadius;
    float variation  = 0.1;

    vec3 dir = normalize(p);
    float dirNoise = noise3(dir * 3.0);
    float coreRadius = baseRadius + dirNoise * variation;
    float shellRadius = coreRadius + nebulaOuterRadius;

    float coreMask = 1.0 - smoothstep(coreRadius - 0.2,
                                      coreRadius,
                                      r);

    float shellMask = 1.0 - smoothstep(shellRadius - 0.25,
                                       shellRadius,
                                       r);
    shellMask *= (1.0 - coreMask);

    float haloMask = 0.0;
    if (r > shellRadius)
    {
        haloMask = smoothstep(shellRadius,
                              shellRadius + cloudRadius,
                              r);
    }

    float haloOuterRadius = shellRadius + cloudRadius;

    // Thin outer rim (0.01 thickness)
    float rimMask = smoothstep(haloOuterRadius - 0.01,
                               haloOuterRadius,
                               r);
    rimMask *= haloMask;

    // ---- Colors ----
    vec3 coreColor = vec3(0, 0.1, 0.6);
    vec3 lightGreen = vec3(0.4, 1.0, 0.6);
    vec3 nearWhite  = vec3(0.95, 1.0, 0.95);
    vec3 rimRed = vec3(1.0, 0.25, 0.25);
    float n = noise3(p * 2.0);
    float whiteness = n * n;
    vec3 shellColor = mix(lightGreen, nearWhite, whiteness * 0.6);
    
    // Halo color from density_cloud modulation
    float fbmVal = 0.0;
    float amp = 0.5;
    float freq = 2.0;

    for (int i = 0; i < 4; i++)
    {
        fbmVal += amp * noise3(p * freq);
        freq *= 2.0;
        amp *= 0.5;
    }

    float yellowWhiteMix = smoothstep(0.3, 0.8, fbmVal);

    vec3 yellowColor = vec3(1.0, 0.9, 0.25);
    vec3 whiteColor  = vec3(1.0, 0.95, 0.9);

    vec3 haloColor = mix(yellowColor, whiteColor, yellowWhiteMix);
    vec3 haloFinalColor = mix(haloColor, rimRed, rimMask);

    vec3 color =
        coreMask  * coreColor
        + shellMask * shellColor
        + haloMask  * haloFinalColor;

    float glow = exp(-r * 3.0);
    vec3 rgb = dens * color * (1.4 + glow * 1.2);

    float sigma = dens * 10.0;

    return vec4(rgb, sigma);
}

// From Flickering Stars https://www.shadertoy.com/view/NtsBzB
vec3 hash( vec3 p ) // replace this by something better
{
	p = vec3( dot(p,vec3(127.1,311.7, 74.7)),
			  dot(p,vec3(269.5,183.3,246.1)),
			  dot(p,vec3(113.5,271.9,124.6)));

	return -1.0 + 2.0*fract(sin(p)*43758.5453123);
}
float noise( in vec3 p )
{
    vec3 i = floor( p );
    vec3 f = fract( p );
	
	vec3 u = f*f*(3.0-2.0*f);

    return mix( mix( mix( dot( hash( i + vec3(0.0,0.0,0.0) ), f - vec3(0.0,0.0,0.0) ), 
                          dot( hash( i + vec3(1.0,0.0,0.0) ), f - vec3(1.0,0.0,0.0) ), u.x),
                     mix( dot( hash( i + vec3(0.0,1.0,0.0) ), f - vec3(0.0,1.0,0.0) ), 
                          dot( hash( i + vec3(1.0,1.0,0.0) ), f - vec3(1.0,1.0,0.0) ), u.x), u.y),
                mix( mix( dot( hash( i + vec3(0.0,0.0,1.0) ), f - vec3(0.0,0.0,1.0) ), 
                          dot( hash( i + vec3(1.0,0.0,1.0) ), f - vec3(1.0,0.0,1.0) ), u.x),
                     mix( dot( hash( i + vec3(0.0,1.0,1.0) ), f - vec3(0.0,1.0,1.0) ), 
                          dot( hash( i + vec3(1.0,1.0,1.0) ), f - vec3(1.0,1.0,1.0) ), u.x), u.y), u.z );
}
vec3 backgroundStars(vec2 uv)
{
    vec3 stars_direction = normalize(vec3(uv * 2.0f - 1.0f, 1.0f)); // could be view vector for example
	float stars_threshold = 8.0f; // modifies the number of stars that are visible
	float stars_exposure = 200.0f; // modifies the overall strength of the stars
	float stars = pow(clamp(noise(stars_direction * 200.0f), 0.0f, 1.0f), stars_threshold) * stars_exposure;
	stars *= mix(0.4, 1.4, noise(stars_direction * 100.0f + vec3(iTime))); // time based flickering
	return vec3(stars);
}


float sdBox(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}

float density_cube(vec3 pos) {
    float d = sdBox(pos, vec3(0.2));
    return d < 0.0 ? 1.0 : 0.0;
}

float density(vec3 p, int model) {
    if(model == 0) {
        return density_cube(p);
    } else if(model == 1) {
        return density_cloud(p);
    } else if(model == 2) {
        return density_nebula(p);
    } else {
        return 0.0;
    }
}

/////////////////////////////////////////////////////
//// light march: approximate transmittance toward the sun
//// Adapted from https://www.shadertoy.com/view/7sSGR
/////////////////////////////////////////////////////

float lightmarch(vec3 p, vec2 fragCoord, int model) {
    float lightDirMult = 0.9 + blueNoise(fragCoord + vec2(50.0)) * 0.2;

    float light = 0.0;
    vec3 lightP = p;

    float lightStepInc = 2.0;
    float lightStep = 0.002;

    for(int n = 0; n < 8; n++) {
        float lightDens = density(lightP, model);
        light += lightDens * lightStep * 20.0;

        vec3 scatterVec = normalize(SUN_DIR);
        lightP += scatterVec * lightStep * lightDirMult;

        lightStep *= lightStepInc;
    }

    return exp(-light * light * 12.0);
}

/////////////////////////////////////////////////////
//// sampleVolume: returns (rgb, sigma) at point p
//// - rgb: emitted/scattered color at this point
//// - sigma: extinction coefficient for Beer-Lambert alpha computation
//// This rgb resutls can be considered as a mapping from density to color, which you can design as you like.
//// Adapted from https://www.shadertoy.com/view/7sSGR
/////////////////////////////////////////////////////

vec4 sampleVolume(vec3 p, vec2 fragCoord, int model) {
    float dens = density(p, model);
    if(dens <= 0.0)
        return vec4(0.0);

    float lightT = lightmarch(p, fragCoord, model);

    float ambSoft = exp(-dens * 5.0) * 0.9 + 0.1;
    vec3 ambient = mix(vec3(0.15, 0.25, 0.55), vec3(0.58, 0.85, 1.02), ambSoft);

    vec3 sun_color = vec3(2.0, 1.6, 0.9) * 2.3;
    vec3 rgb = vec3(0.7, 0.85, 1.0) * (lightT * sun_color + ambient * 1.4);

    float sigma = dens * 8.0;
    return vec4(rgb, sigma);
}


/////////////////////////////////////////////////////
//// volumetric rendering
/////////////////////////////////////////////////////

/////////////////////////////////////////////////////
//// Step 2: Front-to-back volumetric rendering
//// You are asked to implement the front-to-back volumetric ray tracing algorithm to accummulate colors along each ray. 
//// Your task is to accumulate color and transmittance along the ray based on the absorption-emission volumetric model.
/////////////////////////////////////////////////////

vec4 volumeRenderFrontToBack(vec3 ro, vec3 rd, vec2 fragCoord, int n_samples, int model) {
    vec3 color = vec3(0.0);
    float T = 1.0;

    /// Here we accelerate ray-AABB intersection to find the near and far bounds.
    vec2 box = intersectAABB(ro, rd, -BBOX * 0.5, BBOX * 0.5);
    if(box.x > box.y)
        return vec4(0.0);
    float tNear = max(box.x, 0.0);
    float tFar = box.y;
    float dt = (tFar - tNear) / float(n_samples);

    //// Blue-noise jitter: shift the sampling positions by a small random offset.
    //// This breaks up regular banding/stratification artifacts caused by uniform ray-march steps,
    //// turning structured aliasing into less noticeable noise (especially for smooth volumes like clouds).
    float jitter = blueNoise(fragCoord + vec2(123.0));

    for(int i = 0; i < n_samples; i++) {
        float t = tNear + (float(i) + jitter) * dt;
        vec3 p = ro + t * rd;

        //// color and density sample at p
        vec4 s = sampleVolume(p, fragCoord, model);

        //// TODO Step 2: Front-to-Back volumetric rendering
        //// your implementation starts
        float Ti = exp(-s.a * dt);
        color += T * (1.0 - Ti) * s.rgb;
        T *= Ti;
        //// your implementation ends
    }

    return vec4(color, 1.0 - T);
}

/////////////////////////////////////////////////////
//// Step 3: Back-to-front volumetric rendering
/////////////////////////////////////////////////////

vec4 volumeRenderBackToFront(vec3 ro, vec3 rd, vec2 fragCoord, int n_samples, int model) {
    vec3 color = vec3(0.0);
    float T = 1.0;

    //// TODO Step 3: Back-to-front volumetric rendering
    //// your implementation starts
    // Copy code from volumeRenderFrontToBack
    vec2 box = intersectAABB(ro, rd, -BBOX * 0.5, BBOX * 0.5);
    if(box.x > box.y)
        return vec4(0.0);
    float tNear = max(box.x, 0.0);
    float tFar = box.y;
    float dt = (tFar - tNear) / float(n_samples);
    float jitter = blueNoise(fragCoord + vec2(123.0));

    // Flip the loop order for back-to-front rendering
    for(int i = n_samples - 1; i >= 0; i--) {
        float t = tNear + (float(i) + jitter) * dt;
        vec3 p = ro + t * rd;

        //// color and density sample at p
        vec4 s = vec4(0.0);
        if(model == 2) {
            s = sampleVolumeNebula(p, fragCoord);
        } else {
            s = sampleVolume(p, fragCoord, model);
        }

        float Ti = exp(-s.a * dt);
        vec3 Di = s.rgb * (1.0 - Ti);
        color = Di + (Ti * color);
        T *= Ti;
    }
    //// your implementation ends

    return vec4(color, 1.0 - T);
}

void mainImageCreative(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;

    //// Camera
    float camX = cos(iTime * 0.13 - 1.5);
    float camZ = sin(iTime * 0.13 - 1.5);
    float camYOff = sin(iTime * 0.3) * 0.28;

    vec3 ro = normalize(vec3(camX, camYOff - 0.03, camZ)) * 2.4;

    vec3 cameraDir = -normalize(ro + vec3(0.0, ro.y * 0.30 + 0.11, 0.0));
    vec3 cameraRight = normalize(cross(vec3(0.0, 1.0, 0.0), cameraDir));
    vec3 cameraUp = normalize(cross(cameraDir, cameraRight));

    vec3 rd = normalize(cameraDir + cameraUp * uv.y * 0.8 + cameraRight * uv.x * 0.8);

    //// background sky
    vec2 uvStars = fragCoord/iResolution.xy;
    vec3 bgStars = backgroundStars(uvStars);

    //// volumetric rendering
    int model = 2;
    int n_samples = 128;
    vec4 vol = volumeRenderBackToFront(ro, rd, fragCoord, n_samples, model);

    vec3 finalCol = mix(bgStars, vol.rgb, clamp(vol.a, 0.0, 1.0));

    fragColor = vec4(finalCol, 1.0);
}


/////////////////////////////////////////////////////
//// mainImage
/////////////////////////////////////////////////////

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;

    //// Camera
    float camX = cos(iTime * 0.13 - 1.5);
    float camZ = sin(iTime * 0.13 - 1.5);
    float camYOff = sin(iTime * 0.3) * 0.28;

    vec3 ro = normalize(vec3(camX, camYOff - 0.03, camZ)) * 2.4;

    vec3 cameraDir = -normalize(ro + vec3(0.0, ro.y * 0.30 + 0.11, 0.0));
    vec3 cameraRight = normalize(cross(vec3(0.0, 1.0, 0.0), cameraDir));
    vec3 cameraUp = normalize(cross(cameraDir, cameraRight));

    vec3 rd = normalize(cameraDir + cameraUp * uv.y * 0.8 + cameraRight * uv.x * 0.8);

    //// background sky
    vec3 sky = skyColor(rd, ro);

    //// volumetric rendering
    //// model selects different density fields; 0 = cube, 1 = cloud
    int model = 1;
    int n_samples = 128;
    //// TODO Step 3: Back-to-Front volumetric rendering
    //// Switch between front-to-back and back-to-front rendering here
    // vec4 vol = volumeRenderFrontToBack(ro, rd, fragCoord, n_samples, model);
    vec4 vol = volumeRenderBackToFront(ro, rd, fragCoord, n_samples, model);

    //// composite cloud over sky
    vec3 finalCol = mix(sky, vol.rgb, clamp(vol.a * 1.3, 0.0, 1.0));

    //// tonemap
    vec3 outCol = aces_tonemap(finalCol);

    fragColor = vec4(outCol, 1.0);
}

void main() {
    // mainImage(gl_FragColor, gl_FragCoord.xy);
    mainImageCreative(gl_FragColor, gl_FragCoord.xy);
}
