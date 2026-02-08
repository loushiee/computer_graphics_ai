/////////////////////////////////////////////////////
//// CS 8803/4803 CGAI: Computer Graphics in AI Era
//// Assignment 2A: SDF and Ray Marching
/////////////////////////////////////////////////////

precision highp float;              //// set default precision of float variables to high precision

varying vec2 vUv;                   //// screen uv coordinates (varying, from vertex shader)
uniform vec2 iResolution;           //// screen resolution (uniform, from CPU)
uniform float iTime;                //// time elapsed (uniform, from CPU)

const vec3 CAM_POS = vec3(-0.35, 1.0, -3.0);
// float sdf2(vec3 p);

/////////////////////////////////////////////////////
//// sdf functions
/////////////////////////////////////////////////////

/////////////////////////////////////////////////////
//// Step 1: sdf primitives
//// You are asked to implement sdf primitive functions for sphere, plane, and box.
//// In each function, you will calculate the sdf value based on the function arguments.
/////////////////////////////////////////////////////

//// sphere: p - query point; c - sphere center; r - sphere radius
float sdfSphere(vec3 p, vec3 c, float r)
{
    //// your implementation starts
    
    return length(p - c) - r;

    //// your implementation ends
}

//// plane: p - query point; h - height
float sdfPlane(vec3 p, float h)
{
    //// your implementation starts
    
    return p.y - h;
    
    //// your implementation ends
}

//// box: p - query point; c - box center; b - box half size (i.e., the box size is (2*b.x, 2*b.y, 2*b.z))
float sdfBox(vec3 p, vec3 c, vec3 b)
{
    //// your implementation starts
    
    vec3 d = abs(p - c) - b;
    float outsideDist = length(max(d, 0.));
    float insideDist = min(max(max(d.x, d.y), d.z), 0.0);
    return outsideDist + insideDist;
    
    //// your implementation ends
}

// From https://iquilezles.org/articles/distfunctions/
float sdCapsule(vec3 p, vec3 a, vec3 b, float r)
{
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

// From https://iquilezles.org/articles/distfunctions/
float sdCappedCylinder( vec3 p, float r, float h )
{
  vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(r,h);
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

// From https://iquilezles.org/articles/distfunctions/
float sdCone( vec3 p, vec2 c, float h )
{
  // c is the sin/cos of the angle, h is height
  // Alternatively pass q instead of (c,h),
  // which is the point at the base in 2D
  vec2 q = h*vec2(c.x/c.y,-1.0);
    
  vec2 w = vec2( length(p.xz), p.y );
  vec2 a = w - q*clamp( dot(w,q)/dot(q,q), 0.0, 1.0 );
  vec2 b = w - q*vec2( clamp( w.x/q.x, 0.0, 1.0 ), 1.0 );
  float k = sign( q.y );
  float d = min(dot( a, a ),dot(b, b));
  float s = max( k*(w.x*q.y-w.y*q.x),k*(w.y-q.y)  );
  return sqrt(d)*sign(s);
}

float dot2(vec2 v) { return dot(v, v); } 

// From https://iquilezles.org/articles/distfunctions/
float sdCappedCone(vec3 p, float h, float r1, float r2 )
{
  vec2 q = vec2( length(p.xz), p.y );
  vec2 k1 = vec2(r2,h);
  vec2 k2 = vec2(r2-r1,2.0*h);
  vec2 ca = vec2(q.x-min(q.x,(q.y<0.0)?r1:r2), abs(q.y)-h);
  vec2 cb = q - k1 + k2*clamp( dot(k1-q,k2)/dot2(k2), 0.0, 1.0 );
  float s = (cb.x<0.0 && ca.y<0.0) ? -1.0 : 1.0;
  return s*sqrt( min(dot2(ca),dot2(cb)) );
}

// From https://iquilezles.org/articles/distfunctions/
float sdEllipsoid( vec3 p, vec3 r)
{
  float k0 = length(p/r);
  float k1 = length(p/(r*r));
  return k0*(k0-1.0)/k1;
}

float sdfDeathStar(vec3 p, float ra, float rb, vec3 d)
{
    float d1 = sdfSphere(p, vec3(0), ra);
    float d2 = sdfSphere(p, d, rb);
    return max(d1, -d2);
}

float sdfCylinder(vec3 p, vec3 a, vec3 b, float r)
{
    vec3 pa = p - a;
    vec3 ba = b - a;

    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);

    // Radial distance to axis
    float dSide = length(pa - ba * h) - r;

    // Distance to caps (flat)
    float dCap = abs(dot(pa, normalize(ba))) - length(ba) * 0.5;

    return max(dSide, dCap);
}

/////////////////////////////////////////////////////
//// boolean operations
/////////////////////////////////////////////////////

/////////////////////////////////////////////////////
//// Step 2: sdf boolean operations
//// You are asked to implement sdf boolean operations for intersection, union, and subtraction.
/////////////////////////////////////////////////////

float sdfIntersection(float s1, float s2)
{
    //// your implementation starts
    
    return max(s1, s2);

    //// your implementation ends
}

float sdfUnion(float s1, float s2)
{
    //// your implementation starts
    
    return min(s1, s2);

    //// your implementation ends
}

float sdfSubtraction(float s1, float s2)
{
    //// your implementation starts
    
    return max(s1, -s2);

    //// your implementation ends
}

/////////////////////////////////////////////////////
//// sdf calculation
/////////////////////////////////////////////////////

/////////////////////////////////////////////////////
//// Step 3: scene sdf
//// You are asked to use the implemented sdf boolean operations to draw the following objects in the scene by calculating their CSG operations.
/////////////////////////////////////////////////////

//// sdf: p - query point
float sdf(vec3 p)
{
    float s = 0.;

    //// 1st object: plane
    float plane1_h = -0.1;
    
    //// 2nd object: sphere
    vec3 sphere1_c = vec3(-2.0, 1.0, 0.0);
    float sphere1_r = 0.25;

    //// 3rd object: box
    vec3 box1_c = vec3(-1.0, 1.0, 0.0);
    vec3 box1_b = vec3(0.2, 0.2, 0.2);

    //// 4th object: box-sphere subtraction
    vec3 box2_c = vec3(0.0, 1.0, 0.0);
    vec3 box2_b = vec3(0.3, 0.3, 0.3);

    vec3 sphere2_c = vec3(0.0, 1.0, 0.0);
    float sphere2_r = 0.4;

    //// 5th object: sphere-sphere intersection
    vec3 sphere3_c = vec3(1.0, 1.0, 0.0);
    float sphere3_r = 0.4;

    vec3 sphere4_c = vec3(1.3, 1.0, 0.0);
    float sphere4_r = 0.3;

    //// calculate the sdf based on all objects in the scene
    
    //// your implementation starts
    
    float sdfPlane1 = sdfPlane(p, plane1_h);
    float sdfSphere1 = sdfSphere(p, sphere1_c, sphere1_r);
    float sdfBox1 = sdfBox(p, box1_c, box1_b);
    s = sdfUnion(sdfUnion(sdfPlane1, sdfSphere1), sdfBox1);

    float sdfBox2 = sdfBox(p, box2_c, box2_b);
    float sdfSphere2 = sdfSphere(p, sphere2_c, sphere2_r);
    float sdfBoxSphereSubtraction = sdfSubtraction(sdfBox2, sdfSphere2);
    s = sdfUnion(s, sdfBoxSphereSubtraction);

    float sdfSphere3 = sdfSphere(p, sphere3_c, sphere3_r);
    float sdfSphere4 = sdfSphere(p, sphere4_c, sphere4_r);
    float sdfSphereIntersection = sdfIntersection(sdfSphere3, sdfSphere4);
    s = sdfUnion(s, sdfSphereIntersection);

    //// your implementation ends

    return s;
}

/////////////////////////////////////////////////////
//// ray marching
/////////////////////////////////////////////////////

/////////////////////////////////////////////////////
//// Step 4: ray marching
//// You are asked to implement the ray marching algorithm within the following for-loop.
/////////////////////////////////////////////////////

//// ray marching: origin - ray origin; dir - ray direction 
float rayMarching(vec3 origin, vec3 dir)
{
    float s = 0.0;
    for(int i = 0; i < 100; i++)
    {
        //// your implementation starts
        vec3 originCurrent = origin + (s * dir);
        float sCurrent = sdf(originCurrent);
        if (sCurrent < 0.001) {
            break;
        }
        s += sCurrent;
        //// your implementation ends
    }
    
    return s;
}

/////////////////////////////////////////////////////
//// normal calculation
/////////////////////////////////////////////////////

/////////////////////////////////////////////////////
//// Step 5: normal calculation
//// You are asked to calculate the sdf normal based on finite difference.
/////////////////////////////////////////////////////

//// normal: p - query point
vec3 normal(vec3 p)
{
    float s = sdf(p);          //// sdf value in p
    float dx = 0.01;           //// step size for finite difference

    //// your implementation starts
    
    float gx = sdf(p + vec3(dx, 0.0, 0.0)) - sdf(p - vec3(dx, 0.0, 0.0));
    float gy = sdf(p + vec3(0.0, dx, 0.0)) - sdf(p - vec3(0.0, dx, 0.0));
    float gz = sdf(p + vec3(0.0, 0.0, dx)) - sdf(p - vec3(0.0, 0.0, dx));
    vec3 g = vec3(gx, gy, gz) / 2.0*dx;
    return normalize(g);

    //// your implementation ends
}

/////////////////////////////////////////////////////
//// Phong shading
/////////////////////////////////////////////////////

/////////////////////////////////////////////////////
//// Step 6: lighting and coloring
//// You are asked to specify the color for each object in the scene.
//// Each object must have a separate color without mixing.
//// Notice that we have implemented the default Phong shading model for you.
/////////////////////////////////////////////////////

vec3 phong_shading(vec3 p, vec3 n)
{
    //// background
    if(p.z > 10.0){
        return vec3(0.9, 0.6, 0.2);
    }

    //// phong shading
    vec3 lightPos = vec3(4.*sin(iTime), 4., 4.*cos(iTime));  
    vec3 l = normalize(lightPos - p);               
    float amb = 0.1;
    float dif = max(dot(n, l), 0.) * 0.7;
    vec3 eye = CAM_POS;
    float spec = pow(max(dot(reflect(-l, n), normalize(eye - p)), 0.0), 128.0) * 0.9;

    vec3 sunDir = vec3(0, 1, -1);
    float sunDif = max(dot(n, sunDir), 0.) * 0.2;

    //// shadow
    float s = rayMarching(p + n * 0.02, l);
    if(s < length(lightPos - p)) dif *= .2;

    vec3 color = vec3(1.0, 1.0, 1.0);

    //// your implementation for coloring starts

    //// 1st object: plane
    float plane1_h = -0.1;
    
    //// 2nd object: sphere
    vec3 sphere1_c = vec3(-2.0, 1.0, 0.0);
    float sphere1_r = 0.25;

    //// 3rd object: box
    vec3 box1_c = vec3(-1.0, 1.0, 0.0);
    vec3 box1_b = vec3(0.2, 0.2, 0.2);

    //// 4th object: box-sphere subtraction
    vec3 box2_c = vec3(0.0, 1.0, 0.0);
    vec3 box2_b = vec3(0.3, 0.3, 0.3);

    vec3 sphere2_c = vec3(0.0, 1.0, 0.0);
    float sphere2_r = 0.4;

    //// 5th object: sphere-sphere intersection
    vec3 sphere3_c = vec3(1.0, 1.0, 0.0);
    float sphere3_r = 0.4;

    vec3 sphere4_c = vec3(1.3, 1.0, 0.0);
    float sphere4_r = 0.3;

    float distSmallest = 1e9;
    color = vec3(228, 153, 51) / 255.0;

    float sdfPlane1 = sdfPlane(p, plane1_h);
    if (sdfPlane1 < distSmallest) {
        distSmallest = sdfPlane1;
        color = vec3(228, 228, 0) / 255.0;
    }
    float sdfSphere1 = sdfSphere(p, sphere1_c, sphere1_r);
    if (sdfSphere1 < distSmallest) {
        distSmallest = sdfSphere1;
        color = vec3(255, 0, 0) / 255.0;
    }
    float sdfBox1 = sdfBox(p, box1_c, box1_b);
    if (sdfBox1 < distSmallest) {
        distSmallest = sdfBox1;
        color = vec3(0, 255, 0) / 255.0;
    }

    float sdfBox2 = sdfBox(p, box2_c, box2_b);
    float sdfSphere2 = sdfSphere(p, sphere2_c, sphere2_r);
    float sdfBoxSphereSubtraction = sdfSubtraction(sdfBox2, sdfSphere2);
    if (sdfBoxSphereSubtraction < distSmallest) {
        distSmallest = sdfBoxSphereSubtraction;
        color = vec3(0, 0, 255) / 255.0;
    }

    float sdfSphere3 = sdfSphere(p, sphere3_c, sphere3_r);
    float sdfSphere4 = sdfSphere(p, sphere4_c, sphere4_r);
    float sdfSphereIntersection = sdfIntersection(sdfSphere3, sdfSphere4);
    if (sdfSphereIntersection < distSmallest) {
        distSmallest = sdfSphereIntersection;
        color = vec3(3, 248, 251) / 255.0;
    }

    //// your implementation for coloring ends

    return (amb + dif + spec + sunDif) * color;
}

/////////////////////////////////////////////////////
//// Step 7: creative expression
//// You will create your customized sdf scene with new primitives and CSG operations in the sdf2 function.
//// Call sdf2 in your ray marching function to render your customized scene.
/////////////////////////////////////////////////////

#define MAT_SAND     0
#define MAT_IGLOO    1
#define MAT_TOWER    2
#define MAT_SUN      3
#define MAT_DEATH    4
#define MAT_R2       5
#define MAT_C3PO     6

struct SDFHit
{
    float d;
    int mat;
};

float smin(float a, float b, float k)
{
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

float noise(vec3 q)
{
    return 0.02 * sin(q.x * 8.0) * sin(q.z * 6.0);
}

SDFHit SDFHitUnion(SDFHit a, SDFHit b) 
{
    if (a.d < b.d) return a;
    return b;
}

SDFHit sand(vec3 p)
{
    float amp = 0.02;
    vec2 windDir = normalize(vec2(1.0, 0.3));
    float ripple1 = sin(dot(p.xz, windDir) * 5.0 + iTime * 0.15);
    float ripple2 = sin(dot(p.xz, vec2(-windDir.y, windDir.x)) * 3.0);
    float ripple = (ripple1 + ripple2 * 0.4) * amp;
    float d = p.y - ripple;

    return SDFHit(d, MAT_SAND);
}

SDFHit igloo(vec3 p, vec3 pos)
{
    // --------------------------------------------------
    // 1. Position the house in the world
    // --------------------------------------------------
    vec3 q = p - pos;

    // --------------------------------------------------
    // 2. Main dome (flattened sphere)
    // --------------------------------------------------
    float dome = sdfSphere(q, vec3(0.), 0.75);

    // Keep only upper half
    float domeCut = sdfPlane(q, 0.0);
    dome = sdfSubtraction(dome, domeCut);

    // --------------------------------------------------
    // 3. Base ring (thick foundation)
    // --------------------------------------------------
    vec3 sphereP = vec3(0,-0.05,0);
    float baseOuter = sdfSphere(q, sphereP, 0.85);
    float baseInner = sdfSphere(q, sphereP, 0.60);
    float base = sdfSubtraction(baseOuter, baseInner);

    // Trim base to bottom only
    base = sdfIntersection(base, sdfPlane(q, 0.01));

    // --------------------------------------------------
    // 4. Blend dome + base smoothly
    // --------------------------------------------------
    float body = smin(dome, base, 0.12);

    // --------------------------------------------------
    // 5. Entrance tunnel (horizontal capsule)
    // --------------------------------------------------
    vec3 t0 = sphereP - vec3(0.0, 0.3, 0.0);
    vec3 t1 = t0 + vec3(2.0, 0.0, -1.0);
    float tunnelOuter = sdfCylinder(q, t0, t1, 0.4);
    float tunnelInner = sdfCylinder(q, t0, t1, 0.3);
    float tunnel = sdfSubtraction(tunnelOuter, tunnelInner-0.01);

    body = smin(body, tunnel, 0.1);

    // --------------------------------------------------
    // 6. Subtle erosion / blobby noise
    // --------------------------------------------------
    body += noise(q);

    return SDFHit(body, MAT_IGLOO);
}

SDFHit tower(vec3 p, vec3 pos)
{
    float base = 100000.;
    for (int i = 0; i < 5; i++) {
        float dy = float(i) * 0.2;
        vec3 t0 = pos + vec3(0.0, dy, 0.0);
        vec3 t1 = t0 + vec3(0.0, 0.1, 0.0);
        float fat = sdfCylinder(p, t0, t1, 0.15);
        base = sdfUnion(fat, base);
    }
    float thin = sdfCylinder(p, pos, pos + vec3(0.0, 1.5, 0.0), 0.01);
    base = smin(thin, base, 0.12);
    // base = sdfUnion(thin, base);
    // return SDFHit(base, MAT_TOWER);

    vec3 q = p - pos;
    float top = sdCappedCone(q - vec3(0, 0.5, 0), 1.0, 0.2, 0.001);
    // return SDFHit(top, MAT_TOWER);

    float d = smin(base, top, 0.05);

    d += noise(q);

    return SDFHit(d, MAT_TOWER);
}

SDFHit r2d2(vec3 p, vec3 pos)
{
    vec3 t0 = pos + vec3(0.0, 0.2, 0.0);
    vec3 t1 = t0 + vec3(0.0, 0.7, 0.005);
    float body = sdfCylinder(p, t0, t1, 0.2);
    float dome = sdfSphere(p, pos + vec3(0, 0.5, 0), 0.2);
    float bodyAndDome = sdfUnion(body, dome);
    float leg1 = sdCapsule(p, pos + vec3(-0.15, 0, -0.2), pos + vec3(-0.15, 0, 0.2), 0.05);
    float leg2 = sdCapsule(p, pos + vec3(0.2, 0, -0.2), pos + vec3(0.2, 0, 0.2), 0.05);
    float legs = sdfUnion(leg1, leg2);
    float arm1 = sdCapsule(p, pos + vec3(-0.2, 0.45, -0.05), pos + vec3(-0.2, 0.2, -0.05), 0.05);
    float arm2 = sdCapsule(p, pos + vec3(0.2, 0.45, -0.05), pos + vec3(0.2, 0.2, -0.05), 0.05);
    float arms = sdfUnion(arm1, arm2);
    float d = sdfUnion(bodyAndDome, sdfUnion(legs, arms));
    vec3 q = p - pos;
    d += noise(q);

    return SDFHit(d, MAT_R2);
}

SDFHit c3po(vec3 p, vec3 pos)
{
    // float head = sdfSphere(p, pos + vec3(0, 1.16, 0), 0.12);
    float head = sdEllipsoid(p - (pos + vec3(0, 1.2, 0)), vec3(0.12, 0.16, 0.12));
    vec3 t0 = pos + vec3(0.0, 0.6, 0.0);
    vec3 t1 = t0 + vec3(0.0, 0.9, 0.0);
    float body = sdfCylinder(p, t0, t1, 0.16);
    float headAndBody = sdfUnion(head, body);
    float arm1 = sdCapsule(p, pos + vec3(-0.2, 1, 0.0), pos + vec3(-0.23, 0.5, 0.0), 0.04);
    float arm2 = sdCapsule(p, pos + vec3(0.2, 1, 0.0), pos + vec3(0.23, 0.52, 0.0), 0.04);
    float arms = sdfUnion(arm1, arm2);
    float leg1 = sdCapsule(p, pos + vec3(-0.06, 0.5, -0.05), pos + vec3(-0.07, 0.01, -0.05), 0.05);
    float leg2 = sdCapsule(p, pos + vec3(0.06, 0.5, -0.05), pos + vec3(0.07, 0.01, -0.05), 0.05);
    float legs = sdfUnion(leg1, leg2);
    float foot1 = sdCapsule(p, pos + vec3(-0.07, 0.01, -0.15), pos + vec3(-0.07, 0.01, -0.05), 0.05);
    float foot2 = sdCapsule(p, pos + vec3(0.07, 0.01, -0.15), pos + vec3(0.07, 0.01, -0.05), 0.05);
    float feet = sdfUnion(foot1, foot2);
    float d = sdfUnion(headAndBody, sdfUnion(sdfUnion(arms, legs), feet));
    vec3 q = p - pos;
    d += noise(q);
    return SDFHit(d, MAT_C3PO);
}

SDFHit deathStar(vec3 p, vec3 pos)
{
    vec3 q = p - pos;
    float d = sdfDeathStar(q, 2.0, 1.5, vec3(0., 0.2, -3.2));
    d += noise(q);
    return SDFHit(d, MAT_DEATH);
}

vec3 materialColor(int mat)
{
    if (mat == MAT_SAND)  return vec3(0.9,0.75,0.45);
    if (mat == MAT_IGLOO) return vec3(0.8,0.6,0.45);
    if (mat == MAT_TOWER) return vec3(0.3,0.3,0.35);
    if (mat == MAT_SUN)   return vec3(1.0,0.85,0.6);
    if (mat == MAT_DEATH) return vec3(0.6,0.6,0.65);
    if (mat == MAT_R2)    return vec3(0.7,0.75,0.85);
    if (mat == MAT_C3PO)  return vec3(0.9,0.75,0.2);
    return vec3(1.0);
}

//// sdf2: p - query point
SDFHit sdf2(vec3 p)
{
    //// your implementation starts
    SDFHit hit = sand(p);
    hit = SDFHitUnion(hit, igloo(p, vec3(-2.0, 0.35, 1.0)));
    hit = SDFHitUnion(hit, tower(p, vec3(2.0,0,2.5)));
    hit = SDFHitUnion(hit, tower(p, vec3(1.7,0,0)));
    hit = SDFHitUnion(hit, r2d2(p, vec3(-0.2, 0.0, 0.5)));
    hit = SDFHitUnion(hit, c3po(p, vec3(0.5, 0.0, 0.5)));
    hit = SDFHitUnion(hit, deathStar(p, vec3(2.8, 6.5, 20.0)));
    //// your implementation ends

    return hit;
}

SDFHit rayMarching2(vec3 origin, vec3 dir)
{
    float s = 0.0;
    int mat = 0;
    for(int i = 0; i < 100; i++)
    {
        vec3 originCurrent = origin + (s * dir);
        SDFHit hitCurrent = sdf2(originCurrent);
        if (hitCurrent.d < 0.001) {
            mat = hitCurrent.mat;
            break;
        }
        s += hitCurrent.d;
    }
    
    return SDFHit(s, mat);
}

vec3 applyFog(vec3 color, float dist) {
    vec3 fogColor = vec3(0.95, 0.85, 0.65);
    float fog = 1.0 - exp(-dist * 0.08);
    return mix(color, fogColor, fog);
}

vec3 phong_shading2(vec3 p, vec3 n)
{
    if (p.z > 30.0){
        return vec3(0.1, 0.1, 0.1);
    }

    vec3 lightPos = vec3(4.*sin(iTime), 4., 4.*cos(iTime));  
    vec3 l = normalize(lightPos - p);               
    float amb = 0.1;
    float dif = max(dot(n, l), 0.) * 0.7;
    vec3 eye = CAM_POS;
    float spec = pow(max(dot(reflect(-l, n), normalize(eye - p)), 0.0), 128.0) * 0.9;

    vec3 sunDir = vec3(0, 1, -1);
    float sunDif = max(dot(n, sunDir), 0.) * 0.2;

    //// shadow
    SDFHit shadowHit = rayMarching2(p + n * 0.02, l);
    if(shadowHit.d < length(lightPos - p)) dif *= .2;

    SDFHit hit = sdf2(p);

    vec3 color = materialColor(hit.mat);

    return applyFog((amb + dif + spec + sunDif) * color, hit.d);
}

vec3 normal2(vec3 p)
{
    float dx = 0.01;           //// step size for finite difference
    float gx = sdf2(p + vec3(dx, 0.0, 0.0)).d - sdf2(p - vec3(dx, 0.0, 0.0)).d;
    float gy = sdf2(p + vec3(0.0, dx, 0.0)).d - sdf2(p - vec3(0.0, dx, 0.0)).d;
    float gz = sdf2(p + vec3(0.0, 0.0, dx)).d - sdf2(p - vec3(0.0, 0.0, dx)).d;
    vec3 g = vec3(gx, gy, gz) / 2.0*dx;
    return normalize(g);
}

vec3 getCameraPos(float time)
{
    float angle = time * 0.3;  // Rotation speed
    float radius = 5.0;        // Distance from scene center
    float height = 2.0;        // Height above ground
    vec3 center = vec3(-2.0, 0.35, -15.0);  // Scene center to orbit around
    
    return center + vec3(
        radius * cos(angle),
        height,
        radius * sin(angle)
    );
}

vec3 getCameraPosCycled(float time)
{
    float duration = 15.0;
    float cycle = mod(time, duration);

    if (cycle < 5.0) return CAM_POS;

    if (cycle < duration)
    {
        float t = cycle - 5.0;

        // smooth fade-in over first second
        float fadeIn = smoothstep(0.0, 1.0, t);

        // smooth fade-out during last second
        float fadeOut = smoothstep(10.0, 9.0, t);

        float blend = fadeIn * fadeOut;
        vec3 dynamicPos = getCameraPos(t);
        return mix(CAM_POS, dynamicPos, blend);
    }

    return CAM_POS;
}

void mainImage2(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = (fragCoord.xy - .5 * iResolution.xy) / iResolution.y;         //// screen uv
    // vec3 origin = CAM_POS;                                                  //// camera position 
    vec3 origin = getCameraPosCycled(iTime);                                // Use dynamic position
    vec3 dir = normalize(vec3(uv.x, uv.y, 1));                              //// camera direction
    SDFHit hit = rayMarching2(origin, dir);                                 //// ray marching
    vec3 p = origin + (dir * hit.d);                                        //// ray-sdf intersection
    vec3 n = normal2(p);                                                    //// sdf normal
    vec3 color = phong_shading2(p, n);              //// phong shading
    fragColor = vec4(color, 1.);                                            //// fragment color
}

/////////////////////////////////////////////////////
//// main function
/////////////////////////////////////////////////////

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = (fragCoord.xy - .5 * iResolution.xy) / iResolution.y;         //// screen uv
    vec3 origin = CAM_POS;                                                  //// camera position 
    vec3 dir = normalize(vec3(uv.x, uv.y, 1));                              //// camera direction
    float s = rayMarching(origin, dir);                                     //// ray marching
    vec3 p = origin + dir * s;                                              //// ray-sdf intersection
    vec3 n = normal(p);                                                     //// sdf normal
    vec3 color = phong_shading(p, n);                                       //// phong shading
    fragColor = vec4(color, 1.);                                            //// fragment color
}

void main() 
{
    mainImage2(gl_FragColor, gl_FragCoord.xy);
}