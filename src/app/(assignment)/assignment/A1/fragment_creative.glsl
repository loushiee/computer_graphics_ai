// #version 300 es
// precision highp float;
// in vec2 vUv; // UV (screen) coordinates in [0,1]^2
// out vec4 FragColor; 
/////////////////////////////////////////////////////
//// CS 8803/4803 CGAI: Computer Graphics in AI Era
//// Assignment a: Ray Tracing
/////////////////////////////////////////////////////

// varying vec2 vUv; // UV (screen) coordinates in [0,1]^2


uniform vec2 iResolution;
uniform float iTime;
uniform int iFrame;

uniform sampler2D floorTex;

#define M_PI 3.1415925585
#define Epsilon 1e-6
vec3 sampleDiffuse(int matId, vec3 p);

//============================================================================
// Primitive data types.
//============================================================================

struct Camera 
{
    vec3 origin;
    vec3 lookAt;
    vec3 up;
    vec3 right;
    float aspectRatio;
};

struct Ray 
{
    vec3 ori;
    vec3 dir;
};

struct Plane 
{
    vec3 n;
    vec3 p;
    int matId;
};

struct Sphere 
{
    vec3 ori;
    float r;
    int matId;
};

struct Box 
{
    vec3 ori;
    vec3 halfWidth;
    mat3 rot;
    int matId;
};

struct Light 
{
    vec3 position;
    vec3 Ia;
    vec3 Id;
    vec3 Is;
};

struct Hit 
{
    float t;
    vec3 p;
    vec3 normal;
    int matId;
};

struct Material 
{
    vec3 ka;   // Ambient coefficient.
    vec3 kd;   // Diffuse coefficient.
    vec3 ks;   // Reflected specular coefficient.
    float shininess; // Shininess of the material.

    vec3 kr;   // Reflected coefficient.
};

//============================================================================
// Global scene data.
//============================================================================
Camera camera;
Light lights[2];
Material materials[7];

Sphere spheres[2];
Box boxes[7];
Plane planes[1];
//////////// Random functions ///////////
float g_seed = 0.;

vec3 gamma2(vec3 col) {
    return vec3(sqrt(col.r), sqrt(col.g), sqrt(col.b));
}

float deg2rad(float deg) 
{
    return deg * M_PI / 180.0;
}

uint base_hash(uvec2 p) 
{
    p = 1103515245U * ((p >> 1U) ^ (p.yx));
    uint h32 = 1103515245U * ((p.x) ^ (p.y >> 3U));
    return h32 ^ (h32 >> 16);
}

void initRand(in vec2 frag_coord, in float time) 
{
    g_seed = float(base_hash(floatBitsToUint(frag_coord))) / float(0xffffffffU) + time;
}

vec2 rand2(inout float seed) 
{
    uint n = base_hash(floatBitsToUint(vec2(seed += .1, seed += .1)));
    uvec2 rz = uvec2(n, n * 48271U);
    return vec2(rz.xy & uvec2(0x7fffffffU)) / float(0x7fffffff);
}

/////////////////////////////////////////
const Hit noHit = Hit(
                 /* time or distance */ -1.0, 
                 /* hit position */     vec3(0), 
                 /* hit normal */       vec3(0), 
                 /* hit material id*/   -1);

Hit hitPlane(const Ray r, const Plane pl) 
{
    Hit hit = noHit;
	
    float t = dot(pl.p - r.ori, pl.n) / dot(r.dir, pl.n);

    if (t <= 0.0)
        return noHit;

    vec3 hitP = r.ori + t * r.dir;
    hit = Hit(t, hitP, pl.n, pl.matId);
    
	return hit;
}

// TODO Step 1.1: Implement the sphere intersection
Hit hitSphere(const Ray r, const Sphere s) 
{
    Hit hit = noHit;

    /* your implementation starts */
    vec3 raySrc = r.ori;
    vec3 rayDir = r.dir;
    vec3 sphereCtr = s.ori;
    float radius = s.r;

    // Quadratic coefficients
    float A = dot(rayDir, rayDir);
    float B = 2.0 * (dot(raySrc, rayDir) - dot(sphereCtr, rayDir));
    float C = dot(raySrc, raySrc) + dot(sphereCtr, sphereCtr) - (2.0 * dot(raySrc, sphereCtr)) - (radius * radius);
    float D = (B * B) - (4.0 * A * C);
    if (D < 0.0) {
        return hit;
    }

    // Compute t values
    float sqrtD = sqrt(D);
    float t1 = (-B - sqrtD) / (2.0 * A);
    float t2 = (-B + sqrtD) / (2.0 * A);
    float EpsilonLocal = Epsilon*100.0;
    
    // Check for valid intersection and compute hit information
    if ((t1 > EpsilonLocal) || (t2 > EpsilonLocal)) {
        float tFinal = (t1 > EpsilonLocal) ? t1 : t2;
        vec3 hitSphere = raySrc + (tFinal * rayDir);
        vec3 normal = normalize(hitSphere - sphereCtr);
        hit = Hit(tFinal, hitSphere, normal, s.matId);
    }

	/* your implementation ends */
    
	return hit;
}

// TODO Step 1.2: Implement the box intersection
Hit hitBox(const Ray r, const Box b) 
{
    Hit hit = noHit;
	
    /* your implementation starts */
    vec3 raySrc = r.ori;
    vec3 rayDir = r.dir;
    vec3 boxCtr = b.ori;
    vec3 boxHalfWidth = b.halfWidth;
    mat3 boxRot = b.rot;

    // Transform ray to box local space
    mat3 rotInverse = transpose(boxRot);
    vec3 boxLocal = rotInverse * (raySrc - boxCtr);
    vec3 rayDirLocal = rotInverse * rayDir;

    // Ray-box intersection in local space (compute t)
    vec3 t1 = ( -boxHalfWidth - boxLocal) / rayDirLocal;
    vec3 t2 = ( boxHalfWidth - boxLocal) / rayDirLocal;
    vec3 tmin = min(t1, t2);
    vec3 tmax = max(t1, t2);
    float tEnter = max(max(tmin.x, tmin.y), tmin.z);
    float tExit = min(min(tmax.x, tmax.y), tmax.z);

    // Check for valid intersection and compute hit information in global space
    float EpsilonLocal = Epsilon*100.0;
    if ((tEnter <= tExit) && (tEnter > EpsilonLocal || tExit > EpsilonLocal)) {
        float tFinal = (tEnter > EpsilonLocal) ? tEnter : tExit;
        vec3 hitLocal = boxLocal + (tFinal * rayDirLocal);
        vec3 hitLocalMinus = abs(hitLocal - boxHalfWidth);
        vec3 hitLocalPlus = abs(hitLocal + boxHalfWidth);
        vec3 normalLocal = vec3(0.0);
        if (hitLocalMinus.x < EpsilonLocal) {
            normalLocal.x = 1.0;
        } else if (hitLocalPlus.x < EpsilonLocal) {
            normalLocal.x = -1.0;
        } else if (hitLocalMinus.y < EpsilonLocal) {
            normalLocal.y = 1.0;
        } else if (hitLocalPlus.y < EpsilonLocal) {
            normalLocal.y = -1.0;
        } else if (hitLocalMinus.z < EpsilonLocal) {
            normalLocal.z = 1.0;
        } else {
            normalLocal.z = -1.0;
        }
        vec3 normalWorld = normalize(boxRot * normalLocal);
        vec3 hitWorld = boxCtr + (boxRot * hitLocal);
        hit = Hit(tFinal, hitWorld, normalWorld, b.matId);
    }
    
	/* your implementation ends */
    
	return hit;
}

Hit findHit(Ray r) 
{
    Hit h = noHit;
    
	for(int i = 0; i < spheres.length(); i++) {
        Hit tempH = hitSphere(r, spheres[i]);
        if(tempH.t > Epsilon && (h.t < 0. || h.t > tempH.t))
            h = tempH;
    }
	
    for(int i = 0; i < planes.length(); i++) {
        Hit tempH = hitPlane(r, planes[i]);
        if(tempH.t > Epsilon && (h.t < 0. || h.t > tempH.t))
            h = tempH;
    }

    for(int i = 0; i < boxes.length(); i++) {
        Hit tempH = hitBox(r, boxes[i]);
        if(tempH.t > Epsilon && (h.t < 0. || h.t > tempH.t))
            h = tempH;
    }

    return h;
}

// TODO Step 2: Implement the Phong shading model
vec3 shading_phong(Light light, int matId, vec3 e, vec3 p, vec3 s, vec3 n) 
{
	//// default color: return dark red for the ground and dark blue for spheres
    vec3 color = matId == 0 ? vec3(0.2, 0, 0) : vec3(0, 0, 0.3);
	
    /* your implementation starts */
    Material material = materials[matId];
    vec3 ambient = material.ka * light.Ia;
    vec3 colorDiffuse = sampleDiffuse(matId, p);
    vec3 lightDir = normalize(s - p);
    vec3 diffuse = colorDiffuse * light.Id * max(dot(lightDir, n), 0.0);
    vec3 reflectDir = reflect(-lightDir, n);
    vec3 viewDir = normalize(e - p);
    vec3 specular = material.ks * light.Is * pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    color = ambient + diffuse + specular;
	/* your implementation ends */
    
	return color;
}

// TODO Step 3: Implement the shadow test 
bool isShadowed(Light light, Hit h) 
{
    bool shadowed = false;
	
    /* your implementation starts */
    Ray shadowRay = Ray(h.p + Epsilon, normalize(light.position - h.p));
    Hit hit = findHit(shadowRay);
    float tmax = length(light.position - h.p);
    if (hit.t > Epsilon && hit.t < tmax) {
        shadowed = true;
    }
	/* your implementation ends */
    
	return shadowed;
}

// TODO Step 4: Implement the texture mapping
vec3 sampleDiffuse(int matId, vec3 p) 
{
    if(matId == 0) {
        vec3 color = materials[matId].kd;
		
        /* your implementation starts */
        color = color * texture(floorTex, p.xz * 0.2).rgb;
		/* your implementation ends */
        
		return color;
    }
    return materials[matId].kd;
}

vec3 rayTrace(in Ray r, out Hit hit) 
{
    vec3 col = vec3(0);
    Hit h = findHit(r);
    hit = h;
    if(h.t > 0. && h.t < 1e8) {
        // shading
        for(int i = 0; i < lights.length(); i++) {
            if(isShadowed(lights[i], h)) {
                col += materials[h.matId].ka * lights[i].Ia;
            } else {
                vec3 e = camera.origin;
                vec3 p = h.p;
                vec3 s = lights[i].position;
                vec3 n = h.normal;
                col += shading_phong(lights[i], h.matId, e, p, s, n);
            }
        }
    }
    return col;
}

Ray getPrimaryRay(vec2 uv) 
{
    return Ray(camera.origin, 
               normalize(camera.lookAt + 
                        (uv.x - 0.5) * camera.right * camera.aspectRatio + 
                        (uv.y - 0.5) * camera.up));
}

mat3 getRotXYZ(float pitch, float yaw, float roll) 
{
    
    mat3 rotX = mat3(
        vec3(1, 0, 0),
        vec3(0, cos(pitch), sin(pitch)),
        vec3(0, -sin(pitch), cos(pitch))
    );
    mat3 rotY = mat3(
        vec3(cos(yaw), 0, -sin(yaw)),
        vec3(0, 1, 0),
        vec3(sin(yaw), 0, cos(yaw))
    );
    mat3 rotZ = mat3(
        vec3(cos(roll), sin(roll), 0),
        vec3(-sin(roll), cos(roll), 0),
        vec3(0, 0, 1)
    );
    
    return rotZ * rotY * rotX;
}


const int MAT_FLOOR    = 0;
const int MAT_MONITOR  = 1;
const int MAT_FRAME    = 2;
const int MAT_KEYBOARD = 3;
const int MAT_MOUSE    = 4;
const int MAT_SPEAKER  = 5;

void initSpheres()
{
    // Left speaker
    spheres[0] = Sphere(
        vec3(-1.5, 0.2, 0.15),  // origin
        0.15,                 // radius
        MAT_SPEAKER
    );

    // Right speaker
    spheres[1] = Sphere(
        vec3(1.5, 0.2, 0.15),
        0.15,
        MAT_SPEAKER
    );
}

void initBoxes()
{
    mat3 rot = mat3(1.0);

    // Monitor screen
    boxes[0] = Box(
        vec3(0.0, 0.7, 0.05),
        vec3(0.9, 0.35, 0.000001),
        rot,
        MAT_MONITOR
    );

    // Monitor frame and stand
    boxes[1] = Box(
        vec3(0.0, 0.7, 0.0),
        vec3(0.9, 0.35, 0.05),
        rot,
        MAT_FRAME
    );
    boxes[2] = Box(
        vec3(0.0, 0.35, -0.05),
        vec3(0.25, 0.15, 0.1),
        rot,
        MAT_FRAME
    );

    // Keyboard
    boxes[3] = Box(
        vec3(0.0, 0.1, 0.7),
        vec3(0.7, 0.05, 0.2),
        rot,
        MAT_KEYBOARD
    );

    // Mouse
    boxes[4] = Box(
        vec3(1.0, 0.08, 0.7),
        vec3(0.1, 0.04, 0.15),
        rot,
        MAT_MOUSE
    );

    // Speakers
    boxes[5] = Box(
        vec3(-1.5, 0.2, 0.0),     // left speaker 
        vec3(0.2, 0.2, 0.2),
        rot,
        MAT_SPEAKER
    );
    boxes[6] = Box(
        vec3(1.5, 0.2, 0.0),      // right speaker 
        vec3(0.2, 0.2, 0.2),
        rot,
        MAT_SPEAKER
    );
}
vec3 materialColor(int material)
{
    if (material == MAT_MONITOR)
        return vec3(0.001); // dark gray

    if (material == MAT_FRAME)
        return vec3(0.25, 0.31, 0.39); // darker blue

    if (material == MAT_KEYBOARD)
        return vec3(0.78, 0.50, 0.50); // reddish

    if (material == MAT_MOUSE)
        return vec3(0.80, 0.72, 0.45); // yellowish
        
    if (material == MAT_SPEAKER)
        return vec3(0.95, 0.55, 0.25); // orangish

    return vec3(0.0);
}

void initScene() 
{
    float aspectRatio = iResolution.x / iResolution.y;
    vec3 origin = vec3(3., 2., 3.);
    vec3 lookAt = normalize(vec3(0.,0.,0.) - origin);
    vec3 up = vec3(0, 1, 0);
    vec3 right = normalize(cross(lookAt, up));
    up = normalize(cross(right, lookAt));
    camera = Camera(origin, lookAt, up, right, aspectRatio);

    // Floor Material 
    vec3 ambientColor = vec3(0.15);
    int i = MAT_FLOOR;
    materials[i].ka = ambientColor;
    materials[i].kd = vec3(0.5);
    materials[i].ks = vec3(0.8);
    materials[i].shininess = 10.0;
    materials[i].kr = 0.3 * materials[i].ks;

    // Monitor Material
    i = MAT_MONITOR;
    materials[i].ka = ambientColor;
    materials[i].kd = materialColor(i);
    materials[i].ks = vec3(0.1);
    materials[i].shininess = 256.;
    materials[i].kr = 0.8 * materials[i].ks;

    i = MAT_FRAME;
    materials[i].ka = ambientColor;
    materials[i].kd = materialColor(i);
    materials[i].ks = vec3(0.1);
    materials[i].shininess = 32.;
    materials[i].kr = 0.8 * materials[i].ks;

    i = MAT_KEYBOARD;
    materials[i].ka = ambientColor;
    materials[i].kd = materialColor(i);
    materials[i].ks = vec3(0.1);
    materials[i].shininess = 32.;
    materials[i].kr = 0.5 * materials[i].ks;

    i = MAT_MOUSE;
    materials[i].ka = ambientColor;
    materials[i].kd = materialColor(i);
    materials[i].ks = vec3(0.1);
    materials[i].shininess = 32.;
    materials[i].kr = 0.5 * materials[i].ks;

    i = MAT_SPEAKER;
    materials[i].ka = ambientColor;
    materials[i].kd = materialColor(i);
    materials[i].ks = vec3(0.3);
    materials[i].shininess = 64.;
    materials[i].kr = 0.4 * materials[i].ks;

    lights[0] = Light(vec3(-4., 5., 2.5), 
                            /*Ia*/ vec3(0.1, 0.1, 0.1), 
                            /*Id*/ vec3(1.0, 1.0, 1.0), 
                            /*Is*/ vec3(0.8, 0.8, 0.8));
    lights[1] = Light(vec3(1.5, 4., 3.), 
                            /*Ia*/ vec3(0.1, 0.1, 0.1), 
                            /*Id*/ vec3(0.9, 0.9, 0.9), 
                            /*Is*/ vec3(0.5, 0.5, 0.5));
    planes[0] = Plane(vec3(0., 1., 0.), vec3(0., 0., 0.), 0);

    initSpheres();
    initBoxes();

    // spheres[0] = Sphere(vec3(0., 1.5, 0.), 0.5, 1);
    // spheres[1] = Sphere(vec3(-0.6, 0.4, 1.1), 0.4, 2);

    // boxes[0] = Box(vec3(0., 0.5, 0.), vec3(0.5), getRotXYZ(0., 0., 0.), 3);
    // boxes[1] = Box(vec3(-1.2, 0.85, 0.0), vec3(0.4, 0.85, 0.4), getRotXYZ(0., 0.4 * M_PI, 0.), 4);
    // boxes[2] = Box(vec3(0.8, 0.3, 0.8), vec3(0.75, 0.3, 0.3), getRotXYZ(0., 0.2 * M_PI, 0.), 5);
}

// TODO Step 5: Change the value of numberOfSampling to 50

/* your implementation starts */

const int numberOfSampling = 50;

/* your implementation ends */

void main() 
{
    initScene();
    initRand(gl_FragCoord.xy, iTime); 
    vec2 uv = gl_FragCoord.xy / iResolution.xy;

    vec3 resultCol = vec3(0.);
    vec3 compounded_kr = vec3(1.0); // cumulative reflection coefficient

    Ray recursive_ray = getPrimaryRay(uv + rand2(g_seed) / iResolution.xy);
    for(int i = 0; i < numberOfSampling; i++) {
        Hit hit;
        vec3 col = rayTrace(recursive_ray, hit);

        resultCol += compounded_kr * col;

        if(hit.t < 0.0 || hit.t > 1e8 || length(compounded_kr) < 0.001)
            break;

        compounded_kr *= materials[hit.matId].kr;
		
        // TODO Step 5: Define the reflected ray and assign this ray to recursive_ray
        
		/* your implementation starts */
        vec3 reflected_dir = reflect(recursive_ray.dir, hit.normal);
        recursive_ray = Ray(hit.p + Epsilon, reflected_dir);
		/* your implementation ends */
    }

    resultCol = gamma2(resultCol);
    // FragColor = vec4(resultCol, 1.);
    gl_FragColor = vec4(resultCol, 1.);
}