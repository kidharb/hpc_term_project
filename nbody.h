// Cuda
#define MAX_EPSILON_ERROR 5e-3f
#define TILE_SIZE 64 // 64 * 64 * 4 = 16384 < 49152bytes
#define MAX_SHARED_MEMORY_BYTES 16384

// Physics
#define G 6.67428e-11
#define AU  (149.6e6 * 1000)     // 149.6 million km, in meters.
#define SCALE  (250 / AU)
#define NUM_PLANETS 9
#define NUM_ROCKETS 30000
#define TIMESTEP 24*3600
#define NUM_TYPES 6
#define NUM_STEPS 1000000
#define BLOCK_SIZE 256

typedef struct {
    double mass;
    double px;
    double py;
    double vx;
    double vy;
}Body;

typedef struct {
    double fx;
    double fy;
}Force;
