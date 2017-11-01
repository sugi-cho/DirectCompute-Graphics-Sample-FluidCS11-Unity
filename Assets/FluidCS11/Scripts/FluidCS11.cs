using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using UnityEngine;

public class FluidCS11 : MonoBehaviour
{
    [System.Serializable]
    public struct Particle
    {
        public Vector2 position;
        public Vector2 velocity;
    }
    [System.Serializable]
    public struct ParticleDensity
    {
        public float Density;
    }
    [System.Serializable]
    public struct ParticleForce
    {
        public Vector2 acceleration;
    }

    struct Int2
    {
        public int x;
        public int y;
    }

    [System.Serializable]
    public struct SimulationConstants
    {
        public int iNumParticles;
        public float fTimeStep;
        public float fSmoothlen;
        public float fPressureStiffness;
        public float fRestDensity;
        public float fDensityCoef;
        public float fGradPressureCoef;
        public float fLapViscosityCoef;
        public float fWallStiffness;

        public Vector2 vGravity;
        public Vector4 vGridDim;

        public Vector3[] vPlanes;
    }

    public enum NUM_PARTICLES
    {
        NUM_PARTICLES_8K = 8 * 1024,
        NUM_PARTICLES_16K = 16 * 1024,
        NUM_PARTICLES_32K = 32 * 1024,
        NUM_PARTICLES_64K = 64 * 1024,
    }

    public enum SIMULATION_MODE
    {
        SIMPLE,
        SHARED,
        GRID,
    }

    const int NUM_GRID_INDICES = 65536;

    // Numthreads size for the simulation
    const int SIMULATION_BLOCK_SIZE = 256;

    // Numthreads size for the sort
    const int BITONIC_BLOCK_SIZE = 512;
    const int TRANSPOSE_BLOCK_SIZE = 16;

    [Header("shaders")]
    public ComputeShader fluidCS;
    public ComputeShader sortCS;
    public Material fluidRenderer;

    public NUM_PARTICLES numParticles;
    public SIMULATION_MODE simMode;

    int g_iNumParticles;
    [Header("Particle Properties")]
    public float g_fInitialParticleSpacing = 0.0045f;
    public float g_fSmoothlen = 0.012f;
    public float g_fPressureStiffness = 200.0f;
    public float g_fRestDensity = 1000.0f;
    public float g_fParticleMass = 0.0002f;
    public float g_fViscosity = 0.1f;
    public float g_fMaxAllowableTimeStep = 0.005f;
    public float g_fParticleRenderSize = 0.003f;
    public Vector2 g_vGravity;

    SimulationConstants pData;

    float g_fMapHeight = 1.2f;
    float g_fMapWidth = (4f / 3f) * 1.2f;
    public float g_fWallStiffness = 3000.0f;
    Vector3[] g_vPlanes;

    ComputeBuffer particlesBufferRW;
    ComputeBuffer particlesBufferRO;

    ComputeBuffer sortedParticleBuffer;
    ComputeBuffer densityBuffer;
    ComputeBuffer forceBuffer;
    ComputeBuffer gridBuffer;
    ComputeBuffer gridPingPongBuffer;
    ComputeBuffer gridIndicesBuffer;

    ComputeBuffer CreateBuffer<T>(int count, ComputeBufferType type = ComputeBufferType.Default)
    {
        return new ComputeBuffer(count, Marshal.SizeOf(typeof(T)), type);
    }

    void CreateBuffers()
    {
        particlesBufferRW = CreateBuffer<Particle>(g_iNumParticles);
        particlesBufferRO = CreateBuffer<Particle>(g_iNumParticles);
        sortedParticleBuffer = CreateBuffer<Particle>(g_iNumParticles);

        densityBuffer = CreateBuffer<ParticleDensity>(g_iNumParticles);
        forceBuffer = CreateBuffer<ParticleForce>(g_iNumParticles);
        gridBuffer = CreateBuffer<uint>(g_iNumParticles);
        gridPingPongBuffer = CreateBuffer<uint>(g_iNumParticles);
        gridIndicesBuffer = CreateBuffer<Int2>(NUM_GRID_INDICES);

        var iStartingWidth = (int)Mathf.Sqrt(g_iNumParticles);
        var particles = new Particle[g_iNumParticles];
        for (var i = 0; i < g_iNumParticles; i++)
        {
            var x = i % iStartingWidth;
            var y = i / iStartingWidth;
            particles[i] = new Particle()
            {
                position = new Vector2(g_fInitialParticleSpacing * x, g_fInitialParticleSpacing * y)
            };
        }
        particlesBufferRO.SetData(particles);
    }
    void ReleaseBuffers()
    {
        new[] {
            particlesBufferRW,particlesBufferRO,sortedParticleBuffer,
            densityBuffer,forceBuffer,gridBuffer,gridPingPongBuffer,gridIndicesBuffer}
        .ToList().ForEach(b => { if (b != null) b.Release(); });
    }

    void GPUSort(ComputeBuffer inBuffer, ComputeBuffer tempBuffer)
    {
        int KERNEL_ID_BITONICSORT = sortCS.FindKernel("BitonicSort");
        int KERNEL_ID_TRANSPOSE = sortCS.FindKernel("MatrixTranspose");

        uint NUM_ELEMENTS = (uint)g_iNumParticles;
        uint MATRIX_WIDTH = BITONIC_BLOCK_SIZE;
        uint MATRIX_HEIGHT = NUM_ELEMENTS / BITONIC_BLOCK_SIZE;

        for (uint level = 2; level <= BITONIC_BLOCK_SIZE; level <<= 1)
        {
            SetGPUSortConstants(sortCS, level, level, MATRIX_HEIGHT, MATRIX_WIDTH);

            // Sort the row data
            sortCS.SetBuffer(KERNEL_ID_BITONICSORT, "Data", inBuffer);
            sortCS.Dispatch(KERNEL_ID_BITONICSORT, (int)(NUM_ELEMENTS / BITONIC_BLOCK_SIZE), 1, 1);
        }

        // Then sort the rows and columns for the levels > than the block size
        // Transpose. Sort the Columns. Transpose. Sort the Rows.
        for (uint level = (BITONIC_BLOCK_SIZE << 1); level <= NUM_ELEMENTS; level <<= 1)
        {
            // Transpose the data from buffer 1 into buffer 2
            SetGPUSortConstants(sortCS, level / BITONIC_BLOCK_SIZE, (level & ~NUM_ELEMENTS) / BITONIC_BLOCK_SIZE, MATRIX_WIDTH, MATRIX_HEIGHT);
            sortCS.SetBuffer(KERNEL_ID_TRANSPOSE, "Input", inBuffer);
            sortCS.SetBuffer(KERNEL_ID_TRANSPOSE, "Data", tempBuffer);
            sortCS.Dispatch(KERNEL_ID_TRANSPOSE, (int)(MATRIX_WIDTH / TRANSPOSE_BLOCK_SIZE), (int)(MATRIX_HEIGHT / TRANSPOSE_BLOCK_SIZE), 1);

            // Sort the transposed column data
            sortCS.SetBuffer(KERNEL_ID_BITONICSORT, "Data", tempBuffer);
            sortCS.Dispatch(KERNEL_ID_BITONICSORT, (int)(NUM_ELEMENTS / BITONIC_BLOCK_SIZE), 1, 1);

            // Transpose the data from buffer 2 back into buffer 1
            SetGPUSortConstants(sortCS, BITONIC_BLOCK_SIZE, level, MATRIX_HEIGHT, MATRIX_WIDTH);
            sortCS.SetBuffer(KERNEL_ID_TRANSPOSE, "Input", tempBuffer);
            sortCS.SetBuffer(KERNEL_ID_TRANSPOSE, "Data", inBuffer);
            sortCS.Dispatch(KERNEL_ID_TRANSPOSE, (int)(MATRIX_HEIGHT / TRANSPOSE_BLOCK_SIZE), (int)(MATRIX_WIDTH / TRANSPOSE_BLOCK_SIZE), 1);

            // Sort the row data
            sortCS.SetBuffer(KERNEL_ID_BITONICSORT, "Data", inBuffer);
            sortCS.Dispatch(KERNEL_ID_BITONICSORT, (int)(NUM_ELEMENTS / BITONIC_BLOCK_SIZE), 1, 1);
        }
    }

    void SetGPUSortConstants(ComputeShader cs, uint level, uint levelMask, uint width, uint height)
    {
        cs.SetInt("_Level", (int)level);
        cs.SetInt("_LevelMask", (int)levelMask);
        cs.SetInt("_Width", (int)width);
        cs.SetInt("_Height", (int)height);
    }

    void SimulateFluid_Simple()
    {
        var kernel = fluidCS.FindKernel("DensityCS_Simple");
        fluidCS.SetBuffer(kernel, "ParticlesRO", particlesBufferRO);
        fluidCS.SetBuffer(kernel, "ParticlesDensityRW", densityBuffer);
        fluidCS.Dispatch(kernel, g_iNumParticles / SIMULATION_BLOCK_SIZE, 1, 1);

        kernel = fluidCS.FindKernel("ForceCS_Simple");
        fluidCS.SetBuffer(kernel, "ParticlesRO", particlesBufferRO);
        fluidCS.SetBuffer(kernel, "ParticlesDensityRO", densityBuffer);
        fluidCS.SetBuffer(kernel, "ParticlesForcesRW", forceBuffer);
        fluidCS.Dispatch(kernel, g_iNumParticles / SIMULATION_BLOCK_SIZE, 1, 1);

        kernel = fluidCS.FindKernel("IntegrateCS");
        fluidCS.SetBuffer(kernel, "ParticlesRO", particlesBufferRO);
        fluidCS.SetBuffer(kernel, "ParticlesRW", particlesBufferRW);
        fluidCS.SetBuffer(kernel, "ParticlesForcesRO", forceBuffer);
        fluidCS.Dispatch(kernel, g_iNumParticles / SIMULATION_BLOCK_SIZE, 1, 1);
    }

    void SimulateFluid_Shared()
    {
        var kernel = fluidCS.FindKernel("DensityCS_Shared");
        fluidCS.SetBuffer(kernel, "ParticlesRO", particlesBufferRO);
        fluidCS.SetBuffer(kernel, "ParticlesDensityRW", densityBuffer);
        fluidCS.Dispatch(kernel, g_iNumParticles / SIMULATION_BLOCK_SIZE, 1, 1);

        kernel = fluidCS.FindKernel("ForceCS_Shared");
        fluidCS.SetBuffer(kernel, "ParticlesRO", particlesBufferRO);
        fluidCS.SetBuffer(kernel, "ParticlesDensityRO", densityBuffer);
        fluidCS.SetBuffer(kernel, "ParticlesForcesRW", forceBuffer);
        fluidCS.Dispatch(kernel, g_iNumParticles / SIMULATION_BLOCK_SIZE, 1, 1);

        kernel = fluidCS.FindKernel("IntegrateCS");
        fluidCS.SetBuffer(kernel, "ParticlesRO", particlesBufferRO);
        fluidCS.SetBuffer(kernel, "ParticlesRW", particlesBufferRW);
        fluidCS.SetBuffer(kernel, "ParticlesForcesRO", forceBuffer);
        fluidCS.Dispatch(kernel, g_iNumParticles / SIMULATION_BLOCK_SIZE, 1, 1);
    }

    void SimulateFluid_Grid()
    {
        var kernel = fluidCS.FindKernel("BuildGridCS");
        fluidCS.SetBuffer(kernel, "ParticlesRO", particlesBufferRO);
        fluidCS.SetBuffer(kernel, "GridRW", gridBuffer);
        fluidCS.Dispatch(kernel, g_iNumParticles / SIMULATION_BLOCK_SIZE, 1, 1);

        GPUSort(gridBuffer, gridPingPongBuffer);

        kernel = fluidCS.FindKernel("ClearGridIndicesCS");
        fluidCS.SetBuffer(kernel, "GridIndicesRW", gridIndicesBuffer);
        fluidCS.Dispatch(kernel, NUM_GRID_INDICES / SIMULATION_BLOCK_SIZE, 1, 1);

        kernel = fluidCS.FindKernel("BuildGridIndicesCS");
        fluidCS.SetBuffer(kernel, "GridRO", gridBuffer);
        fluidCS.SetBuffer(kernel, "GridIndicesRW", gridIndicesBuffer);
        fluidCS.Dispatch(kernel, g_iNumParticles / SIMULATION_BLOCK_SIZE, 1, 1);


        kernel = fluidCS.FindKernel("RearrangeParticlesCS");
        fluidCS.SetBuffer(kernel, "GridRO", gridBuffer);
        fluidCS.SetBuffer(kernel, "ParticlesRO", particlesBufferRO);
        fluidCS.SetBuffer(kernel, "ParticlesRW", sortedParticleBuffer);
        fluidCS.Dispatch(kernel, g_iNumParticles / SIMULATION_BLOCK_SIZE, 1, 1);


        kernel = fluidCS.FindKernel("DensityCS_Grid");
        fluidCS.SetBuffer(kernel, "ParticlesRO", sortedParticleBuffer);
        fluidCS.SetBuffer(kernel, "GridIndicesRO", gridIndicesBuffer);
        fluidCS.SetBuffer(kernel, "ParticlesDensityRW", densityBuffer);
        fluidCS.Dispatch(kernel, g_iNumParticles / SIMULATION_BLOCK_SIZE, 1, 1);

        kernel = fluidCS.FindKernel("ForceCS_Grid");
        fluidCS.SetBuffer(kernel, "ParticlesRO", sortedParticleBuffer);
        fluidCS.SetBuffer(kernel, "ParticlesDensityRO", densityBuffer);
        fluidCS.SetBuffer(kernel, "GridIndicesRO", gridIndicesBuffer);
        fluidCS.SetBuffer(kernel, "ParticlesForcesRW", forceBuffer);
        fluidCS.Dispatch(kernel, g_iNumParticles / SIMULATION_BLOCK_SIZE, 1, 1);

        kernel = fluidCS.FindKernel("IntegrateCS");
        fluidCS.SetBuffer(kernel, "ParticlesRO", sortedParticleBuffer);
        fluidCS.SetBuffer(kernel, "ParticlesRW", particlesBufferRW);
        fluidCS.SetBuffer(kernel, "ParticlesForcesRO", forceBuffer);
        fluidCS.Dispatch(kernel, g_iNumParticles / SIMULATION_BLOCK_SIZE, 1, 1);
    }

    void SimulateFluid()
    {
        var PI = Mathf.PI;
        // Simulation Constants
        pData.iNumParticles = g_iNumParticles;
        // Clamp the time step when the simulation runs slowly to prevent numerical explosion
        pData.fTimeStep = Mathf.Min(g_fMaxAllowableTimeStep, Time.deltaTime);
        pData.fSmoothlen = g_fSmoothlen;
        pData.fPressureStiffness = g_fPressureStiffness;
        pData.fRestDensity = g_fRestDensity;
        pData.fDensityCoef = g_fParticleMass * 315.0f / (64.0f * PI * Mathf.Pow(g_fSmoothlen, 9));
        pData.fGradPressureCoef = g_fParticleMass * -45.0f / (PI * Mathf.Pow(g_fSmoothlen, 6));
        pData.fLapViscosityCoef = g_fParticleMass * g_fViscosity * 45.0f / (PI * Mathf.Pow(g_fSmoothlen, 6));

        pData.vGravity = g_vGravity;

        // Cells are spaced the size of the smoothing length search radius
        // That way we only need to search the 8 adjacent cells + current cell
        pData.vGridDim.x = 1.0f / g_fSmoothlen;
        pData.vGridDim.y = 1.0f / g_fSmoothlen;
        pData.vGridDim.z = 0;
        pData.vGridDim.w = 0;

        // Collision information for the map
        pData.fWallStiffness = g_fWallStiffness;
        if (pData.vPlanes == null)
            pData.vPlanes = new Vector3[4];
        pData.vPlanes[0] = g_vPlanes[0];
        pData.vPlanes[1] = g_vPlanes[1];
        pData.vPlanes[2] = g_vPlanes[2];
        pData.vPlanes[3] = g_vPlanes[3];

        SetFluidConstants();

        switch (simMode)
        {
            case SIMULATION_MODE.SIMPLE:
                SimulateFluid_Simple();
                break;
            case SIMULATION_MODE.SHARED:
                SimulateFluid_Shared();
                break;
            case SIMULATION_MODE.GRID:
                SimulateFluid_Grid();
                break;
        }

        Swap(ref particlesBufferRO, ref particlesBufferRW);
    }

    void SetFluidConstants()
    {
        fluidCS.SetInt("g_iNumParticles", pData.iNumParticles);
        fluidCS.SetFloat("g_fTimeStep", pData.fTimeStep);
        fluidCS.SetFloat("g_fSmoothlen", pData.fSmoothlen);
        fluidCS.SetFloat("g_fPressureStiffness", pData.fPressureStiffness);
        fluidCS.SetFloat("g_fRestDensity", pData.fRestDensity);
        fluidCS.SetFloat("g_fDensityCoef", pData.fDensityCoef);
        fluidCS.SetFloat("g_fGradPressureCoef", pData.fGradPressureCoef);
        fluidCS.SetFloat("g_fLapViscosityCoef", pData.fLapViscosityCoef);
        fluidCS.SetFloat("g_fWallStiffness", pData.fWallStiffness);

        fluidCS.SetVector("g_vGravity", pData.vGravity);
        fluidCS.SetVector("g_vGridDim", pData.vGridDim);

        //バグかな？？値が渡せない。。
        //fluidCS.SetFloats("g_vPlanes", new[] {
        //    pData.vPlanes[0].x,pData.vPlanes[0].y,pData.vPlanes[0].z,
        //    pData.vPlanes[1].x,pData.vPlanes[1].y,pData.vPlanes[1].z,
        //    pData.vPlanes[2].x,pData.vPlanes[2].y,pData.vPlanes[2].z,
        //    pData.vPlanes[3].x,pData.vPlanes[3].y,pData.vPlanes[3].z,
        //});
    }
    void Swap<T>(ref T lhs, ref T rhs)
    {
        T tmp = lhs;
        lhs = rhs;
        rhs = tmp;
    }
    private void OnRenderObject()
    {
        fluidRenderer.SetBuffer("ParticlesRO", particlesBufferRO);
        fluidRenderer.SetBuffer("ParticleDensityRO", densityBuffer);
        fluidRenderer.SetFloat("g_fParticleSize", g_fParticleRenderSize);
        fluidRenderer.SetPass(0);

        Graphics.DrawProcedural(MeshTopology.Points, g_iNumParticles);
    }

    // Use this for initialization
    void Start()
    {
        g_iNumParticles = (int)numParticles;
        CreateBuffers();
        g_vPlanes = new[] {
                new Vector3(1,0,0),
                new Vector3(0,1,0),
                new Vector3(-1,0,g_fMapWidth),
                new Vector3(0,-1,g_fMapHeight),
            };
    }

    private void OnDestroy()
    {
        ReleaseBuffers();
    }

    // Update is called once per frame
    void Update()
    {
        SimulateFluid();
    }

}
