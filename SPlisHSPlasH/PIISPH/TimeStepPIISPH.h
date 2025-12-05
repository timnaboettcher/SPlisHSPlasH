#ifndef __TimeStepPIISPH_h__
#define __TimeStepPIISPH_h__

#include "SPlisHSPlasH/Common.h"
#include "SPlisHSPlasH/TimeStep.h"
#include "SimulationDataPIISPH.h"
#include "SPlisHSPlasH/SPHKernels.h"
#include "SPlisHSPlasH/Utilities/MatrixFreeSolver.h"

#define USE_WARMSTART

namespace SPH
{
	class TimeStepPIISPH;
	class Viscosity_PIISPH;
	class Elasticity_PIISPH;
	class Coupling_PIISPH;

	/** \brief Class to store particle coloring information */
	class PIISPHMaterialParameterObject : public GenParam::ParameterObject
	{
	protected:
		TimeStepPIISPH* m_timeStep;
	public:

		bool m_isFluid;
		bool m_isCompressible;

		// fluid parameters
		Real m_viscosity;
		Real m_viscosityBoundary;

		// (porous) solid parameters
		Real m_porosity;
		Real m_youngsModulus;
		Real m_poissonRatio;
		Real m_bloating;
		Real m_softeningVolume;
		Real m_softeningShear;
		Vector3r m_fixedBoxMin;
		Vector3r m_fixedBoxMax;

		PIISPHMaterialParameterObject(TimeStepPIISPH* timeStep)
		{
			m_timeStep = timeStep;

			// Default values
			m_isFluid = true;
			m_isCompressible = false;
			m_viscosity = 0.0;
			m_viscosityBoundary = 0.0;
			m_porosity = 0.0;
			m_youngsModulus = 0.0;
			m_poissonRatio = 0.0;
			m_bloating = 0.0;
			m_softeningVolume = 0.0;
			m_softeningShear = 0.0;
			m_fixedBoxMin.setZero();
			m_fixedBoxMax.setZero();
		}

		static int IS_FLUID;
		static int IS_COMPRESSIBLE;
		static int VISCOSITY_COEFFICIENT;
		static int VISCOSITY_COEFFICIENT_BOUNDARY;
		static int POROSITY;
		static int YOUNGS_MODULUS;
		static int POISSON_RATIO;
		static int BLOATING;
		static int SOFTENING_VOLUME;
		static int SOFTENING_SHEAR;
		static int FIXED_BOX_MIN;
		static int FIXED_BOX_MAX;

		virtual void initParameters();
	};


	class SimulationDataPIISPH;

	/** \brief This class implements the porous flow solver by Böttcher et al. [BWJB25],
	* based on the Implicit Incompressible SPH approach introduced by Ihmsen et al. [ICS+14]
	* with modifications to the constant density solver as proposed by Bender and Koschier 
	* [BK15,BK17,KBST19] (without the divergence-free solver).
	*
	* References:
	* - [BWJB25] Timna Böttcher, Lukas Westhofen, Stefan Rhys Jeske, and Jan Bender. 2025. Implicit Incompressible Porous Flow using SPH. ACM Trans. Graph. 44, 6, Article 267 (December 2025), 13 pages. https://doi.org/10.1145/3763325
	* - [ICS+14] Markus Ihmsen, Jens Cornelis, Barbara Solenthaler, Christopher Horvath, and Matthias Teschner. Implicit incompressible SPH. IEEE Transactions on Visualization and Computer Graphics, 20(3):426-435, March 2014. URL: http://dx.doi.org/10.1109/TVCG.2013.105
	* - [BK15] Jan Bender and Dan Koschier. Divergence-free smoothed particle hydrodynamics. In ACM SIGGRAPH / Eurographics Symposium on Computer Animation, SCA '15, 147-155. New York, NY, USA, 2015. ACM. URL: http://doi.acm.org/10.1145/2786784.2786796
	* - [BK17] Jan Bender and Dan Koschier. Divergence-free SPH for incompressible and viscous fluids. IEEE Transactions on Visualization and Computer Graphics, 23(3):1193-1206, 2017. URL: http://dx.doi.org/10.1109/TVCG.2016.2578335
	* - [KBST19] Dan Koschier, Jan Bender, Barbara Solenthaler, and Matthias Teschner. Smoothed particle hydrodynamics for physically-based simulation of fluids and solids. In Eurographics 2019 - Tutorials. Eurographics Association, 2019. URL: https://interactivecomputergraphics.github.io/SPH-Tutorial
	*/
	class TimeStepPIISPH : public TimeStep
	{
	protected:
		SimulationDataPIISPH m_simulationData;
		const Real m_eps = static_cast<Real>(1.0e-5);
		unsigned int m_iterations;
		Real m_maxError;
		unsigned int m_minIterations;
		unsigned int m_maxIterations;
		unsigned int m_iterationsCG;
		Real m_maxErrorCG;
		unsigned int m_maxIterationsCG;
		std::vector<unsigned int> m_baseIndex;
		std::vector<PIISPHMaterialParameterObject> m_materials;

		Viscosity_PIISPH* m_viscositySolver;
		Elasticity_PIISPH* m_elasticitySolver;
		Coupling_PIISPH* m_couplingSolver;

		// porous flow coupling parameters
		Real m_drag;
		Real m_adhesion;
		Real m_adhesionFalloff;

		void solveStronglyCoupledNonPressure();

		void computeDFSPHFactor(const unsigned int fluidModelIndex);
		void pressureSolve();
		void pressureSolveIteration(const unsigned int fluidModelIndex, Real &avg_density_err);
		void computeDensityAdv(const unsigned int fluidModelIndex, const unsigned int index, const Real h, const Real density0);
		void computeDensityChange(const unsigned int fluidModelIndex, const unsigned int index, const Real h);

		void computePressureAccel(const unsigned int fluidModelIndex, const unsigned int i, const Real density0, std::vector<std::vector<Real>>& pressure_rho2, const bool applyBoundaryForces = false);
		Real compute_aij_pj(const unsigned int fluidModelIndex, const unsigned int i);

		virtual void performNeighborhoodSearchSort();
		virtual void emittedParticles(FluidModel *model, const unsigned int startIndex);

		void computeSaturationAndPorosity(const unsigned int fluidModelIndex);

		typedef Eigen::ConjugateGradient<MatrixReplacement, Eigen::Lower | Eigen::Upper, Eigen::IdentityPreconditioner> Solver;
		Solver m_solver;
		static void matrixVecProd(const Real* vec, Real* result, void* userData);
		void computeRHS(VectorXr& b, VectorXr& g);
		void applyForces(const VectorXr& x);

		/** Init all generic parameters */
		virtual void initParameters();
		void computeMatrixL(const unsigned int fluidModelIndex);
		void computeRotations(const unsigned int fluidModelIndex);
		void initValues();

	public:
		static std::string METHOD_NAME;
		static int SOLVER_ITERATIONS;
		static int MIN_ITERATIONS;
		static int MAX_ITERATIONS;
		static int MAX_ERROR;
		static int SOLVER_ITERATIONS_CG;
		static int MAX_ITERATIONS_CG;
		static int MAX_ERROR_CG;
		static int DRAG;
		static int ADHESION;
		static int ADHESION_FALLOFF;

		TimeStepPIISPH();
		virtual ~TimeStepPIISPH(void);

		Viscosity_PIISPH* getViscositySolver() { return m_viscositySolver; }
		Elasticity_PIISPH* getElasticitySolver() { return m_elasticitySolver; }
		Coupling_PIISPH* getCouplingSolver() { return m_couplingSolver; }
		const Real getAdhesion() { return m_adhesion; }
		const Real getAdhesionFalloff() { return m_adhesionFalloff; }
		const Real getDrag() { return m_drag; }

		/** perform a simulation step */
		virtual void step();
		virtual void reset();
		virtual void deferredInit();

		virtual void resize();
		virtual std::string getMethodName() { return METHOD_NAME; }
		virtual int getNumIterations() { return m_iterations; }

		void computeDensities(const unsigned int fluidModelIndex);
		const Real getVolumeFactor(PIISPHMaterialParameterObject* model, PIISPHMaterialParameterObject* neighbor, const unsigned int object_id, const unsigned int neighbor_object_id);

		void determineFixedParticles();

		SimulationDataPIISPH& getSimulationData() { return m_simulationData; };
		PIISPHMaterialParameterObject* getMaterialObject(const unsigned int fluidModelIndex) { return &m_materials[fluidModelIndex]; }
		const unsigned int getBaseIndex(const unsigned int fluidModelIndex) const { return m_baseIndex[fluidModelIndex]; };
	};
}

#endif
