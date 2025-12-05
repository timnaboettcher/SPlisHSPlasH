#include "TimeStepPIISPH.h"
#include "SPlisHSPlasH/Utilities/MathFunctions.h"
#include "SPlisHSPlasH/TimeManager.h"
#include "SPlisHSPlasH/SPHKernels.h"
#include "SimulationDataPIISPH.h"
#include <iostream>
#include "Utilities/Timing.h"
#include "Utilities/Counting.h"
#include "SPlisHSPlasH/Simulation.h"
#include "SPlisHSPlasH/BoundaryModel_Akinci2012.h"
#include "SPlisHSPlasH/BoundaryModel_Koschier2017.h"
#include "SPlisHSPlasH/BoundaryModel_Bender2019.h"
#include "Viscosity_PIISPH.h"
#include "Elasticity_PIISPH.h"
#include "Coupling_PIISPH.h"

using namespace SPH;
using namespace std;
using namespace GenParam;

std::string TimeStepPIISPH::METHOD_NAME = "PIISPH";
int TimeStepPIISPH::SOLVER_ITERATIONS = -1;
int TimeStepPIISPH::MIN_ITERATIONS = -1;
int TimeStepPIISPH::MAX_ITERATIONS = -1;
int TimeStepPIISPH::MAX_ERROR = -1;
int TimeStepPIISPH::SOLVER_ITERATIONS_CG = -1;
int TimeStepPIISPH::MAX_ITERATIONS_CG = -1;
int TimeStepPIISPH::MAX_ERROR_CG = -1;
int TimeStepPIISPH::DRAG = -1;
int TimeStepPIISPH::ADHESION = -1;
int TimeStepPIISPH::ADHESION_FALLOFF = -1;
int PIISPHMaterialParameterObject::IS_FLUID = -1;
int PIISPHMaterialParameterObject::IS_COMPRESSIBLE = -1;
int PIISPHMaterialParameterObject::VISCOSITY_COEFFICIENT = -1;
int PIISPHMaterialParameterObject::VISCOSITY_COEFFICIENT_BOUNDARY = -1;
int PIISPHMaterialParameterObject::POROSITY = -1;
int PIISPHMaterialParameterObject::YOUNGS_MODULUS = -1;
int PIISPHMaterialParameterObject::POISSON_RATIO = -1;
int PIISPHMaterialParameterObject::BLOATING = -1;
int PIISPHMaterialParameterObject::SOFTENING_VOLUME = -1;
int PIISPHMaterialParameterObject::SOFTENING_SHEAR = -1;
int PIISPHMaterialParameterObject::FIXED_BOX_MAX = -1;
int PIISPHMaterialParameterObject::FIXED_BOX_MIN = -1;

void PIISPHMaterialParameterObject::initParameters()
{
	IS_FLUID = createBoolParameter("isFluid", "Is fluid", &m_isFluid);
	setGroup(IS_FLUID, "Fluid Model|PIISPH - Fluid");
	setDescription(IS_FLUID, "Material is a fluid (and not a porous object).");

	VISCOSITY_COEFFICIENT = createNumericParameter("viscosity", "Kinematic viscosity", &m_viscosity);
	setGroup(VISCOSITY_COEFFICIENT, "Fluid Model|PIISPH - Fluid");
	setDescription(VISCOSITY_COEFFICIENT, "Coefficient for the viscosity force computation.");
	RealParameter* rparam = static_cast<RealParameter*>(getParameter(VISCOSITY_COEFFICIENT));
	rparam->setMinValue(0.0);

	VISCOSITY_COEFFICIENT_BOUNDARY = createNumericParameter("viscosityBoundary", "Viscosity coefficient (Boundary)", &m_viscosityBoundary);
	setGroup(VISCOSITY_COEFFICIENT_BOUNDARY, "Fluid Model|PIISPH - Fluid");
	setDescription(VISCOSITY_COEFFICIENT_BOUNDARY, "Coefficient for the viscosity force computation at the boundary.");
	rparam = static_cast<RealParameter*>(getParameter(VISCOSITY_COEFFICIENT_BOUNDARY));
	rparam->setMinValue(0.0);

	IS_FLUID = createBoolParameter("isFluid", "Is fluid", &m_isFluid);
	setGroup(IS_FLUID, "Fluid Model|PIISPH - Solid");
	setDescription(IS_FLUID, "Material is a fluid (and not a porous object).");

	IS_COMPRESSIBLE = createBoolParameter("isCompressible", "Is compressible", &m_isCompressible);
	setGroup(IS_COMPRESSIBLE, "Fluid Model|PIISPH - Solid");
	setDescription(IS_COMPRESSIBLE, "Material can be compressed until zero porosity.");

	POROSITY = createNumericParameter("porosity", "Porosity", &m_porosity);
	setGroup(POROSITY, "Fluid Model|PIISPH - Solid");
	setDescription(POROSITY, "Porosity of the material (on top of initial 20% volume reduction).");
	rparam = static_cast<RealParameter*>(getParameter(POROSITY));
	rparam->setMinValue(0.0);
	rparam->setMaxValue(1.0);

	YOUNGS_MODULUS = createNumericParameter("youngsModulus", "Young`s modulus", &m_youngsModulus);
	setGroup(YOUNGS_MODULUS, "Fluid Model|PIISPH - Solid");
	setDescription(YOUNGS_MODULUS, "Stiffness of the elastic material");
	rparam = static_cast<RealParameter*>(getParameter(YOUNGS_MODULUS));
	rparam->setMinValue(0.0);

	POISSON_RATIO = createNumericParameter("poissonsRatio", "Poisson`s ratio", &m_poissonRatio);
	setGroup(POISSON_RATIO, "Fluid Model|PIISPH - Solid");
	setDescription(POISSON_RATIO, "Ratio of transversal expansion and axial compression");
	rparam = static_cast<RealParameter*>(getParameter(POISSON_RATIO));
	rparam->setMinValue(static_cast<Real>(-1.0 + 1e-4));
	rparam->setMaxValue(static_cast<Real>(0.5 - 1e-4));

	BLOATING = createNumericParameter("bloating", "Bloating", &m_bloating);
	setGroup(BLOATING, "Fluid Model|PIISPH - Solid");
	setDescription(BLOATING, "Volume increase after fluid absorption.");
	rparam = static_cast<RealParameter*>(getParameter(BLOATING));

	SOFTENING_VOLUME = createNumericParameter("softeningVolume", "Softening (Volume change)", &m_softeningVolume);
	setGroup(SOFTENING_VOLUME, "Fluid Model|PIISPH - Solid");
	setDescription(SOFTENING_VOLUME, "Decrease of volume change resistance after fluid absorption.");
	rparam = static_cast<RealParameter*>(getParameter(SOFTENING_VOLUME));

	SOFTENING_SHEAR = createNumericParameter("softeningShear", "Softening (Shear)", &m_softeningShear);
	setGroup(SOFTENING_SHEAR, "Fluid Model|PIISPH - Solid");
	setDescription(SOFTENING_SHEAR, "Decrease of shear resistance after fluid absorption.");
	rparam = static_cast<RealParameter*>(getParameter(SOFTENING_SHEAR));

	ParameterBase::GetVecFunc<Real> getFct = [&]()-> Real* { return m_fixedBoxMin.data(); };
	ParameterBase::SetVecFunc<Real> setFct = [&](Real* val)
		{
			m_fixedBoxMin = Vector3r(val[0], val[1], val[2]);
			m_timeStep->determineFixedParticles();
		};
	FIXED_BOX_MIN = createVectorParameter("fixedBoxMin", "Fixed box min", 3u, getFct, setFct);
	setGroup(FIXED_BOX_MIN, "Fluid Model|PIISPH - Solid");
	setDescription(FIXED_BOX_MIN, "Minimum point of box of which contains the fixed particles.");
	getParameter(FIXED_BOX_MIN)->setReadOnly(true);

ParameterBase::GetVecFunc<Real> getFct2 = [&]()-> Real* { return m_fixedBoxMax.data(); };
ParameterBase::SetVecFunc<Real> setFct2 = [&](Real* val)
	{
		m_fixedBoxMax = Vector3r(val[0], val[1], val[2]);
		m_timeStep->determineFixedParticles();
	};
FIXED_BOX_MAX = createVectorParameter("fixedBoxMax", "Fixed box max", 3u, getFct2, setFct2);
setGroup(FIXED_BOX_MAX, "Fluid Model|PIISPH - Solid");
setDescription(FIXED_BOX_MAX, "Maximum point of box of which contains the fixed particles.");
getParameter(FIXED_BOX_MAX)->setReadOnly(true);
}

TimeStepPIISPH::TimeStepPIISPH() :
	TimeStep(),
	m_simulationData()
{
	m_simulationData.init();

	m_viscositySolver = new Viscosity_PIISPH(this);
	m_elasticitySolver = new Elasticity_PIISPH(this);
	m_couplingSolver = new Coupling_PIISPH(this);

	m_iterations = 0;
	m_minIterations = 2;
	m_maxIterations = 100;
	m_maxError = static_cast<Real>(0.01);
	m_iterationsCG = 0;
	m_maxIterationsCG = 100;
	m_maxErrorCG = static_cast<Real>(1.0e-4);

	m_drag = static_cast<Real>(0.0);
	m_adhesion = static_cast<Real>(0.0);
	m_adhesionFalloff = static_cast<Real>(0.0);

	// add particle fields - then they can be used for the visualization and export
	Simulation* sim = Simulation::getCurrent();
	const unsigned int nModels = sim->numberOfFluidModels();
	m_materials.resize(nModels, PIISPHMaterialParameterObject(this));

	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nModels; fluidModelIndex++)
	{
		m_materials[fluidModelIndex].initParameters();
		FluidModel* model = sim->getFluidModel(fluidModelIndex);
		model->addField({ "factor", METHOD_NAME, FieldType::Scalar, [this, fluidModelIndex](const unsigned int i) -> Real* { return &m_simulationData.getFactor(fluidModelIndex, i); } });
		model->addField({ "advected density", METHOD_NAME, FieldType::Scalar, [this, fluidModelIndex](const unsigned int i) -> Real* { return &m_simulationData.getDensityAdv(fluidModelIndex, i); } });
		model->addField({ "p / rho^2", METHOD_NAME, FieldType::Scalar, [this, fluidModelIndex](const unsigned int i) -> Real* { return &m_simulationData.getPressureRho2(fluidModelIndex, i); }, true });
		model->addField({ "pressure acceleration", METHOD_NAME, FieldType::Vector3, [this, fluidModelIndex](const unsigned int i) -> Real* { return &m_simulationData.getPressureAccel(fluidModelIndex, i)[0]; } });
		model->addField({ "saturation", METHOD_NAME, FieldType::Scalar, [this, fluidModelIndex](const unsigned int i) -> Real* { return &m_simulationData.getSaturation(fluidModelIndex, i); }, false});
		model->addField({ "porosity", METHOD_NAME, FieldType::Scalar, [this, fluidModelIndex](const unsigned int i) -> Real* { return &m_simulationData.getPorosity(fluidModelIndex, i); }, false});
	}
}

TimeStepPIISPH::~TimeStepPIISPH(void)
{
	// remove all particle fields
	Simulation* sim = Simulation::getCurrent();
	const unsigned int nModels = sim->numberOfFluidModels();
	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nModels; fluidModelIndex++)
	{
		FluidModel* model = sim->getFluidModel(fluidModelIndex);
		model->removeFieldByName("factor");
		model->removeFieldByName("advected density");
		model->removeFieldByName("p / rho^2");
		model->removeFieldByName("pressure acceleration");
		model->removeFieldByName("saturation");
		model->removeFieldByName("porosity");
	}

	delete m_viscositySolver;
	delete m_elasticitySolver;
	delete m_couplingSolver;
}

void TimeStepPIISPH::initParameters()
{
	TimeStep::initParameters();

	SOLVER_ITERATIONS = createNumericParameter("iterations", "Iterations (pressure)", &m_iterations);
	setGroup(SOLVER_ITERATIONS, "Simulation|PIISPH");
	setDescription(SOLVER_ITERATIONS, "Iterations required by the pressure solver.");
	getParameter(SOLVER_ITERATIONS)->setReadOnly(true);

	MIN_ITERATIONS = createNumericParameter("minIterations", "Min. iterations (pressure)", &m_minIterations);
	setGroup(MIN_ITERATIONS, "Simulation|PIISPH");
	setDescription(MIN_ITERATIONS, "Minimal number of iterations of the pressure solver.");
	static_cast<NumericParameter<unsigned int>*>(getParameter(MIN_ITERATIONS))->setMinValue(0);

	MAX_ITERATIONS = createNumericParameter("maxIterations", "Max. iterations (pressure)", &m_maxIterations);
	setGroup(MAX_ITERATIONS, "Simulation|PIISPH");
	setDescription(MAX_ITERATIONS, "Maximal number of iterations of the pressure solver.");
	static_cast<NumericParameter<unsigned int>*>(getParameter(MAX_ITERATIONS))->setMinValue(1);

	MAX_ERROR = createNumericParameter("maxError", "Max. density error(%)", &m_maxError);
	setGroup(MAX_ERROR, "Simulation|PIISPH");
	setDescription(MAX_ERROR, "Maximal density error (%).");
	static_cast<RealParameter*>(getParameter(MAX_ERROR))->setMinValue(static_cast<Real>(1e-6));

	SOLVER_ITERATIONS_CG = createNumericParameter("iterationsCG", "Iterations (CG)", &m_iterationsCG);
	setGroup(SOLVER_ITERATIONS_CG, "Simulation|PIISPH");
	setDescription(SOLVER_ITERATIONS_CG, "Iterations required by the conjugate gradient solver.");
	getParameter(SOLVER_ITERATIONS_CG)->setReadOnly(true);

	MAX_ITERATIONS_CG = createNumericParameter("maxIterationsCG", "Max. iterations (CG)", &m_maxIterationsCG);
	setGroup(MAX_ITERATIONS_CG, "Simulation|PIISPH");
	setDescription(MAX_ITERATIONS_CG, "Maximal number of iterations of the conjugate gradient solver.");
	static_cast<NumericParameter<unsigned int>*>(getParameter(MAX_ITERATIONS_CG))->setMinValue(1);

	MAX_ERROR_CG = createNumericParameter("maxErrorCG", "Max. CG error", &m_maxErrorCG);
	setGroup(MAX_ERROR_CG, "Simulation|PIISPH");
	setDescription(MAX_ERROR_CG, "Maximal conjugate gradient error.");
	static_cast<RealParameter*>(getParameter(MAX_ERROR_CG))->setMinValue(static_cast<Real>(1e-6));

	DRAG = createNumericParameter("drag", "Porous drag coefficient", &m_drag);
	setGroup(DRAG, "Simulation|PIISPH");
	setDescription(DRAG, "Viscous drag between fluid and solid.");
	static_cast<RealParameter*>(getParameter(DRAG))->setMinValue(static_cast<Real>(0.0));

	ADHESION = createNumericParameter("adhesion", "Porous adhesion coefficient", &m_adhesion);
	setGroup(ADHESION, "Simulation|PIISPH");
	setDescription(ADHESION, "Adhesion coefficient between fluid and solid.");
	static_cast<RealParameter*>(getParameter(ADHESION))->setMinValue(static_cast<Real>(0.0));

	ADHESION_FALLOFF = createNumericParameter("adhesionFalloff", "Porous adhesion falloff", &m_adhesionFalloff);
	setGroup(ADHESION_FALLOFF, "Simulation|PIISPH");
	setDescription(ADHESION_FALLOFF, "Adhesion force falloff with increasing fluid saturation.");
	static_cast<RealParameter*>(getParameter(ADHESION_FALLOFF))->setMinValue(static_cast<Real>(0.0));
	static_cast<RealParameter*>(getParameter(ADHESION_FALLOFF))->setMaxValue(static_cast<Real>(1.0));
}

const Real TimeStepPIISPH::getVolumeFactor(PIISPHMaterialParameterObject* model, PIISPHMaterialParameterObject* neighbor, const unsigned int object_id, const unsigned int neighbor_object_id)
{
	// Fluid particle with any neighbor
	if (model->m_isFluid)
		return static_cast<Real>(1.0) - neighbor->m_porosity;

	// Porous solid particle with fluid neighbor
	if (!model->m_isFluid && neighbor->m_isFluid)
		return static_cast<Real>(0.0);

	// Get volume factor for compressible solid particle with other solid particle in same object
	else if (model->m_isCompressible && model == neighbor && object_id == neighbor_object_id)
		return static_cast<Real>(1.0) - model->m_porosity;

	// Solid particle with solid neighbor
	return static_cast<Real>(1.0);
}

void TimeStepPIISPH::computeDensities(const unsigned int fluidModelIndex)
{
	Simulation* sim = Simulation::getCurrent();
	FluidModel* model = sim->getFluidModel(fluidModelIndex);
	const Real density0 = model->getDensity0();
	const unsigned int numParticles = model->numActiveParticles();
	const unsigned int nFluids = sim->numberOfFluidModels();
	const unsigned int nBoundaries = sim->numberOfBoundaryModels();

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < (int)numParticles; i++)
		{
			Real& density = model->getDensity(i);
			PIISPHMaterialParameterObject* material = &m_materials[fluidModelIndex];

			// Compute current density for particle i
			if (material->m_isCompressible)
				density = (static_cast<Real>(1.0) - material->m_porosity) * model->getVolume(i) * sim->W_zero();
			else
				density = model->getVolume(i) * sim->W_zero();

			const Vector3r& xi = model->getPosition(i);
			const unsigned int obj_i = model->getObjectId(i);

			//////////////////////////////////////////////////////////////////////////
			// Fluid
			//////////////////////////////////////////////////////////////////////////
			forall_fluid_neighbors(
				const Real factor = getVolumeFactor(material, &m_materials[pid], obj_i, fm_neighbor->getObjectId(neighborIndex));
				density += factor * fm_neighbor->getVolume(neighborIndex) * sim->W(xi - xj);
			);

			//////////////////////////////////////////////////////////////////////////
			// Boundary
			//////////////////////////////////////////////////////////////////////////
			if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
			{
				forall_boundary_neighbors(
					// Boundary: Akinci2012
					density += bm_neighbor->getVolume(neighborIndex) * sim->W(xi - xj);
				);
			}
			else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Koschier2017)
			{
				forall_density_maps(
					density += rho;
				);
			}
			else   // Bender2019
			{
				forall_volume_maps(
					density += Vj * sim->W(xi - xj);
				);
			}

			density *= density0;
		}
	}
}

void TimeStepPIISPH::step()
{
	Simulation *sim = Simulation::getCurrent();
	TimeManager *tm = TimeManager::getCurrent ();
	const Real h = tm->getTimeStepSize();
	const unsigned int nModels = sim->numberOfFluidModels();

	//////////////////////////////////////////////////////////////////////////
	// search the neighbors for all particles
	//////////////////////////////////////////////////////////////////////////
	sim->performNeighborhoodSearch();

#ifdef USE_PERFORMANCE_OPTIMIZATION
	//////////////////////////////////////////////////////////////////////////
	// precompute the values V_j * grad W_ij for all neighbors
	//////////////////////////////////////////////////////////////////////////
	START_TIMING("precomputeValues")
	precomputeValues();
	STOP_TIMING_AVG
#endif

	//////////////////////////////////////////////////////////////////////////
	// compute volume/density maps boundary contribution
	//////////////////////////////////////////////////////////////////////////
	if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Bender2019)
		computeVolumeAndBoundaryX();
	else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Koschier2017)
		computeDensityAndGradient();

	//////////////////////////////////////////////////////////////////////////
	// compute densities
	//////////////////////////////////////////////////////////////////////////
	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nModels; fluidModelIndex++)
		computeDensities(fluidModelIndex);

	//////////////////////////////////////////////////////////////////////////
	// Compute the factor alpha_i for all particles i
	// using the equation (11) in [BK17]
	//////////////////////////////////////////////////////////////////////////
	START_TIMING("computeDFSPHFactor");
	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nModels; fluidModelIndex++)
		computeDFSPHFactor(fluidModelIndex);
	STOP_TIMING_AVG;

	//////////////////////////////////////////////////////////////////////////
	// Reset accelerations and add gravity
	//////////////////////////////////////////////////////////////////////////
	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nModels; fluidModelIndex++)
		clearAccelerations(fluidModelIndex);

	//////////////////////////////////////////////////////////////////////////
	// Compute all nonpressure forces like viscosity, vorticity, ...
	//////////////////////////////////////////////////////////////////////////

	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nModels; fluidModelIndex++)
		computeSaturationAndPorosity(fluidModelIndex);

	START_TIMING("StronglyCoupledSolve");
	solveStronglyCoupledNonPressure();
	STOP_TIMING_AVG;
	sim->computeNonPressureForces();

	//////////////////////////////////////////////////////////////////////////
	// Update the time step size, e.g. by using a CFL condition
	//////////////////////////////////////////////////////////////////////////
	sim->updateTimeStepSize();

	//////////////////////////////////////////////////////////////////////////
	// compute new velocities only considering non-pressure forces
	//////////////////////////////////////////////////////////////////////////
	for (unsigned int m = 0; m < nModels; m++)
	{
		FluidModel *fm = sim->getFluidModel(m);
		const unsigned int numParticles = fm->numActiveParticles();
		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static)  
			for (int i = 0; i < (int)numParticles; i++)
			{
				if (fm->getParticleState(i) == ParticleState::Active)
				{
					Vector3r &vel = fm->getVelocity(i);
					vel += h * fm->getAcceleration(i);
				}
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// Perform constant density solve (see Algorithm 3 in [BK17])
	//////////////////////////////////////////////////////////////////////////
	START_TIMING("pressureSolve");
	pressureSolve();
	STOP_TIMING_AVG;

	//////////////////////////////////////////////////////////////////////////
	// compute final positions
	//////////////////////////////////////////////////////////////////////////
	for (unsigned int m = 0; m < nModels; m++)
	{
		FluidModel *fm = sim->getFluidModel(m);
		const unsigned int numParticles = fm->numActiveParticles();
		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static)  
			for (int i = 0; i < (int)numParticles; i++)
			{
				if (fm->getParticleState(i) == ParticleState::Active)
				{
					Vector3r &xi = fm->getPosition(i);
					const Vector3r &vi = fm->getVelocity(i);
					xi += h * vi;
				}
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// emit new particles and perform an animation field step
	//////////////////////////////////////////////////////////////////////////
	sim->emitParticles();
	sim->animateParticles();

	//////////////////////////////////////////////////////////////////////////
	// Compute new time
	//////////////////////////////////////////////////////////////////////////
	tm->setTime (tm->getTime () + h);
}

void TimeStepPIISPH::computeSaturationAndPorosity(const unsigned int fluidModelIndex)
{
	START_TIMING("Porosity - saturation")
		Simulation* sim = Simulation::getCurrent();
		FluidModel* model = sim->getFluidModel(fluidModelIndex);
		const unsigned int numParticles = model->numActiveParticles();
		const unsigned int nFluids = sim->numberOfFluidModels();
		const unsigned int nBoundaries = sim->numberOfBoundaryModels();

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < (int)numParticles; i++)
		{
			const Vector3r& xi = model->getPosition(i);
			Real Ns = model->getMass(i) / model->getDensity(i) * sim->W_zero();

			if (!m_materials[fluidModelIndex].m_isFluid) // solid: store saturation
			{
				// for all fluid neighbors
				Real saturation = static_cast<Real>(0.0);
				for (unsigned int pid = 0; pid < nFluids; pid++)
				{
					if (m_materials[pid].m_isFluid)
					{
						FluidModel* fm_neighbor = sim->getFluidModelFromPointSet(pid);
						for (unsigned int j = 0; j < sim->numberOfNeighbors(fluidModelIndex, pid, i); j++)
						{
							const unsigned int neighborIndex = sim->getNeighbor(fluidModelIndex, pid, i, j);
							const Vector3r& xj = fm_neighbor->getPosition(neighborIndex);
							saturation += fm_neighbor->getVolume(neighborIndex) * sim->W(xi - xj);
						}
					}
				}
				forall_fluid_neighbors_in_same_phase(
					Ns += model->getMass(neighborIndex) / model->getDensity(neighborIndex) * sim->W(xi - xj);
				)
				
				if (m_materials[fluidModelIndex].m_porosity > 0.0)
				{
					saturation = saturation / (Ns * m_materials[fluidModelIndex].m_porosity);
					if (saturation > 1.0) saturation = static_cast<Real>(1.0);
				}
				else if (saturation > 0.0)  saturation = static_cast<Real>(1.0);
				m_simulationData.setSaturation(fluidModelIndex, i, saturation);
			}
			else // fluid: store local porosity
			{
				Real porosity = static_cast<Real>(1.0);
				// for all solid neighbors
				for (unsigned int pid = 0; pid < nFluids; pid++)
				{
					if (!m_materials[pid].m_isFluid)
					{
						FluidModel* fm_neighbor = sim->getFluidModelFromPointSet(pid);
						const Real porosity_s = m_materials[pid].m_porosity;
						for (unsigned int j = 0; j < sim->numberOfNeighbors(fluidModelIndex, pid, i); j++)
						{
							const unsigned int neighborIndex = sim->getNeighbor(fluidModelIndex, pid, i, j);
							const Vector3r& xj = fm_neighbor->getPosition(neighborIndex);
							porosity -= (static_cast<Real>(1.0) - porosity_s) * fm_neighbor->getVolume(neighborIndex) * sim->W(xi - xj);
						}
					}
				}

				if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
				{
					forall_boundary_neighbors(
						// Boundary: Akinci2012
						porosity -= bm_neighbor->getVolume(neighborIndex) * sim->W(xi - xj);
					);
				}
				else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Koschier2017)
				{
					forall_density_maps(
						porosity -= rho;
					);
				}
				else   // Bender2019
				{
					forall_volume_maps(
						porosity -= Vj * sim->W(xi - xj);
					);
				}

				m_simulationData.setPorosity(fluidModelIndex, i, porosity);
			}
		}
	}
	STOP_TIMING_AVG;
}
void TimeStepPIISPH::computeRHS(VectorXr& b, VectorXr& g)
{
	Simulation* sim = Simulation::getCurrent();
	const unsigned int nFluids = sim->numberOfFluidModels();
	const unsigned int nBoundaries = sim->numberOfBoundaryModels();
	const Real dt = TimeManager::getCurrent()->getTimeStepSize();

	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
	{
		FluidModel* model = sim->getFluidModel(fluidModelIndex);
		const int numParticles = (int)model->numActiveParticles();

		//////////////////////////////////////////////////////////////////////////
		// Compute RHS
		//////////////////////////////////////////////////////////////////////////
#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static) nowait
			for (int i = 0; i < (int)numParticles; i++)
			{
				const unsigned int idx = 3 * (i + getBaseIndex(fluidModelIndex));
				const Vector3r& vi = model->getVelocity(i);
				const Real mi = model->getMass(i);
				b[idx] = mi * vi[0];
				b[idx + 1] = mi * vi[1];
				b[idx + 2] = mi * vi[2];

				if (model->getParticleState(i) == ParticleState::Active)
				{
					g[idx] = vi[0] + dt * m_simulationData.getVDiff(fluidModelIndex, i)[0];
					g[idx + 1] = vi[1] + dt * m_simulationData.getVDiff(fluidModelIndex, i)[1];
					g[idx + 2] = vi[2] + dt * m_simulationData.getVDiff(fluidModelIndex, i)[2];
				}
				else
				{
					g[idx] = vi[0];
					g[idx + 1] = vi[1];
					g[idx + 2] = vi[2];
				}
			}
		}
	}

	m_viscositySolver->computeRHS(b);
	m_elasticitySolver->computeRHS(b);
	m_couplingSolver->computeRHS(b);
}

void TimeStepPIISPH::matrixVecProd(const Real* vec, Real* result, void* userData)
{
	Simulation* sim = Simulation::getCurrent();
	TimeStepPIISPH* timeStep = (TimeStepPIISPH*)userData;

	const unsigned int nFluids = sim->numberOfFluidModels();
	const Real dt = TimeManager::getCurrent()->getTimeStepSize();

	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
	{
		FluidModel* model = sim->getFluidModel(fluidModelIndex);
		const unsigned int numParticles = model->numActiveParticles();
		timeStep->getViscositySolver()->computeForces(fluidModelIndex, &vec[3 * timeStep->getBaseIndex(fluidModelIndex)]);
		timeStep->getElasticitySolver()->computeForces(fluidModelIndex, &vec[3 * timeStep->getBaseIndex(fluidModelIndex)]);
		timeStep->getCouplingSolver()->computeForces(fluidModelIndex, &vec[0]);

#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static) 
			for (int i = 0; i < (int)numParticles; i++)
			{
				const unsigned int idx = 3 * (i + timeStep->getBaseIndex(fluidModelIndex));
				const Real mi = model->getMass(i);
				if (model->getParticleState(i) == ParticleState::Active)
				{
					const Vector3r& fi_visco = timeStep->m_simulationData.getViscoForce(fluidModelIndex, i);
					const Vector3r& fi_elasticity = timeStep->m_simulationData.getElasticityForce(fluidModelIndex, i);
					const Vector3r& fi_coupling = timeStep->m_simulationData.getCouplingForce(fluidModelIndex, i);

					result[idx] = mi * vec[idx] - dt * (fi_visco[0] + fi_elasticity[0] + fi_coupling[0]);
					result[idx + 1] = mi * vec[idx + 1] - dt * (fi_visco[1] + fi_elasticity[1] + fi_coupling[1]);
					result[idx + 2] = mi * vec[idx + 2] - dt * (fi_visco[2] + fi_elasticity[2] + fi_coupling[2]);
	
				}
				else
				{
					result[idx] = mi * vec[idx];
					result[idx + 1] = mi * vec[idx + 1];
					result[idx + 2] = mi * vec[idx + 2];
				}
			}
		}
	}
}

void TimeStepPIISPH::applyForces(const VectorXr& x)
{
	Simulation* sim = Simulation::getCurrent();
	const Real dt = TimeManager::getCurrent()->getTimeStepSize();
	const unsigned int nFluids = sim->numberOfFluidModels();

	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
	{
		FluidModel* model = sim->getFluidModel(fluidModelIndex);
		const int numParticles = (int)model->numActiveParticles();

#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static)
			for (int i = 0; i < (int)numParticles; i++)
			{
				if (model->getParticleState(i) == ParticleState::Active)
				{
					// Compute the acceleration from the velocity change
					//Vector3r& ai = model->getAcceleration(i);
					const int idx = 3 * (i + getBaseIndex(fluidModelIndex));
					const Vector3r newVi(x[idx], x[idx + 1], x[idx + 2]);
					Vector3r v_diff = newVi - model->getVelocity(i);
					m_simulationData.getVDiff(fluidModelIndex, i) = v_diff;
					Vector3r& ai = model->getAcceleration(i);
					ai += (1.0 / dt) * (v_diff);
				}
			}
		}
	}
}

void TimeStepPIISPH::solveStronglyCoupledNonPressure()
{
	Simulation* sim = Simulation::getCurrent();
	const Real h = TimeManager::getCurrent()->getTimeStepSize();
	const unsigned int nFluids = sim->numberOfFluidModels();

	int numParticlesTotal = 0;
	m_baseIndex.clear();
	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
	{
		m_baseIndex.push_back(numParticlesTotal);
		numParticlesTotal += (int)sim->getFluidModel(fluidModelIndex)->numActiveParticles();

		const Real youngsModulus = m_materials[fluidModelIndex].m_youngsModulus;
		if (youngsModulus != 0.0)
			computeRotations(fluidModelIndex);
	}

	MatrixReplacement A(3 * numParticlesTotal, matrixVecProd, (void*)this);

	m_solver.setTolerance(m_maxErrorCG);
	m_solver.setMaxIterations(m_maxIterationsCG);
	m_solver.compute(A);

	VectorXr b(3 * numParticlesTotal);
	VectorXr x(3 * numParticlesTotal);
	VectorXr g(3 * numParticlesTotal);

	START_TIMING("computeRHS");
	computeRHS(b, g);
	STOP_TIMING_AVG;

 	START_TIMING("CG solve");
	x = m_solver.solveWithGuess(b, g);
	m_iterationsCG = (int)m_solver.iterations();
	STOP_TIMING_AVG;

	INCREASE_COUNTER("Porosity - CG iterations", static_cast<Real>(m_iterationsCG));
	
	applyForces(x);
}

void TimeStepPIISPH::pressureSolve()
{
	const Real h = TimeManager::getCurrent()->getTimeStepSize();
	const Real h2 = h*h;
	const Real invH = static_cast<Real>(1.0) / h;
	const Real invH2 = static_cast<Real>(1.0) / h2;
	Simulation *sim = Simulation::getCurrent();
	const unsigned int nFluids = sim->numberOfFluidModels();

	//////////////////////////////////////////////////////////////////////////
	// Compute rho_adv
	//////////////////////////////////////////////////////////////////////////
	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
	{
		FluidModel *model = sim->getFluidModel(fluidModelIndex);
		
		const Real density0 = model->getDensity0();
		const int numParticles = (int)model->numActiveParticles();
		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static)  
			for (int i = 0; i < numParticles; i++)
			{
				//////////////////////////////////////////////////////////////////////////
				// Compute rho_adv,i^(0) (see equation in Section 3.3 in [BK17])
				// using the velocities after the non-pressure forces were applied.
				//////////////////////////////////////////////////////////////////////////
				computeDensityAdv(fluidModelIndex, i, h, density0);

				//////////////////////////////////////////////////////////////////////////
				// In the end of Section 3.3 [BK17] we have to multiply the density 
				// error with the factor alpha_i divided by h^2. Hence, we multiply 
				// the factor directly by 1/h^2 here.
				//////////////////////////////////////////////////////////////////////////
				m_simulationData.getFactor(fluidModelIndex, i) *= invH2;

				//////////////////////////////////////////////////////////////////////////
				// For the warm start we use 0.5 times the old pressure value.
				// Note: We divide the value by h^2 since we multiplied it by h^2 at the end of 
				// the last time step to make it independent of the time step size.
				//////////////////////////////////////////////////////////////////////////
#ifdef USE_WARMSTART
				if (m_simulationData.getDensityAdv(fluidModelIndex, i) > 1.0)
					m_simulationData.getPressureRho2(fluidModelIndex, i) = static_cast<Real>(0.5) * min(m_simulationData.getPressureRho2(fluidModelIndex, i), static_cast<Real>(0.00025)) * invH2;
				else 
					m_simulationData.getPressureRho2(fluidModelIndex, i) = 0.0;
#else 
				//////////////////////////////////////////////////////////////////////////
				// If we don't use a warm start, we directly compute a pressure value
				// by multiplying the density error with the factor.
				//////////////////////////////////////////////////////////////////////////
				//m_simulationData.getPressureRho2(fluidModelIndex, i) = 0.0;
				const Real s_i = static_cast<Real>(1.0) - m_simulationData.getDensityAdv(fluidModelIndex, i);
				const Real residuum = min(s_i, static_cast<Real>(0.0));     // r = b - A*p
				m_simulationData.getPressureRho2(fluidModelIndex, i) = -residuum * m_simulationData.getFactor(fluidModelIndex, i);
#endif
			}
		}
	}

	m_iterations = 0;

	//////////////////////////////////////////////////////////////////////////
	// Start solver
	//////////////////////////////////////////////////////////////////////////
	
	Real avg_density_err = 0.0;
	bool chk = false;


	//////////////////////////////////////////////////////////////////////////
	// Perform solver iterations
	//////////////////////////////////////////////////////////////////////////
	while ((!chk || (m_iterations < m_minIterations)) && (m_iterations < m_maxIterations))
	{
		chk = true;
		for (unsigned int i = 0; i < nFluids; i++)
		{
			FluidModel *model = sim->getFluidModel(i);
			const Real density0 = model->getDensity0();

			avg_density_err = 0.0;
			pressureSolveIteration(i, avg_density_err);

			// Maximal allowed density fluctuation
			const Real eta = m_maxError * static_cast<Real>(0.01) * density0;  // maxError is given in percent
			chk = chk && (avg_density_err <= eta);
		}

		m_iterations++;
	}

	INCREASE_COUNTER("PIISPH - iterations", static_cast<Real>(m_iterations));

	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
	{
		FluidModel *model = sim->getFluidModel(fluidModelIndex);
		const int numParticles = (int)model->numActiveParticles();
		const Real density0 = model->getDensity0();
		
		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static)  
			for (int i = 0; i < numParticles; i++)
			{
				//////////////////////////////////////////////////////////////////////////
				// Time integration of the pressure accelerations to get new velocities
				//////////////////////////////////////////////////////////////////////////
				computePressureAccel(fluidModelIndex, i, density0, m_simulationData.getPressureRho2Data(), true);
				model->getVelocity(i) += h * m_simulationData.getPressureAccel(fluidModelIndex, i);
			}
		}
	}
#ifdef USE_WARMSTART
	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
	{
		FluidModel* model = sim->getFluidModel(fluidModelIndex);
		const int numParticles = (int)model->numActiveParticles();
		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static)  
			for (int i = 0; i < numParticles; i++)
			{
				//////////////////////////////////////////////////////////////////////////
				// Multiply by h^2, the time step size has to be removed 
				// to make the pressure value independent 
				// of the time step size
				//////////////////////////////////////////////////////////////////////////
				m_simulationData.getPressureRho2(fluidModelIndex, i) *= h2;
			}		
		}
	}
#endif
}

void TimeStepPIISPH::pressureSolveIteration(const unsigned int fluidModelIndex, Real &avg_density_err)
{
	Simulation *sim = Simulation::getCurrent();
	FluidModel *model = sim->getFluidModel(fluidModelIndex);
	const Real density0 = model->getDensity0();
	const int numParticles = (int)model->numActiveParticles();
	if (numParticles == 0)
		return;

	const unsigned int nFluids = sim->numberOfFluidModels();
	const unsigned int nBoundaries = sim->numberOfBoundaryModels();
	const Real h = TimeManager::getCurrent()->getTimeStepSize();
	const Real invH = static_cast<Real>(1.0) / h;
	
	Real density_error = 0.0;

	#pragma omp parallel default(shared)
	{
		//////////////////////////////////////////////////////////////////////////
		// Compute pressure accelerations using the current pressure values.
		// (see Algorithm 3, line 7 in [BK17])
		//////////////////////////////////////////////////////////////////////////
		#pragma omp for schedule(static) 
		for (int i = 0; i < numParticles; i++)
		{
			computePressureAccel(fluidModelIndex, i, density0, m_simulationData.getPressureRho2Data());
		}

		//////////////////////////////////////////////////////////////////////////
		// Update pressure values
		//////////////////////////////////////////////////////////////////////////
		#pragma omp for reduction(+:density_error) schedule(static) 
		for (int i = 0; i < numParticles; i++)
		{						
			if (model->getParticleState(i) != ParticleState::Active)
				continue;
				
			Real aij_pj = compute_aij_pj(fluidModelIndex, i);
			aij_pj *= h * h;

			//////////////////////////////////////////////////////////////////////////
			// Compute source term: s_i = 1 - rho_adv
			// Note: that due to our multiphase handling, the multiplier rho0
			// is missing here
			//////////////////////////////////////////////////////////////////////////
			const Real& densityAdv = m_simulationData.getDensityAdv(fluidModelIndex, i);
			const Real s_i = static_cast<Real>(1.0) - densityAdv;


			//////////////////////////////////////////////////////////////////////////
			// Update the value p/rho^2 (in [BK17] this is kappa/rho):
			// 
			// alpha_i = -1 / (a_ii * rho_i^2)
			// p_rho2_i = (p_i / rho_i^2)
			// 
			// Therefore, the following lines compute the Jacobi iteration:
			// p_i := p_i + 1/a_ii (source_term_i - a_ij * p_j)
			//////////////////////////////////////////////////////////////////////////
			Real& p_rho2_i = m_simulationData.getPressureRho2(fluidModelIndex, i);
			const Real residuum = min(s_i - aij_pj, static_cast<Real>(0.0));     // r = b - A*p
			//p_rho2_i -= residuum * m_simulationData.getFactor(fluidModelIndex, i);

			p_rho2_i = max(p_rho2_i - static_cast<Real>(0.5) * (s_i - aij_pj) * m_simulationData.getFactor(fluidModelIndex, i), static_cast<Real>(0.0));

			//////////////////////////////////////////////////////////////////////////
			// Compute the sum of the density errors
			//////////////////////////////////////////////////////////////////////////
			density_error -= density0 * residuum;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// Compute the average density error
	//////////////////////////////////////////////////////////////////////////
	avg_density_err = density_error / numParticles;
}

void TimeStepPIISPH::reset()
{
	TimeStep::reset();
	m_simulationData.reset();
	m_iterations = 0;
	initValues();
}

void TimeStepPIISPH::performNeighborhoodSearchSort()
{
	m_simulationData.performNeighborhoodSearchSort();
}

void TimeStepPIISPH::emittedParticles(FluidModel *model, const unsigned int startIndex)
{
	m_simulationData.emittedParticles(model, startIndex);
}

void TimeStepPIISPH::resize()
{
	m_simulationData.init();
	initValues();
}

void TimeStepPIISPH::deferredInit()
{
	initValues();
}

void TimeStepPIISPH::computeRotations(const unsigned int fluidModelIndex)
{
	Simulation* sim = Simulation::getCurrent();
	FluidModel* model = sim->getFluidModel(fluidModelIndex);
	const unsigned int numParticles = model->numActiveParticles();

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < (int)numParticles; i++)
		{
			const unsigned int i0 = m_simulationData.getCurrentToInitialIndex(fluidModelIndex, i);
			const Vector3r& xi = model->getPosition(i);
			const Vector3r& xi0 = model->getPosition0(i0);
			Matrix3r F;
			F.setZero();

			std::vector<unsigned int>& initialNeighbors = m_simulationData.getInitialNeighbors(fluidModelIndex, i0);
			const size_t numNeighbors = initialNeighbors.size();

			//////////////////////////////////////////////////////////////////////////
			// Fluid
			//////////////////////////////////////////////////////////////////////////
			for (unsigned int j = 0; j < numNeighbors; j++)
			{
				const unsigned int neighborIndex = m_simulationData.getInitialToCurrentIndex(fluidModelIndex, initialNeighbors[j]);
				// get initial neighbor index considering the current particle order 
				const unsigned int neighborIndex0 = initialNeighbors[j];

				const Vector3r& xj = model->getPosition(neighborIndex);
				const Vector3r& xj0 = model->getPosition0(neighborIndex0);
				const Vector3r xj_xi = xj - xi;
				const Vector3r xi_xj_0 = xi0 - xj0;
				const Vector3r correctedKernel = m_simulationData.getL(fluidModelIndex, i) * sim->gradW(xi_xj_0);
				F += m_simulationData.getRestVolume(fluidModelIndex, neighborIndex) * xj_xi * correctedKernel.transpose();
			}

			if (sim->is2DSimulation())
				F(2, 2) = 1.0;

			//  			Vector3r sigma; 
			//  			Matrix3r U, VT;
			//  			MathFunctions::svdWithInversionHandling(F, sigma, U, VT);
			//  			m_rotations[i] = U * VT;
			Matrix3r& Ri = m_simulationData.getRotation(fluidModelIndex, i);
			Quaternionr q(Ri);
			MathFunctions::extractRotation(F, q, 10);
			Ri = q.matrix();

			m_simulationData.getRL(fluidModelIndex, i) = Ri * m_simulationData.getL(fluidModelIndex, i);
		}
	}
}

void TimeStepPIISPH::computeMatrixL(const unsigned int fluidModelIndex)
{
	Simulation* sim = Simulation::getCurrent();
	FluidModel* model = sim->getFluidModel(fluidModelIndex);
	const unsigned int numParticles = model->numActiveParticles();

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < (int)numParticles; i++)
		{
			const unsigned int i0 = m_simulationData.getCurrentToInitialIndex(fluidModelIndex, i);
			const Vector3r& xi0 = model->getPosition0(i0);
			Matrix3r L;
			L.setZero();

			std::vector<unsigned int>& initialNeighbors = m_simulationData.getInitialNeighbors(fluidModelIndex, i0);
			const size_t numNeighbors = initialNeighbors.size();

			//////////////////////////////////////////////////////////////////////////
			// Fluid
			//////////////////////////////////////////////////////////////////////////
			for (unsigned int j = 0; j < numNeighbors; j++)
			{
				const unsigned int neighborIndex = m_simulationData.getInitialToCurrentIndex(fluidModelIndex, initialNeighbors[j]);
				// get initial neighbor index considering the current particle order 
				const unsigned int neighborIndex0 = initialNeighbors[j];

				const Vector3r& xj0 = model->getPosition0(neighborIndex0);
				const Vector3r xj_xi_0 = xj0 - xi0;
				const Vector3r gradW = sim->gradW(xj_xi_0);

				// minus because gradW(xij0) == -gradW(xji0)
				L -= m_simulationData.getRestVolume(fluidModelIndex, neighborIndex) * gradW * xj_xi_0.transpose();
			}

			// add 1 to z-component. otherwise we get a singular matrix in 2D
			if (sim->is2DSimulation())
				L(2, 2) = 1.0;

			bool invertible = false;
			Matrix3r& L_res = m_simulationData.getL(fluidModelIndex, i);
			L.computeInverseWithCheck(L_res, invertible, 1e-9);
			if (!invertible)
			{
				//MathFunctions::pseudoInverse(L, L_res);
				L_res = Matrix3r::Identity();
			}
		}
	}
}

void TimeStepPIISPH::initValues()
{
	Simulation* sim = Simulation::getCurrent();
	sim->getNeighborhoodSearch()->find_neighbors();
	const unsigned int nFluids = sim->numberOfFluidModels();
	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
	{
		FluidModel* model = sim->getFluidModel(fluidModelIndex);
		const unsigned int numParticles = model->numActiveParticles();

		// Store the neighbors in the reference configurations and
		// compute the volume of each particle in rest state
#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static) 
			for (int i = 0; i < (int)numParticles; i++)
			{
				m_simulationData.getCurrentToInitialIndex(fluidModelIndex, i) = i;
				m_simulationData.getInitialToCurrentIndex(fluidModelIndex, i) = i;

				if (model->getParticleState(i) == ParticleState::Fixed)
					model->setParticleState(i, ParticleState::Active);

				// only neighbors in same phase will influence elasticity
				const unsigned int numNeighbors = sim->numberOfNeighbors(fluidModelIndex, fluidModelIndex, i);
				std::vector<unsigned int>& initialNeighbors = m_simulationData.getInitialNeighbors(fluidModelIndex, i);
				initialNeighbors.resize(numNeighbors);
				for (unsigned int j = 0; j < numNeighbors; j++)
					initialNeighbors[j] = sim->getNeighbor(fluidModelIndex, fluidModelIndex, i, j);

				// compute volume
				Real density = model->getMass(i) * sim->W_zero();
				const Vector3r& xi = model->getPosition(i);
				for (size_t j = 0; j < initialNeighbors.size(); j++)
				{
					const unsigned int neighborIndex = initialNeighbors[j];
					const Vector3r& xj = model->getPosition(neighborIndex);
					density += model->getMass(neighborIndex) * sim->W(xi - xj);
				}

				m_simulationData.getRestVolume(fluidModelIndex, i) = model->getMass(i) / density;
				m_simulationData.getRotation(fluidModelIndex, i).setIdentity();
				m_simulationData.setSaturation(fluidModelIndex,i,static_cast<Real>(0.0));
				if (m_materials[fluidModelIndex].m_isFluid)
					m_simulationData.setPorosity(fluidModelIndex,i,static_cast<Real>(1.0));
				else
					m_simulationData.setPorosity(fluidModelIndex,i,m_materials[fluidModelIndex].m_porosity);
			}
		}

		computeMatrixL(fluidModelIndex);
	}

	// mark all particles in the bounding box as fixed
	determineFixedParticles();
}

/** Mark all particles in the bounding box as fixed.
*/
void TimeStepPIISPH::determineFixedParticles()
{
	Simulation* sim = Simulation::getCurrent();
	const unsigned int nFluids = sim->numberOfFluidModels();
	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
	{
		FluidModel* model = sim->getFluidModel(fluidModelIndex);
		const unsigned int numParticles = model->numActiveParticles();
		if (!m_materials[fluidModelIndex].m_fixedBoxMin.isZero() || !m_materials[fluidModelIndex].m_fixedBoxMax.isZero())
		{
			for (int i = 0; i < (int)numParticles; i++)
			{
				const Vector3r& x = model->getPosition0(i);
				if ((x[0] > m_materials[fluidModelIndex].m_fixedBoxMin[0]) && (x[1] > m_materials[fluidModelIndex].m_fixedBoxMin[1]) && (x[2] > m_materials[fluidModelIndex].m_fixedBoxMin[2]) &&
					(x[0] < m_materials[fluidModelIndex].m_fixedBoxMax[0]) && (x[1] < m_materials[fluidModelIndex].m_fixedBoxMax[1]) && (x[2] < m_materials[fluidModelIndex].m_fixedBoxMax[2]))
				{
					model->setParticleState(i, ParticleState::Fixed);
				}
			}
		}
	}
}

void TimeStepPIISPH::computeDFSPHFactor(const unsigned int fluidModelIndex)
{
	//////////////////////////////////////////////////////////////////////////
	// Init parameters
	//////////////////////////////////////////////////////////////////////////

	Simulation *sim = Simulation::getCurrent();
	const unsigned int nFluids = sim->numberOfFluidModels();
	const unsigned int nBoundaries = sim->numberOfBoundaryModels();
	FluidModel *model = sim->getFluidModel(fluidModelIndex);
	const int numParticles = (int) model->numActiveParticles();
	PIISPHMaterialParameterObject* material = &m_materials[fluidModelIndex];

	#pragma omp parallel default(shared)
	{
		//////////////////////////////////////////////////////////////////////////
		// Compute pressure stiffness denominator
		//////////////////////////////////////////////////////////////////////////

		#pragma omp for schedule(static)  
		for (int i = 0; i < numParticles; i++)
		{
			//////////////////////////////////////////////////////////////////////////
			// Compute gradient dp_i/dx_j * (1/kappa)  and dp_j/dx_j * (1/kappa)
			// (see Equation (8) and the previous one [BK17])
			// Note: That in all quantities rho0 is missing due to our
			// implementation of multiphase simulations.
			//////////////////////////////////////////////////////////////////////////
			const Vector3r &xi = model->getPosition(i);
			Real sum_grad_p_k = 0.0;
			Vector3r grad_p_i;
			grad_p_i.setZero();
			const unsigned int obj_i = model->getObjectId(i);

			//////////////////////////////////////////////////////////////////////////
			// Fluid
			//////////////////////////////////////////////////////////////////////////
			forall_fluid_neighbors(
				const Real factor = getVolumeFactor(material, &m_materials[pid], obj_i, fm_neighbor->getObjectId(neighborIndex));
				const Vector3r grad_p_j = -factor * fm_neighbor->getVolume(neighborIndex) * sim->gradW(xi - xj);
				sum_grad_p_k += grad_p_j.squaredNorm();
				grad_p_i -= grad_p_j;
			);
			
			//////////////////////////////////////////////////////////////////////////
			// Boundary
			//////////////////////////////////////////////////////////////////////////
			if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
			{
				forall_boundary_neighbors(
					const Vector3r grad_p_j = -bm_neighbor->getVolume(neighborIndex) * sim->gradW(xi - xj);
					grad_p_i -= grad_p_j;
				);
			}

			else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Koschier2017)
			{
				forall_density_maps(
					grad_p_i -= gradRho;
				);
			}
			else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Bender2019)
			{
				forall_volume_maps(
					const Vector3r grad_p_j = -Vj * sim->gradW(xi - xj);
					grad_p_i -= grad_p_j;
				);
			}		

			sum_grad_p_k += grad_p_i.squaredNorm();

			//////////////////////////////////////////////////////////////////////////
			// Compute factor as: factor_i = -1 / (a_ii * rho_i^2)
			// where a_ii is the diagonal entry of the linear system 
			// for the pressure A * p = source term
			//////////////////////////////////////////////////////////////////////////
			Real &factor = m_simulationData.getFactor(fluidModelIndex, i);
			if (sum_grad_p_k > m_eps)
				factor = static_cast<Real>(1.0) / (sum_grad_p_k);
			else
				factor = 0.0;
		}
	}
}

/** Compute rho_adv,i^(0) (see equation in Section 3.3 in [BK17])
  * using the velocities after the non-pressure forces were applied.
**/
void TimeStepPIISPH::computeDensityAdv(const unsigned int fluidModelIndex, const unsigned int i, const Real h, const Real density0)
{
	Simulation *sim = Simulation::getCurrent();
	FluidModel *model = sim->getFluidModel(fluidModelIndex);
	const Real &density = model->getDensity(i);
	Real &densityAdv = m_simulationData.getDensityAdv(fluidModelIndex, i);
	const Vector3r &xi = model->getPosition(i);
	const Vector3r &vi = model->getVelocity(i);
	Real delta = 0.0;
	const unsigned int nFluids = sim->numberOfFluidModels();
	const unsigned int nBoundaries = sim->numberOfBoundaryModels();
	PIISPHMaterialParameterObject* material = &m_materials[fluidModelIndex];
	const unsigned int obj_i = model->getObjectId(i);

	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	forall_fluid_neighbors(
		const Vector3r & vj = fm_neighbor->getVelocity(neighborIndex);
		const Real factor = getVolumeFactor(material, &m_materials[pid], obj_i, fm_neighbor->getObjectId(neighborIndex));
		delta += factor * fm_neighbor->getVolume(neighborIndex) * (vi - vj).dot(sim->gradW(xi - xj));
	);

	//////////////////////////////////////////////////////////////////////////
	// Boundary
	//////////////////////////////////////////////////////////////////////////
	if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
	{
		forall_boundary_neighbors(
			const Vector3r &vj = bm_neighbor->getVelocity(neighborIndex);
			delta += bm_neighbor->getVolume(neighborIndex) * (vi - vj).dot(sim->gradW(xi - xj));
		);
	}
	else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Koschier2017)
	{
		forall_density_maps(
			Vector3r vj;
			bm_neighbor->getPointVelocity(xi, vj);
			delta -= (vi - vj).dot(gradRho);
		);
	}
	else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Bender2019)
	{
		forall_volume_maps(
			Vector3r vj;
			bm_neighbor->getPointVelocity(xj, vj);
			delta += Vj * (vi - vj).dot(sim->gradW(xi - xj));
		);
	}

	densityAdv = density / density0 + h*delta;
}

/** Compute rho_adv,i^(0) (see equation (9) in Section 3.2 [BK17])
  * using the velocities after the non-pressure forces were applied.
  */
void TimeStepPIISPH::computeDensityChange(const unsigned int fluidModelIndex, const unsigned int i, const Real h)
{
	Simulation *sim = Simulation::getCurrent();
	FluidModel *model = sim->getFluidModel(fluidModelIndex);
	Real &densityAdv = m_simulationData.getDensityAdv(fluidModelIndex, i);
	const Vector3r &xi = model->getPosition(i);
	const Vector3r& vi = model->getVelocity(i);
	densityAdv = 0.0;
	unsigned int numNeighbors = 0;
	const unsigned int nFluids = sim->numberOfFluidModels();
	const unsigned int nBoundaries = sim->numberOfBoundaryModels();
	PIISPHMaterialParameterObject* material = &m_materials[fluidModelIndex];
	const unsigned int obj_i = model->getObjectId(i);

	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	forall_fluid_neighbors(
		const Vector3r & vj = fm_neighbor->getVelocity(neighborIndex);
		const Real factor = getVolumeFactor(material, &m_materials[pid], obj_i, fm_neighbor->getObjectId(neighborIndex));
		densityAdv += factor * fm_neighbor->getVolume(neighborIndex) * (vi - vj).dot(sim->gradW(xi - xj));
	);

	//////////////////////////////////////////////////////////////////////////
	// Boundary
	//////////////////////////////////////////////////////////////////////////
	if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
	{
		forall_boundary_neighbors(
			const Vector3r &vj = bm_neighbor->getVelocity(neighborIndex);
			densityAdv += bm_neighbor->getVolume(neighborIndex) * (vi - vj).dot(sim->gradW(xi - xj));
		);
	}
	else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Koschier2017)
	{
		forall_density_maps(
			Vector3r vj;
			bm_neighbor->getPointVelocity(xi, vj);
			densityAdv -= (vi - vj).dot(gradRho);
		);
	}
	else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Bender2019)
	{
		forall_volume_maps(
			Vector3r vj;
			bm_neighbor->getPointVelocity(xj, vj);
			densityAdv += Vj * (vi - vj).dot(sim->gradW(xi - xj));
		);
	}
}

/** Compute pressure accelerations using the current pressure values of the particles
 */
void TimeStepPIISPH::computePressureAccel(const unsigned int fluidModelIndex, const unsigned int i, const Real density0, std::vector<std::vector<Real>>& pressure_rho2, const bool applyBoundaryForces)
{
	Simulation* sim = Simulation::getCurrent();
	FluidModel* model = sim->getFluidModel(fluidModelIndex);
	const unsigned int nFluids = sim->numberOfFluidModels();
	const unsigned int nBoundaries = sim->numberOfBoundaryModels();
	PIISPHMaterialParameterObject* material = &m_materials[fluidModelIndex];

	Vector3r& ai = m_simulationData.getPressureAccel(fluidModelIndex, i);
	ai.setZero();

	if (model->getParticleState(i) != ParticleState::Active)
		return;

	// p_rho2_i = (p_i / rho_i^2)
	const Real p_rho2_i = pressure_rho2[fluidModelIndex][i];
	const Vector3r &xi = model->getPosition(i);
	const unsigned int obj_i = model->getObjectId(i);

	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	forall_fluid_neighbors(			
		// p_rho2_j = (p_j / rho_j^2)
		const Real p_rho2_j = pressure_rho2[pid][neighborIndex];
		const Real pSum = p_rho2_i + getVolumeFactor(&m_materials[pid], material, fm_neighbor->getObjectId(neighborIndex), obj_i) * fm_neighbor->getDensity0()/density0 * p_rho2_j;
		if (fabs(pSum) > m_eps)
		{
			const Real factor = getVolumeFactor(material, &m_materials[pid], obj_i, fm_neighbor->getObjectId(neighborIndex));
			const Vector3r grad_p_j = -factor * fm_neighbor->getVolume(neighborIndex) * sim->gradW(xi - xj);
			ai += pSum * grad_p_j;
		}
	)

	//////////////////////////////////////////////////////////////////////////
	// Boundary
	//////////////////////////////////////////////////////////////////////////
	if (fabs(p_rho2_i) > m_eps)
	{
		if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
		{
			forall_boundary_neighbors(
				const Vector3r grad_p_j = -bm_neighbor->getVolume(neighborIndex) * sim->gradW(xi - xj);

				const Vector3r a = (Real) 1.0 * p_rho2_i * grad_p_j;		
				ai += a;
				if (applyBoundaryForces)
					bm_neighbor->addForce(xj, -model->getMass(i) * a);
			);
		}
		else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Koschier2017)
		{
			forall_density_maps(
				const Vector3r a = (Real) 1.0 * p_rho2_i * gradRho;			
				ai += a;
				if (applyBoundaryForces)
					bm_neighbor->addForce(xj, -model->getMass(i) * a);
			);
		}
		else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Bender2019)
		{
			forall_volume_maps(
				const Vector3r grad_p_j = -Vj * sim->gradW(xi - xj);
				const Vector3r a = (Real) 1.0 * p_rho2_i * grad_p_j;		
				ai += a;

				if (applyBoundaryForces)
					bm_neighbor->addForce(xj, -model->getMass(i) * a);  
			);
		}
	}
}


Real TimeStepPIISPH::compute_aij_pj(const unsigned int fluidModelIndex, const unsigned int i)
{
	Simulation* sim = Simulation::getCurrent();
	FluidModel* model = sim->getFluidModel(fluidModelIndex);
	const unsigned int nFluids = sim->numberOfFluidModels();
	const unsigned int nBoundaries = sim->numberOfBoundaryModels();
	PIISPHMaterialParameterObject* material = &m_materials[fluidModelIndex];

	//////////////////////////////////////////////////////////////////////////
	// Compute A*p which is the change of the density when applying the 
	// pressure forces. 
	// \sum_j a_ij * p_j = h^2 \sum_j V_j (a_i - a_j) * gradW_ij
	// This is the RHS of Equation (12) in [BK17]
	//////////////////////////////////////////////////////////////////////////
	const Vector3r& xi = model->getPosition(i);
	const Vector3r& ai = m_simulationData.getPressureAccel(fluidModelIndex, i);
	Real aij_pj = 0.0;
	const unsigned int obj_i = model->getObjectId(i);

	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	forall_fluid_neighbors(
		const Vector3r & aj = m_simulationData.getPressureAccel(pid, neighborIndex);
		const Real factor = getVolumeFactor(material, &m_materials[pid], obj_i, fm_neighbor->getObjectId(neighborIndex));
		aij_pj += factor * fm_neighbor->getVolume(neighborIndex) * (ai - aj).dot(sim->gradW(xi - xj));
	);

	//////////////////////////////////////////////////////////////////////////
	// Boundary
	//////////////////////////////////////////////////////////////////////////
	if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
	{
		forall_boundary_neighbors(
			aij_pj += bm_neighbor->getVolume(neighborIndex) * ai.dot(sim->gradW(xi - xj));
		);
	}
	else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Koschier2017)
	{
		forall_density_maps(
			aij_pj -= ai.dot(gradRho);
		);
	}
	else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Bender2019)
	{
		forall_volume_maps(
			aij_pj += Vj * ai.dot(sim->gradW(xi - xj));
		);
	}
	return aij_pj;
}
